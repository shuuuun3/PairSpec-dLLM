
import argparse
import logging
import time
import queue
from typing import Dict, Optional, Tuple

import torch
import types
from transformers import AutoTokenizer

from model.generation_utils_block import DreamGenerationMixin
from model.modeling_dream import DreamModel
from specdraft.acceptor import compute_prefix_hash, verify_and_commit
from specdraft.dispatcher import DraftRequest, start_draft_worker, shutdown_draft_worker
from specdraft.kv_manager import KVManager

LOGGER = logging.getLogger(__name__)


def _generator_name(args) -> str:
    if not args.use_cache:
        return "vanilla"
    return "dual_cache" if args.dual_cache else "prefix_cache"


def _dream_generate_block(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args,
    *,
    block_size: int,
    steps_per_block: int,
) -> Tuple[torch.Tensor, int]:
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=block_size,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps_per_block,
        temperature=args.temperature,
        top_p=args.top_p,
        alg=args.alg,
        alg_temp=args.alg_temp,
        top_k=args.top_k,
        block_length=block_size,
        threshold=args.threshold,
    )
    sequences = output.sequences
    block_tensor = sequences[:, -block_size:]
    # Empirically `diffusion_generate` performs `steps_per_block` forward passes.
    return block_tensor, steps_per_block


def _dream_pairspec_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, int, Dict[str, float]]:
    device = input_ids.device
    block_size = args.block_length
    gen_length = args.max_new_tokens
    if gen_length % block_size != 0:
        raise ValueError("--max_new_tokens must be divisible by --block_length when using --pairspec")
    total_blocks = gen_length // block_size
    steps_per_block = max(1, args.steps // max(total_blocks, 1))
    kv_mgr = KVManager(model)
    prefix_hash = compute_prefix_hash(input_ids[0].tolist())
    policy = args.accept_policy.lower()
    stats = {
        "accepted_blocks": 0,
        "attempted_blocks": total_blocks,
        "verify_nfe": 0,
        "fallback_nfe": 0,
        "draft_nfe": 0,
    }
    worker = None
    request_queue = None
    spec_queue = None
    pending_blocks = set()
    generator_name = _generator_name(args)

    if not args.draft_model:
        raise ValueError("--draft_model must be provided when enabling PairSpec")

    try:
        worker, request_queue, spec_queue = start_draft_worker(
            backend="dream",
            model_path=args.draft_model,
            device=args.draft_device,
            generator=generator_name,
            max_depth=args.draft_depth,
            dtype=torch.bfloat16,
        )

        def enqueue(block_id: int, current_ids: torch.Tensor, current_hash: str) -> None:
            if block_id >= total_blocks or block_id in pending_blocks:
                return
            req = DraftRequest(
                block_id=block_id,
                prefix_tokens=current_ids[0].tolist(),
                prefix_hash=current_hash,
                block_size=block_size,
                steps_per_block=max(1, args.draft_steps or steps_per_block),
                temperature=args.temperature,
                remasking="low_confidence",
                threshold=args.threshold,
                generator=generator_name,
            )
            try:
                request_queue.put_nowait(req)
                pending_blocks.add(block_id)
            except queue.Full:
                LOGGER.debug('Draft queue full; skip block %s', block_id)

        enqueue(0, input_ids, prefix_hash)
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        generated_segments = []

        mask_id = tokenizer.mask_token_id
        if mask_id is None and tokenizer.mask_token:
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        if mask_id is None:
            raise ValueError("Tokenizer must provide a mask_token_id for PairSpec verification.")

        for block_id in range(total_blocks):
            draft = spec_queue.try_get(block_id) if spec_queue else None
            if draft is not None:
                pending_blocks.discard(block_id)
                stats["draft_nfe"] += draft.nfe
                if draft.prefix_hash != prefix_hash:
                    LOGGER.debug(
                        "Ignoring stale Dream draft for block %s (%s != %s)",
                        block_id,
                        draft.prefix_hash,
                        prefix_hash,
                    )
                    draft = None

            accepted_tensor: Optional[torch.Tensor] = None
            if draft is not None:
                verification = verify_and_commit(
                    draft,
                    model,
                    current_ids,
                    mask_token_id=mask_id,
                    policy=policy,
                    threshold=args.accept_threshold,
                    model_kwargs={"attention_mask": torch.ones_like(current_ids)},
                )
                stats["verify_nfe"] += verification.nfe
                if verification.accepted:
                    stats["accepted_blocks"] += 1
                    accepted_tensor = torch.tensor(
                        verification.accepted_tokens,
                        dtype=current_ids.dtype,
                        device=device,
                    ).unsqueeze(0)
                    LOGGER.info("Dream PairSpec accepted block %s", block_id)

            if accepted_tensor is None:
                block_tensor, nfe_block = _dream_generate_block(
                    model,
                    current_ids,
                    current_mask,
                    args,
                    block_size=block_size,
                    steps_per_block=steps_per_block,
                )
                stats["fallback_nfe"] += nfe_block
            else:
                block_tensor = accepted_tensor

            generated_segments.append(block_tensor)
            pad_mask = torch.ones((current_mask.shape[0], block_size), dtype=current_mask.dtype, device=device)
            current_mask = torch.cat([current_mask, pad_mask], dim=1)
            current_ids = torch.cat([current_ids, block_tensor], dim=1)
            prefix_hash = compute_prefix_hash(current_ids[0].tolist())
            kv_mgr.recompute_on_commit(prefix_hash)
            enqueue(block_id + 1, current_ids, prefix_hash)

        full = torch.cat(generated_segments, dim=1) if generated_segments else torch.empty((1, 0), dtype=input_ids.dtype, device=device)
        total_nfe = stats["verify_nfe"] + stats["fallback_nfe"]
        stats["total_nfe"] = total_nfe
        return full, total_nfe, stats
    finally:
        shutdown_draft_worker(worker, request_queue)
        if spec_queue:
            spec_queue.close()


def load_model(args):
    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = model.to(args.device).eval()
    if args.use_cache:
        model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
        model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Dream multi-turn chat with optional PairSpec drafting.")
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dual_cache", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--pairspec", action="store_true")
    parser.add_argument("--draft_model", type=str, default=None)
    parser.add_argument("--draft_device", type=str, default="cuda:1")
    parser.add_argument("--draft_depth", type=int, default=2)
    parser.add_argument("--draft_steps", type=int, default=None)
    parser.add_argument("--accept_policy", type=str, choices=["lossless", "thresholded"], default="lossless")
    parser.add_argument("--accept_threshold", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    model, tokenizer = load_model(args)

    messages = []
    print(f"Multi-turn conversation with {args.model_path}")
    print("Type 'exit' to end the conversation")
    print("----------------------------------------------")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Conversation ended.")
            break

        messages.append({"role": "user", "content": user_input})
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device=args.device)
        attention_mask = inputs.attention_mask.to(device=args.device)

        start = time.time()
        if args.pairspec:
            new_tokens, nfe, stats = _dream_pairspec_generate(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                args,
            )
            sequences = torch.cat([input_ids, new_tokens], dim=1)
            block_msg = f"accepted_blocks={stats['accepted_blocks']}/{stats['attempted_blocks']}"
        else:
            sequences = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                output_history=False,
                return_dict_in_generate=False,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=args.alg_temp,
                top_k=args.top_k,
                block_length=args.block_length,
                threshold=args.threshold,
            )
            if isinstance(sequences, torch.Tensor):
                output_ids = sequences
            else:
                output_ids = sequences.sequences
            new_tokens = output_ids[:, input_ids.shape[1]:]
            nfe = args.steps
            sequences = output_ids
            block_msg = "accepted_blocks=0"
        latency = time.time() - start

        generation = tokenizer.decode(new_tokens[0].tolist())
        generation = generation.split(tokenizer.eos_token)[0].strip()
        print(f"Model: {generation}")
        print(f"NFE (est.): {nfe} ({block_msg}) | Latency: {latency:.2f}s")

        messages.append({"role": "assistant", "content": generation})


if __name__ == "__main__":
    main()
