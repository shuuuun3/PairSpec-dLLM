# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import argparse
import logging
import multiprocessing as mp
import queue
import time
from typing import Dict, Optional, Tuple

import torch

from .generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from .model.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer

from specdraft.acceptor import compute_prefix_hash, verify_and_commit
from specdraft.dispatcher import DraftRequest, start_draft_worker, shutdown_draft_worker
from specdraft.kv_manager import KVManager

LOGGER = logging.getLogger(__name__)


def _normalize_device(device_str: str) -> str:
    try:
        device = torch.device(device_str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid device string '{device_str}'") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available but a CUDA device was requested")

        device_count = torch.cuda.device_count()
        if device.index is None:
            index = 0
        else:
            index = device.index

        if index < 0 or index >= device_count:
            raise ValueError(
                f"CUDA device index {index} is out of range for this system (available: 0..{device_count - 1})"
            )
        return f"cuda:{index}"

    if device.type == "cpu":
        return "cpu"

    raise ValueError(f"unsupported device type '{device.type}'")


def _select_generator_name(args) -> str:
    if not args.use_cache:
        return "vanilla"
    return "dual_cache" if args.if_cache_position else "prefix_cache"


def _select_generator_fn(generator_name):
    if generator_name == "dual_cache":
        return generate_with_dual_cache
    if generator_name == "prefix_cache":
        return generate_with_prefix_cache
    return generate


def _generate_block(
    model,
    prompt: torch.Tensor,
    *,
    block_size: int,
    steps: int,
    temperature: float,
    threshold: Optional[float],
    generator_name: str,
) -> Tuple[torch.Tensor, int]:
    generator_fn = _select_generator_fn(generator_name)
    out, nfe = generator_fn(
        model,
        prompt,
        steps=steps,
        gen_length=block_size,
        block_length=block_size,
        temperature=temperature,
        remasking="low_confidence",
        threshold=threshold,
    )
    block_tensor = out[:, -block_size:]
    return block_tensor, int(nfe)


def _pairspec_generate(
    model,
    tokenizer,
    prompt: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, int, Dict[str, float]]:
    device = prompt.device
    block_size = args.block_size
    gen_length = args.gen_length
    if gen_length % block_size != 0:
        raise ValueError("--gen_length must be divisible by --block_size when using --pairspec")
    total_blocks = gen_length // block_size
    steps_per_block = max(1, args.steps // max(total_blocks, 1))
    generator_name = _select_generator_name(args)

    kv_mgr = KVManager(model)
    prefix_hash = compute_prefix_hash(prompt[0].tolist())
    worker = None
    request_queue = None
    spec_queue = None
    stats = {
        "accepted_blocks": 0,
        "attempted_blocks": total_blocks,
        "draft_nfe": 0,
        "verify_nfe": 0,
        "fallback_nfe": 0,
    }
    pending_blocks = set()

    if not args.draft_model:
        raise ValueError("--draft_model is required when --pairspec is enabled")

    try:
        worker, request_queue, spec_queue = start_draft_worker(
            backend="llada",
            model_path=args.draft_model,
            device=args.draft_device,
            generator=generator_name,
            max_depth=args.draft_depth,
            dtype=torch.bfloat16,
        )
        LOGGER.info(
            "Spawned drafter on %s with model %s (depth=%s)",
            args.draft_device,
            args.draft_model,
            args.draft_depth,
        )

        policy = args.accept_policy.lower()

        def enqueue(block_id: int, current_prompt: torch.Tensor, current_hash: str) -> None:
            if block_id >= total_blocks or block_id in pending_blocks:
                return
            req = DraftRequest(
                block_id=block_id,
                prefix_tokens=current_prompt[0].tolist(),
                prefix_hash=current_hash,
                block_size=block_size,
                steps_per_block=max(1, args.draft_steps or steps_per_block),
                temperature=0.0,
                remasking="low_confidence",
                threshold=args.threshold,
                generator=generator_name,
            )
            try:
                request_queue.put_nowait(req)
                pending_blocks.add(block_id)
            except queue.Full:
                LOGGER.debug('Draft queue full; skip block %s', block_id)

        enqueue(0, prompt, prefix_hash)

        generated_segments = []
        current_prompt = prompt.clone()

        for block_id in range(total_blocks):
            draft = spec_queue.try_get(block_id) if spec_queue else None
            if draft is not None:
                pending_blocks.discard(block_id)
                stats["draft_nfe"] += draft.nfe
                if draft.prefix_hash != prefix_hash:
                    LOGGER.debug(
                        "Discarding stale draft for block %s (hash %s != %s)",
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
                    current_prompt,
                    mask_token_id=tokenizer.mask_token_id or 126336,
                    policy=policy,
                    threshold=args.accept_threshold,
                )
                stats["verify_nfe"] += verification.nfe
                if verification.accepted:
                    stats["accepted_blocks"] += 1
                    block_list = verification.accepted_tokens
                    accepted_tensor = torch.tensor(block_list, dtype=current_prompt.dtype, device=device).unsqueeze(0)
                    LOGGER.info("Accepted draft block %s (%s tokens).", block_id, len(block_list))
                else:
                    LOGGER.debug(
                        "Rejected draft block %s due to %s", block_id, verification.reason
                    )

            if accepted_tensor is None:
                block_tensor, nfe_block = _generate_block(
                    model,
                    current_prompt,
                    block_size=block_size,
                    steps=steps_per_block,
                    temperature=0.0,
                    threshold=args.threshold,
                    generator_name=generator_name,
                )
                stats["fallback_nfe"] += nfe_block
            else:
                block_tensor = accepted_tensor

            current_prompt = torch.cat([current_prompt, block_tensor], dim=1)
            prefix_hash = compute_prefix_hash(current_prompt[0].tolist())
            kv_mgr.recompute_on_commit(prefix_hash)
            generated_segments.append(block_tensor)
            enqueue(block_id + 1, current_prompt, prefix_hash)

        full_generation = torch.cat(generated_segments, dim=1) if generated_segments else torch.empty((1, 0), dtype=prompt.dtype, device=device)
        total_nfe = stats["verify_nfe"] + stats["fallback_nfe"]
        stats["total_nfe"] = total_nfe
        return full_generation, total_nfe, stats
    finally:
        shutdown_draft_worker(worker, request_queue)
        if spec_queue:
            spec_queue.close()

def chat(args):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    device = 'cuda'
    if args.pairspec:
        try:
            args.draft_device = _normalize_device(args.draft_device)
        except ValueError as exc:
            raise SystemExit(f"Invalid --draft_device: {exc}")
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    gen_length = args.gen_length
    steps = args.steps
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
        print(f'use cache: {args.use_cache} use cache position: {args.if_cache_position} threshold: {args.threshold} block size: {args.block_size}')
        start_time = time.time()
        if args.pairspec:
            new_tokens, nfe, stats = _pairspec_generate(model, tokenizer, prompt, args)
            out = torch.cat([prompt, new_tokens], dim=1)
            answer_tokens = new_tokens
            block_msg = f"accepted_blocks={stats['accepted_blocks']}/{stats['attempted_blocks']}"
        else:
            generator_name = _select_generator_name(args)
            out, nfe = _select_generator_fn(generator_name)(
                model,
                prompt,
                steps=steps,
                gen_length=gen_length,
                block_length=args.block_size,
                temperature=0.,
                remasking='low_confidence',
                threshold=args.threshold,
            )
            answer_tokens = out[:, prompt.shape[1]:]
            stats = {}
            block_msg = "accepted_blocks=0"
        elapsed = time.time() - start_time

        answer = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")
        print(f"Number of forward passes: {nfe} ({block_msg})")
        print(f"Latency: {elapsed:.2f}s")

        # remove the <EOS>
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--if_cache_position", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--pairspec", action="store_true", help="Enable PairSpec-dLLM drafting/verification pipeline.")
    parser.add_argument("--draft_model", type=str, default=None, help="Tokenizer/model path for the drafter.")
    parser.add_argument("--draft_device", type=str, default="cuda:1", help="Device string for the drafter worker.")
    parser.add_argument("--draft_depth", type=int, default=2, help="Max number of speculative blocks to queue.")
    parser.add_argument("--draft_steps", type=int, default=None, help="Optional override for drafter diffusion steps per block.")
    parser.add_argument("--accept_policy", type=str, choices=["lossless", "thresholded"], default="lossless")
    parser.add_argument("--accept_threshold", type=float, default=2.0, help="Margin threshold for thresholded policy.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging.")

    args = parser.parse_args()
    chat(args)
