
import logging
import multiprocessing as mp
import queue
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


def _get_mp_context():
    """
    Resolve a spawn-friendly multiprocessing context. We always prefer spawn to
    avoid CUDA handle sharing issues between the verifier and drafter.
    """
    try:
        return mp.get_context("spawn")
    except ValueError:
        return mp.get_context()


@dataclass
class DraftRequest:
    """
    Control packet sent from the verifier process to the drafter worker.
    """

    block_id: int
    prefix_tokens: Sequence[int]
    prefix_hash: str
    block_size: int
    steps_per_block: int
    temperature: float = 0.0
    remasking: str = "low_confidence"
    threshold: Optional[float] = None
    generator: str = "vanilla"


@dataclass
class DraftResult:
    """
    Data packet produced by the drafter worker.
    """

    block_id: int
    prefix_hash: str
    tokens: List[int] = field(default_factory=list)
    nfe: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpecQueue:
    """
    Thin ring buffer that keeps the most recent draft results per block id.

    Draft workers push results through a multiprocessing Queue. The verifier
    drains the transport queue and keeps at most `max_depth` outstanding blocks
    in memory, evicting older entries automatically.
    """

    def __init__(
        self,
        max_depth: int,
        transport_queue: Optional[mp.Queue] = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be >= 1")
        self._max_depth = max_depth
        self._ctx = _get_mp_context()
        self._transport = transport_queue or self._ctx.Queue(max_depth * 2)
        self._buffer: Dict[int, DraftResult] = {}
        self._lock = threading.Lock()

    @property
    def transport(self) -> mp.Queue:
        return self._transport

    def _drain(self) -> None:
        while True:
            try:
                result: DraftResult = self._transport.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._buffer[result.block_id] = result
                # Keep buffer bounded
                while len(self._buffer) > self._max_depth:
                    oldest_block = min(self._buffer)
                    del self._buffer[oldest_block]

    def try_get(self, block_id: int) -> Optional[DraftResult]:
        self._drain()
        with self._lock:
            return self._buffer.pop(block_id, None)

    def pending_blocks(self) -> List[int]:
        self._drain()
        with self._lock:
            return list(self._buffer.keys())

    def close(self) -> None:
        with suppress(Exception):
            self._transport.close()


def _select_llada_generator(generator: str):
    from llada.generate import (
        generate,
        generate_with_dual_cache,
        generate_with_prefix_cache,
    )

    if generator == "dual_cache":
        return generate_with_dual_cache
    if generator == "prefix_cache":
        return generate_with_prefix_cache
    return generate


class DraftWorker(mp.Process):
    """
    Background process that continuously drafts future blocks using a lighter
    diffusion LLM. Communication happens through request/result queues.
    """

    def __init__(
        self,
        *,
        backend: str,
        model_path: str,
        device: str,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
        generator: str = "vanilla",
        tokenizer_path: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(daemon=True)
        self.backend = backend
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = torch.device(device)
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.generator = generator
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _load_llada(self) -> None:
        from transformers import AutoTokenizer
        from llada.model.modeling_llada import LLaDAModelLM

        logger.info("DraftWorker: loading LLaDA drafter from %s", self.model_path)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, trust_remote_code=True
            )
            self._model = (
                LLaDAModelLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=self.dtype,
                )
                .to(self.device)
                .eval()
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load LLaDA drafter. Ensure the draft model is compatible with llada.model LLaDAModelLM, or "
                "provide a custom --draft_backend."
            ) from exc

    def _load_dream(self) -> None:
        from transformers import AutoTokenizer
        from dream.model.modeling_dream import DreamModel
        from dream.model.generation_utils_block import DreamGenerationMixin
        import types

        logger.info("DraftWorker: loading Dream drafter from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )
        model = DreamModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        model = model.eval()
        # Attach block generation helpers to mimic the chat demo behaviour.
        model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
        model._sample = types.MethodType(DreamGenerationMixin._sample, model)
        self._model = model

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        torch.set_grad_enabled(False)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if self.backend == "dream":
            self._load_dream()
        else:
            self._load_llada()

    def _run_llada_block(self, request: DraftRequest) -> Tuple[List[int], int]:
        generator_fn = _select_llada_generator(self.generator)
        prompt = torch.tensor(request.prefix_tokens, device=self.device).unsqueeze(0)
        out, nfe = generator_fn(
            self._model,
            prompt,
            steps=request.steps_per_block,
            gen_length=request.block_size,
            block_length=request.block_size,
            temperature=request.temperature,
            remasking=request.remasking,
            threshold=request.threshold,
        )
        block_tokens = out[0, -request.block_size :].detach().to("cpu").tolist()
        return block_tokens, int(nfe)

    def _run_dream_block(self, request: DraftRequest) -> Tuple[List[int], int]:
        attention_mask = torch.ones(
            (1, len(request.prefix_tokens)), dtype=torch.long, device=self.device
        )
        prompt = torch.tensor(request.prefix_tokens, device=self.device).unsqueeze(0)
        output = self._model.diffusion_generate(
            prompt,
            attention_mask=attention_mask,
            max_new_tokens=request.block_size,
            output_history=False,
            return_dict_in_generate=True,
            steps=request.steps_per_block,
            temperature=request.temperature,
            block_length=request.block_size,
            remasking=request.remasking,
            threshold=request.threshold,
        )
        seq = output.sequences[0].detach().to("cpu").tolist()
        block_tokens = seq[-request.block_size :]
        # Dream's diffusion_generate currently returns history as `logits` length steps;
        # we use sequence length as nfe proxy.
        nfe = request.steps_per_block
        return block_tokens, nfe

    def _process_request(self, request: DraftRequest) -> DraftResult:
        if self.backend == "dream":
            tokens, nfe = self._run_dream_block(request)
        else:
            tokens, nfe = self._run_llada_block(request)
        return DraftResult(
            block_id=request.block_id,
            prefix_hash=request.prefix_hash,
            tokens=tokens,
            nfe=nfe,
        )

    def run(self) -> None:
        try:
            self._ensure_model()
            while True:
                request = self.request_queue.get()
                if request is None:
                    logger.info("DraftWorker shutting down gracefully.")
                    break
                try:
                    result = self._process_request(request)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("DraftWorker error on block %s: %s", request.block_id, exc)
                    result = DraftResult(
                        block_id=request.block_id,
                        prefix_hash=request.prefix_hash,
                        tokens=[],
                        nfe=0,
                        metadata={"error": str(exc)},
                    )
                self.result_queue.put(result)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            logger.info("DraftWorker interrupted, terminating.")
        finally:
            with suppress(Exception):
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()


def start_draft_worker(
    *,
    backend: str,
    model_path: str,
    device: str,
    generator: str,
    max_depth: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[DraftWorker, mp.Queue, SpecQueue]:
    """
    Helper to bootstrap the drafting runtime. Returns the worker handle, request
    queue, and SpecQueue where results can be consumed.
    """

    ctx = _get_mp_context()
    request_queue = ctx.Queue(max_depth * 2)
    result_queue = ctx.Queue(max_depth * 2)
    spec_queue = SpecQueue(max_depth=max_depth, transport_queue=result_queue)
    worker = DraftWorker(
        backend=backend,
        model_path=model_path,
        device=device,
        request_queue=request_queue,
        result_queue=result_queue,
        generator=generator,
        dtype=dtype,
    )
    worker.start()
    return worker, request_queue, spec_queue


def shutdown_draft_worker(worker: Optional[DraftWorker], request_queue: Optional[mp.Queue]) -> None:
    """
    Stop the drafter process and drain any outstanding IPC handles.
    """

    if worker is None or request_queue is None:
        return
    with suppress(Exception):
        request_queue.put_nowait(None)
    worker.join(timeout=5)
    if worker.is_alive():  # pragma: no cover - defensive
        logger.warning("DraftWorker did not exit cleanly, terminating.")
        worker.terminate()
