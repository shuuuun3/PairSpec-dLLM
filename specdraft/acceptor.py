
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_prefix_hash(tokens: Sequence[int]) -> str:
    """
    Stable hash for prefix tokens. We rely on cryptographic hashing to avoid
    collisions when multiple drafter processes race on different prefixes.
    """

    hasher = hashlib.blake2b(digest_size=16)
    payload = ",".join(str(t) for t in tokens).encode("utf-8", errors="ignore")
    hasher.update(payload)
    return hasher.hexdigest()


@dataclass
class VerificationResult:
    accepted_tokens: List[int] = field(default_factory=list)
    accepted: bool = False
    nfe: int = 0
    reason: str = ""
    margins: List[float] = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)


def verify_and_commit(
    draft_result,
    model,
    prefix: torch.Tensor,
    *,
    mask_token_id: int,
    policy: str = "lossless",
    threshold: float = 2.0,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> VerificationResult:
    """
    Compare a drafter proposal against the verifier model and return the block
    if every token satisfies the acceptance policy.
    """

    if draft_result is None or not draft_result.tokens:
        return VerificationResult(reason="empty_draft")

    candidate_tokens = draft_result.tokens
    prefix_ids = prefix.detach()
    if prefix_ids.dim() != 2:
        raise ValueError("prefix tensor must be 2D [batch, seq]")

    device = prefix_ids.device
    block_len = len(candidate_tokens)
    total_len = prefix_ids.shape[1] + block_len
    x = torch.full(
        (1, total_len),
        mask_token_id,
        dtype=prefix_ids.dtype,
        device=device,
    )
    x[:, : prefix_ids.shape[1]] = prefix_ids

    nfe = 0
    accepted_tokens: List[int] = []
    margins: List[float] = []
    forward_kwargs = dict(model_kwargs or {})
    attention_mask = forward_kwargs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = torch.ones_like(x)
        forward_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        for offset, token in enumerate(candidate_tokens):
            position = prefix_ids.shape[1] + offset
            outputs = model(x, **forward_kwargs)
            logits = outputs.logits
            nfe += 1
            token_logits = logits[:, position, :]
            log_probs = F.log_softmax(token_logits, dim=-1)
            top2 = torch.topk(log_probs, k=2, dim=-1)
            predicted_id = int(top2.indices[0, 0])
            margin = float(top2.values[0, 0] - top2.values[0, 1])
            margins.append(margin)

            if policy == "lossless":
                if predicted_id != token:
                    logger.debug(
                        "Rejecting block at offset %s due to mismatch %s!=%s",
                        offset,
                        predicted_id,
                        token,
                    )
                    return VerificationResult(
                        accepted_tokens=[],
                        accepted=False,
                        nfe=nfe,
                        reason="mismatch",
                        margins=margins,
                    )
            elif policy == "thresholded":
                if margin < threshold:
                    logger.debug(
                        "Rejecting block at offset %s due to low margin %.3f < %.3f",
                        offset,
                        margin,
                        threshold,
                    )
                    return VerificationResult(
                        accepted_tokens=[],
                        accepted=False,
                        nfe=nfe,
                        reason="low_margin",
                        margins=margins,
                        metadata={"last_margin": margin},
                    )
            else:
                raise ValueError(f"Unsupported acceptance policy: {policy}")

            accepted_tokens.append(token)
            x[:, position] = token

    return VerificationResult(
        accepted_tokens=accepted_tokens,
        accepted=True,
        nfe=nfe,
        reason="accepted",
        margins=margins,
    )
