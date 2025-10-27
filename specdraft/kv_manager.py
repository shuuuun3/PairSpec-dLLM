
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KVManager:
    """
    Lightweight helper that keeps cache invalidation in one place. Fast-dLLM
    requires that KV caches are recomputed whenever we accept a speculative
    block to guarantee consistency across block boundaries.
    """

    def __init__(self, model) -> None:
        self.model = model
        self._current_hash: Optional[str] = None

    def _maybe_call(self, target, attr: str) -> bool:
        fn = getattr(target, attr, None)
        if callable(fn):
            fn()
            return True
        return False

    def _clear_internal(self) -> None:
        candidates = (
            "clear_cache",
            "reset_cache",
            "clear_kv_cache",
            "reset_kv_cache",
        )
        if any(self._maybe_call(self.model, attr) for attr in candidates):
            return
        inner = getattr(self.model, "model", None)
        if inner is not None and any(self._maybe_call(inner, attr) for attr in candidates):
            return
        # Fall back to logging only. Some diffusion models keep caches in module
        # level globals, so we simply log the event.
        logger.debug("KVManager: no explicit cache reset hook found on %s", type(self.model))

    def recompute_on_commit(self, prefix_hash: Optional[str] = None) -> None:
        if prefix_hash is not None:
            self._current_hash = prefix_hash
        self._clear_internal()

    def invalidate_if_prefix_changed(self, new_hash: str) -> None:
        if self._current_hash is None:
            self._current_hash = new_hash
            return
        if new_hash != self._current_hash:
            logger.debug(
                "KVManager: prefix hash changed (%s -> %s), forcing cache reset.",
                self._current_hash,
                new_hash,
            )
            self._current_hash = new_hash
            self._clear_internal()
