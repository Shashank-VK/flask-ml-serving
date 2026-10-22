"""Thread-safe in-process cache for model predictions."""

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class PredictionCache:
    """In-process LRU cache keyed on canonical feature tuples."""

    def __init__(self, maxsize: int = 1024):
        self._maxsize = maxsize
        self._cache: OrderedDict[Tuple[float, ...], Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Tuple[float, ...]) -> Optional[Dict[str, Any]]:
        """Retrieve cached prediction result for the given feature tuple."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return dict(self._cache[key])
            return None

    def set(self, key: Tuple[float, ...], value: Dict[str, Any]) -> None:
        """Store prediction result in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = dict(value)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
