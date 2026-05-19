"""
cache.py
────────
Deterministic caching for drug pair evaluations.

Provides a unified asynchronous interface `PairCache` that backs onto Redis
if `REDIS_URL` is configured, or an in-memory LRU cache otherwise.
"""
import json
import logging
from collections import OrderedDict
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

class LocalLRUCache:
    """Simple thread-safe asyncio-compatible LRU cache."""
    def __init__(self, capacity: int = 10000):
        self._capacity = capacity
        self._cache: OrderedDict[str, str] = OrderedDict()

    async def get(self, key: str) -> str | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    async def set(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

class PairCache:
    """
    Unified caching interface for drug interaction pairs.
    """
    def __init__(self):
        self._redis = None
        self._local = None
        
        if settings.redis_url:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(settings.redis_url, decode_responses=True)
                logger.info("PairCache initialized with Redis: %s", settings.redis_url)
            except ImportError:
                logger.warning("redis package not installed. Falling back to LRU cache.")
                self._local = LocalLRUCache()
            except Exception as e:
                logger.error("Failed to connect to Redis: %s. Falling back to LRU cache.", e)
                self._local = LocalLRUCache()
        else:
            self._local = LocalLRUCache()
            logger.info("PairCache initialized with local LRU Cache")

    async def get(self, key: str) -> dict | None:
        """Fetch and deserialize a stored pair evaluation."""
        try:
            val = await self._redis.get(key) if self._redis else await self._local.get(key)
            if val:
                return json.loads(val)
            return None
        except Exception as e:
            logger.warning("Cache get error for %s: %s", key, e)
            return None

    async def set(self, key: str, data: dict) -> None:
        """Serialize and store a pair evaluation."""
        try:
            val = json.dumps(data)
            if self._redis:
                # 30 day expiration for Redis
                await self._redis.setex(key, 86400 * 30, val)
            else:
                await self._local.set(key, val)
        except Exception as e:
            logger.warning("Cache set error for %s: %s", key, e)

# Singleton export
cache = PairCache()
