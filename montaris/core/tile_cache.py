from collections import OrderedDict


class TileCache:
    """LRU cache for tile QImages. Max 200 tiles by default."""

    def __init__(self, max_size=200):
        self._cache = OrderedDict()
        self._max_size = max_size

    def get(self, key):
        """Return cached QImage for *key*, or None if not present.

        Accessing a key promotes it to the most-recently-used position.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, tile):
        """Store *tile* under *key*, evicting the oldest entry if full."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = tile
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = tile

    def clear(self):
        """Remove all cached tiles."""
        self._cache.clear()

    @property
    def size(self):
        """Number of tiles currently cached."""
        return len(self._cache)
