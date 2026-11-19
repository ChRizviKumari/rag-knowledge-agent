# Caching Strategies

Caching stores copies of expensive-to-compute or slow-to-fetch data in a faster layer so that repeated reads are cheap. The hard parts are keeping the cache correct and deciding what to evict.

## Write patterns

In cache-aside, the application checks the cache first, reads from the database on a miss, and then populates the cache. In write-through, every write goes to the cache and the database synchronously, which keeps them consistent at the cost of write latency. In write-back, writes land in the cache and are flushed to the database later, which is fast but risks data loss if the cache fails before flushing.

## Eviction policies

When the cache is full, an eviction policy chooses a victim. Least Recently Used (LRU) removes the entry that has gone unused the longest. Least Frequently Used (LFU) removes the entry with the fewest accesses. A time-to-live (TTL) bounds staleness by expiring entries after a fixed duration.

## Cache stampede

A cache stampede, also called the thundering herd, happens when a popular key expires and many requests simultaneously hit the database to rebuild it. Mitigations include request coalescing so only one caller recomputes the value, adding random jitter to TTLs, and refreshing entries before they expire.
