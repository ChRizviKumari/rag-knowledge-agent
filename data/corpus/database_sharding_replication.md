# Database Sharding and Replication

Replication copies data to several nodes for availability and read scaling. Sharding splits data across nodes so that no single machine has to hold or serve everything.

## Replication topologies

In single-leader replication, all writes go to one leader and are streamed to followers. Asynchronous replication keeps writes fast but followers can lag, so a read from a follower may return stale data. Synchronous replication waits for follower acknowledgement and avoids that staleness at the cost of write latency and availability. Multi-leader and leaderless designs accept writes on several nodes and must resolve conflicts.

## Sharding strategies

Range sharding assigns contiguous key ranges to shards, which makes range scans efficient but can create hot spots, for example when keys are timestamps. Hash sharding spreads keys evenly but destroys ordering. Directory-based sharding uses a lookup service to map keys to shards, which is flexible but adds a component that must itself be highly available.

## Resharding and joins

Adding shards requires moving data, so many systems pre-create more logical partitions than physical nodes. Queries that span shards need scatter-gather execution, and cross-shard transactions need a protocol such as two-phase commit, which is slow and blocks if the coordinator fails.
