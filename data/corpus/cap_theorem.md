# CAP Theorem and Consistency Models

The CAP theorem states that a distributed data store cannot simultaneously provide consistency, availability, and partition tolerance. Because network partitions cannot be ruled out in practice, the real choice is what to give up while a partition is happening.

## The three properties

Consistency in CAP means linearizability: every read sees the most recent completed write. Availability means every request to a non-failing node receives a response. Partition tolerance means the system keeps operating even when messages between nodes are dropped or delayed.

## CP and AP systems

A CP system refuses some requests during a partition to avoid returning stale or conflicting data. Coordination services built on consensus behave this way. An AP system keeps answering on both sides of the partition and reconciles divergent replicas afterwards, which is the approach taken by Dynamo-style stores.

## Eventual consistency

Eventual consistency promises that if no new updates are made, all replicas converge to the same value. Conflicts can be resolved with last-write-wins timestamps, vector clocks, or conflict-free replicated data types. PACELC extends CAP by observing that even without a partition there is a trade-off between latency and consistency.
