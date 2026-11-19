# Message Queues and Delivery Semantics

Message queues decouple producers from consumers, absorb traffic spikes, and let services fail and recover independently. Log-based brokers such as Kafka retain messages in an append-only log, while traditional brokers delete a message after it is acknowledged.

## Delivery guarantees

At-most-once delivery never redelivers a message, so a crash can lose it. At-least-once delivery redelivers until the consumer acknowledges, so duplicates are possible. Exactly-once processing is usually achieved by combining at-least-once delivery with idempotent consumers or transactional writes, rather than by the network itself.

## Partitions and ordering

A topic is split into partitions, and ordering is guaranteed only within a single partition. Messages with the same key are routed to the same partition, which preserves per-key order. Consumers in a consumer group divide the partitions between them, so the maximum parallelism of a group equals the number of partitions.

## Dead-letter queues and backpressure

A message that repeatedly fails processing is moved to a dead-letter queue so that it does not block the rest of the stream. Backpressure signals a fast producer to slow down when consumers fall behind, which prevents unbounded queue growth.
