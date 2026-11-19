# Rate Limiting

Rate limiting protects a service from overload and abuse by capping how many requests a client may make in a period. When a limit is exceeded, servers conventionally return HTTP status 429 Too Many Requests, often with a Retry-After header.

## Token bucket

A token bucket holds up to a fixed capacity of tokens and is refilled at a constant rate. Each request consumes one token, and requests are rejected when the bucket is empty. Because a full bucket can be spent at once, the algorithm allows short bursts while enforcing the long-term average rate.

## Leaky bucket and windows

A leaky bucket processes requests at a fixed outflow rate and queues or drops the excess, which smooths traffic. A fixed window counter resets at boundaries and is simple, but a client can send double the limit by straddling two windows. A sliding window log stores a timestamp per request and is exact but memory-hungry, while a sliding window counter approximates it using two adjacent counters.

## Distributed limiting

When many servers share a limit, counters are commonly kept in a shared store such as Redis, using atomic increments or a Lua script to avoid race conditions. A cheaper alternative is to give each server a local share of the limit, accepting some inaccuracy.
