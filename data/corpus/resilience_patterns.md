# Resilience Patterns

Distributed systems fail partially and unpredictably, so callers need defences that stop one slow dependency from taking down everything else.

## Timeouts and retries

Every remote call needs a timeout, because a missing timeout can exhaust threads while waiting on a hung dependency. Retries recover from transient faults, but naive immediate retries amplify load on an already struggling service. Exponential backoff increases the delay between attempts, and adding random jitter prevents many clients from retrying in lockstep. Retries are only safe for idempotent operations.

## Circuit breaker

A circuit breaker wraps calls to a dependency and tracks failures. In the closed state calls pass through. After the failure rate crosses a threshold it opens, and calls fail fast without touching the dependency. After a cool-down it moves to half-open and lets a few trial calls through; success closes the circuit, and failure reopens it.

## Bulkheads and load shedding

A bulkhead isolates resources, for example separate thread pools per dependency, so that one failing dependency cannot consume them all. Load shedding deliberately rejects low-priority work when the system is saturated, preserving capacity for important requests.
