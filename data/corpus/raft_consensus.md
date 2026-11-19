# Raft Consensus

Raft is a consensus algorithm designed to be easier to understand than Paxos. It keeps a replicated log consistent across a cluster of servers so that they behave like a single fault-tolerant state machine.

## Roles and terms

Every server is in one of three states: follower, candidate, or leader. Time is divided into terms, which are numbered with consecutive integers. Each term begins with an election, and at most one leader can be elected in a given term.

## Leader election

Followers expect periodic heartbeats from the leader. If a follower hears nothing for a randomized election timeout, it increments its term, becomes a candidate, votes for itself, and requests votes from the other servers. A candidate that gathers votes from a majority of the cluster becomes the leader. Randomized timeouts make split votes rare and easy to resolve.

## Log replication

Clients send commands to the leader. The leader appends each command to its log and replicates it with AppendEntries requests. An entry is committed once it is stored on a majority of servers, and only then is it applied to the state machine. A cluster of five servers tolerates two failures, because three servers still form a majority.

## Safety

Raft guarantees that a committed entry is never lost. A candidate cannot win an election unless its log is at least as up to date as the logs of a majority of voters, so a new leader always holds every committed entry.
