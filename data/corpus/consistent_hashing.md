# Consistent Hashing

Consistent hashing distributes keys across a changing set of nodes while moving as few keys as possible when nodes join or leave. It is used by distributed caches, key-value stores, and content delivery networks.

## The hash ring

Both nodes and keys are hashed onto the same circular space, often called a ring. A key is owned by the first node found when walking clockwise from the key's position. With simple modulo hashing, changing the number of nodes from N to N+1 remaps almost every key. With a ring, adding a node only takes over the keys between the new node and its predecessor, roughly 1/N of all keys.

## Virtual nodes

Placing each physical server at a single point on the ring leads to uneven load. To smooth this out, each server is assigned many virtual nodes, typically between 100 and 200 points on the ring. Virtual nodes also let a more powerful machine take a proportionally larger share by giving it more points.

## Replication

To survive failures, a key can be stored on the next several distinct physical nodes clockwise on the ring. Dynamo-style systems call this list of nodes the preference list, and skip virtual nodes that belong to a physical server already in the list.
