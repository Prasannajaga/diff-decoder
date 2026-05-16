# Important Notes:

These are the key observation from the composer-2 tech report.


## Training Service

### The reason behind handling with distributed system

* Decoupling training from inference and environment infrastructure naturally makes training more resilient to failures in these services;
* During the training run, we saw many cases where these services had
  partial or full outages without failing the training job.
* To minimize the number of training job restarts, we use a reactive configuration system and support live code updates on a per- process level; when new code is deployed, existing actors are drained of in-flight requests  and transparently replaced.

## Environment Service

### Triggering multiple pods in anyCluster ?

1. Scheduling throughput is particularly important for the bursty nature of RL workloads. Each
   Anyrun cluster is capable of scheduling more than 500 pods per second while maintaining
   desired binpacking requirements. One challenge with a naive packing strategy is that the
   steady-state resource usage for a pod can be dramatically lower than its peak during startup
   and can also be bursty due to overcommits.
2. To solve this, we monitor and schedule with
   awareness of live readings of hardware pressure (CPU, memory, disk) along with more
   conventional scheduling heuristics.

## Inference Service

### how the model weights been shared across clusters ?

1. Compression, upload, and hotload signaling are fully pipelined in background workers so that training is never blocked.

### How GPU compute worked for Inference ?

1. During the Composer 2 training run, we ran inference across geographically distributed
   clusters in the US and Europe. Each cluster independently downloads and reconstructs
   weights from the shared delta chain, requiring no direct connectivity to the training cluster, enabling world-scale distributed RL inference over commodity cloud storage.
