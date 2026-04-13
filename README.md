# Spherical-KV
Implementation of Spherical KV cache that has the initial pipeline ready.

Following are the things that are missing/not fully implemented:
1. CUDA kernel has been implemented but misses:
   - cosine clipping
   - coalesced loading of theta into shared memory
   - Tile based streaming
2. RDR controller is missing the following:
   - We are not accounting for meta-bits
   - No anti thrashing regularizer
