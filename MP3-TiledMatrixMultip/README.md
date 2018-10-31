# MP-3 TiledMatrixMultip

## Objective

The purpose of this lab is to implement a tiled dense matrix multiplication routine using shared memory.

## Instruction

Edit the code to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory
- implement the matrix-matrix multiplication routine using shared memory and tiling
