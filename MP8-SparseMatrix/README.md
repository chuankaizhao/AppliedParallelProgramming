# MP-8 Sparse Matrix-Vector Multiplication

## Objective
The purpose of this lab is to implement a SpMV (Sparse Matrix Vector Multiplication) kernel for an input sparse matrix based on the Jagged Diagonal Storage (JDS) transposed format.

## Instructions

Edit the kernel and the host function in the file to implement sparse matrix-vector multiplication using the JDS format. The kernel shall be launched so that each thread will generate one output Y element. The kernel should have each thread to use the appropriate elements of the JDS data array, the JDS col index array, JDS row index array, and the JDS transposed col ptr array to generate one Y element.
