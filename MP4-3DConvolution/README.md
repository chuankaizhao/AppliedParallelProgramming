# MP-4 3D Convolution

Implementation of a 3D convolution using constant memory for the kernel (mask filter) and 3D shared memory tiling. </br> </br>

The program launches 3D CUDA grid and blocks, where each thread is responsible for computing a single element of the output. </br> 

Input and output formats: float inputData[] = { z_size, y_size, x_size, float1, float2, ... } </br> 
