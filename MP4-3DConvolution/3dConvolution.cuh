// @Chuankai Zhao
// Email: czhao37@illinois.edu
// CUDA program for 3D convolution calculation

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define radius MASK_WIDTH/2
#define BLOCK_WIDTH 8
#define TILE_WIDTH (BLOCK_WIDTH + MASK_WIDTH - 1)
  
//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float subTile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;
  
  //@@ Calculate index for thread within a block
  int ID = threadIdx.x + (threadIdx.y * BLOCK_WIDTH) + (threadIdx.z * BLOCK_WIDTH * BLOCK_WIDTH);
  //@@ Calculate x, y, z index in input tile
  int IDtmp = ID;
  int IDx = IDtmp % TILE_WIDTH;
  IDtmp   = IDtmp / TILE_WIDTH;
  int IDy = IDtmp % TILE_WIDTH;
  int IDz = IDtmp / TILE_WIDTH;
  
  //@@ Use all threads to load the uppermost data in input tile. 
  //@@ Identify location of input data to load. 
  int newZ = IDz + (blockIdx.z*BLOCK_WIDTH) - radius;
  int newY = IDy + (blockIdx.y*BLOCK_WIDTH) - radius;
  int newX = IDx + (blockIdx.x*BLOCK_WIDTH) - radius;
  int newID = newX + newY * x_size + newZ * x_size * y_size;
  
  //@@ Load the data to the shared memory. 
  if (newZ >= 0 && newZ < z_size && newY >= 0 && newY < y_size && newX >= 0 && newX < x_size){
    subTile[IDz][IDy][IDx] = input[newID];
  }
  else {
    subTile[IDz][IDy][IDx] = 0;
  }
  __syncthreads();
  
  //@@ Use all the threads to load the remaining data to shared memory
  //@@ Calculate index for thread within a block
  //@@ Calculate x, y, z index in input tile
  ID = threadIdx.x + (threadIdx.y * BLOCK_WIDTH) + (threadIdx.z * BLOCK_WIDTH * BLOCK_WIDTH) + BLOCK_WIDTH * BLOCK_WIDTH * BLOCK_WIDTH;
  IDtmp = ID;
  IDx = IDtmp % TILE_WIDTH;
  IDtmp   = IDtmp / TILE_WIDTH;
  IDy = IDtmp % TILE_WIDTH;
  IDtmp   = IDtmp / TILE_WIDTH;
  IDz = IDtmp;
  
  //@@ Identify the input tile position. 
  newZ = IDz + (blockIdx.z*BLOCK_WIDTH) - radius;
  newY = IDy + (blockIdx.y*BLOCK_WIDTH) - radius;
  newX = IDx + (blockIdx.x*BLOCK_WIDTH) - radius;
  newID = newX + newY*x_size + newZ * x_size * y_size;
  
  //@@ Load the data to the shared memory. 
  if (IDz < TILE_WIDTH){
    if (newZ >= 0 && newZ < z_size && newY >= 0 && newY < y_size && newX >= 0 && newX < x_size){
      subTile[IDz][IDy][IDx] = input[newID];
    }
    else {
      subTile[IDz][IDy][IDx] = 0;
    }
  }
  __syncthreads();
  
  //@@ Calculate the convolution
  float Pvalue = 0.0;
  for (int k=0; k<MASK_WIDTH; k++){
    for (int j=0; j<MASK_WIDTH; j++){
      for (int i=0; i<MASK_WIDTH; i++){
        Pvalue += subTile[threadIdx.z + k][threadIdx.y + j][threadIdx.x + i]*Mc[k*(MASK_WIDTH*MASK_WIDTH) + j*MASK_WIDTH + i];
      }
    }
  }
  
  //@@ Save the output
  if ( x < x_size && y < y_size && z < z_size ){
    output[z*(y_size*x_size) + y*x_size + x] =  Pvalue;
    __syncthreads();
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data: input, kernel
  // Allocate memory for output
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc( (void **) &deviceInput,  (inputLength - 3)*sizeof(float) );
  cudaMalloc( (void **) &deviceOutput, (inputLength - 3)*sizeof(float) );
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength*sizeof(float));
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3)*sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((1.0*x_size)/BLOCK_WIDTH), ceil((1.0*y_size)/BLOCK_WIDTH), ceil((1.0*z_size)/BLOCK_WIDTH));
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d <<< dimGrid, dimBlock >>> (deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
