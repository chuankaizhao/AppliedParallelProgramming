// @Chuankai Zhao
// Email: czhao37@illinois.edu
// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// kernel to add the scanned block sum to all values of scanned blocks
__global__ void add(float *input1, float *input2, int len){
  unsigned int t=threadIdx.x; 
  unsigned int start=2*(blockIdx.x + 1)*BLOCK_SIZE;

  // each threads do two times add operations
  if ( start + 2*t < len ){
    input1[start + 2*t] += input2[blockIdx.x];
  }
  if ( start + 2*t + 1 < len ){
    input1[start + 2*t + 1] += input2[blockIdx.x];
  }
}

// scan kernel to calculate the prefixed sum of each segment and store block sum to auxiliary array
__global__ void scan(float *input, float *aux, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float blockSum[2*BLOCK_SIZE];
  
  unsigned int t=threadIdx.x;
  unsigned int start=2*blockIdx.x*BLOCK_SIZE;
  
  // load data from global memory to shared memory
  if ( start + 2*t < len){
    blockSum[2*t] = input[start + 2*t];
  }
  else
    blockSum[2*t] = 0.;
  if (start + 2*t + 1 < len){
    blockSum[2*t + 1] = input[start + 2*t + 1];
  }
  else 
    blockSum[2*t + 1] = 0.;
  
  // reduction step 
  int stride = 1;
  while (stride < 2*BLOCK_SIZE){
    __syncthreads(); 
    int index = (t+1)*stride*2 - 1;
    if (index < 2*BLOCK_SIZE && (index - stride) >= 0){
      blockSum[index] += blockSum[index - stride];
    }
    stride = stride*2;
  }
  
  // post scan
  stride = BLOCK_SIZE/2;
  while (stride > 0){
    __syncthreads();
    int index = (t+1)*stride*2 - 1;
    if ( index + stride < 2*BLOCK_SIZE ){
      blockSum[index+stride] += blockSum[index];
    }
    stride = stride/2;
  }
  __syncthreads();
  
  // store block sum to auxiliary array
  if (t == 0) {
    aux[blockIdx.x] = blockSum[2*BLOCK_SIZE - 1];
  }
  
  // output the scanned array to global memory
  if (start + 2*t < len){
    output[start + 2*t] = blockSum[2*t];
  }
  if (start + 2*t + 1 < len){
    output[start + 2*t + 1] = blockSum[2*t + 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;  
  float *deviceAuxArray1;  // The Auxiliary Array 
  float *deviceAuxArray2;  // The updated Auxiliary Array
  float *deviceAuxSum;     // The scan block sums
  int numElements;         // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  int numAuxElements = (numElements + 2*BLOCK_SIZE - 1)/(2*BLOCK_SIZE); 
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxArray1, numAuxElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxArray2, 1 * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxSum, numAuxElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceAuxArray1, 0, numAuxElements * sizeof(float)));
  wbCheck(cudaMemset(deviceAuxArray2, 0, 1 * sizeof(float)));
  wbCheck(cudaMemset(deviceAuxSum, 0, numAuxElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid1(numAuxElements, 1, 1);
  dim3 dimGrid2(1,1,1);
  dim3 dimGrid3(numAuxElements - 1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  // Invoke the Brent-Kung scan kernel to generate per-block scan array and store
  // the block sums into an auxiliary block sum array
  scan <<< dimGrid1, dimBlock >>> (deviceInput, deviceAuxArray1, deviceOutput, numElements);
  if (numAuxElements > 1){
    // Invoke the Brent-Kung scan kernel again to translate the elements into a accumulative block sums.
    scan <<< dimGrid2, dimBlock >>> (deviceAuxArray1, deviceAuxArray2, deviceAuxSum, numAuxElements);
    // Invoke the add kernel to add the accumulative block sums to appropriate elements of the per-block scan array.
    add  <<< dimGrid3, dimBlock >>> (deviceOutput, deviceAuxSum, numElements);
  }
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray1);
  cudaFree(deviceAuxArray2);
  cudaFree(deviceAuxSum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

