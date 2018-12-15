// @Chuankai Zhao
// Email: czhao37@illinois.edu
// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE HISTOGRAM_LENGTH*4

typedef unsigned char uchar;
typedef unsigned int uint;

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//kernel that casts the image from float * to unsigned char *
__global__ void cast(float *input, uchar *output, int len){
  
  uint i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < len){
    output[i] = uchar (255 * input[i]); 
  }
}

//kernel that converts the RGB image to GrayScale
__global__ void convert(uchar *input, uchar *output, int len1, int len2){
  // Strategy 1: directly load the values from global memory
  
  uint i = threadIdx.x + blockIdx.x * blockDim.x;

  if ( (i < len2) && (3*i + 2 < len1) ){
    uchar r=input[3*i];
    uchar g=input[3*i + 1];
    uchar b=input[3*i + 2];
    output[i] = uchar (0.21*r  + 0.71*g + 0.07*b);
  }
  
  
  /*
  // Strategy 2: load the RGB values to shared memory first to utilize the DRAM burst
  __shared__ uchar rgb[3*BLOCK_SIZE];
  
  uint start = 3 * blockIdx.x * blockDim.x + threadIdx.x;
  for (uint i=0; i<3; i++){
    uint idx = start + i*blockDim.x;
    uint index = threadIdx.x + i*blockDim.x;
    if (idx < len1) {
      rgb[index] = input[idx];
    }
    else
      rgb[index] = 0;
  }
  __syncthreads();
  
  uint i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len2) {
    output[i] = uchar (0.21*rgb[3*threadIdx.x] + 0.71*rgb[3*threadIdx.x+1] + 0.07*rgb[3*threadIdx.x+2]);
  }
  */
}

//kernel that computes the histogram of the image
__global__ void histogram(uchar *input, uint *output, int len){
  
  // Strategy 1: atomic operation on the histogram stored in global memory
  /*
  uint i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < len){
    uchar val = input[i];
    atomicAdd( &(output[val]), 1 );
  }
  */
  
  // Strategy 2: atomic operation on the histogram stored in short-latency shared memory
  
  uint i = threadIdx.x + blockIdx.x * blockDim.x;
  
  __shared__ uint histo_s[HISTOGRAM_LENGTH];
  
  if (threadIdx.x < HISTOGRAM_LENGTH){
    histo_s[threadIdx.x] = 0;
  }
  __syncthreads();
  
  if (i < len){
    uchar val = input[i];
    atomicAdd( &(histo_s[val]), 1);
  }
  __syncthreads();
  
  if (threadIdx.x < HISTOGRAM_LENGTH){
    atomicAdd( &(output[threadIdx.x]), histo_s[threadIdx.x] );
  }
}

//kernel that performs the scan operations on the image
//Strategy 1: Brent-Kohn scan kernel
__global__ void Brent_Kohn_scan(uint *input, float *output, int len1, int len2){
  __shared__ float blockSum[HISTOGRAM_LENGTH];
  
  uint t=threadIdx.x;
  uint start=2*blockIdx.x*blockDim.x;
  
  // load the data into shared memory
  if (start + 2*t < len1){
    blockSum[2*t] = (input[start + 2*t])/( (float) (len2) );
  }
  else 
    blockSum[2*t] = 0.;
  
  if (start + 2*t + 1 < len1){
    blockSum[2*t + 1] = (input[start + 2*t + 1])/( (float) (len2));
  }
  else 
    blockSum[2*t + 1] = 0.;
  
  // reduction step 
  int stride = 1;
  while (stride < 2*blockDim.x){
    __syncthreads(); 
    int index = (t+1)*stride*2 - 1;
    if (index < 2*blockDim.x && (index - stride) >= 0){
      blockSum[index] += blockSum[index - stride];
    }
    stride = stride*2;
  }
  
  // post scan
  stride = blockDim.x/2;
  while (stride > 0){
    __syncthreads();
    int index = (t+1)*stride*2 - 1;
    if ( index + stride < 2*blockDim.x ){
      blockSum[index+stride] += blockSum[index];
    }
    stride = stride/2;
  }
  __syncthreads();
  
  // output the scanned array to global memory
  if (start + 2*t < len1){
    output[start + 2*t] = blockSum[2*t];
  }
  if (start + 2*t + 1 < len1){
    output[start + 2*t + 1] = blockSum[2*t + 1];
  }
}
  
//Strategy 2: Kogge-Stone scan kernel
__global__ void Kogge_Stone_scan(uint *input, float *output, int len1, int len2){
  __shared__ float blockSum[BLOCK_SIZE];
  __shared__ float buffer[BLOCK_SIZE];
  
  uint t = threadIdx.x;
  uint index = t + blockIdx.x * blockDim.x;
  
  if (t < len1){
    blockSum[t] = input[index]/( (float) (len2) );
    buffer[t]   = blockSum[t];
  }
  else 
    blockSum[t] = 0.;
    buffer[t]   = blockSum[t];
  
  // scan
  float *source = &blockSum[0];
  float *destination = &buffer[0];
  uint stride=1;
  while (stride < BLOCK_SIZE){
    __syncthreads();
    if ( t >= stride ){
      destination[t] = source[t] + source[t-stride];
      float *temp = destination; 
      destination = source;
      source = temp;
    }
    stride = stride*2;
  }
  __syncthreads();
  
  if (t < len1){
    output[t] = source[t];
  }
}

//kernel that correct the RGB pixel values in parallel
__global__ void correct(uchar *input1, float *input2, uchar *output, int len){  
  
  uint i = threadIdx.x + blockIdx.x * blockDim.x;
  
  float cdfmin = input2[0];  
  if ( i < len ){
    float temp = 255 * (input2[ input1[i] ] - cdfmin) / (1.0 - cdfmin);
    output[i] = uchar (min(max(temp, 0.0), 255.0));
  } 
}

//kernel that casts the image back from unsigned char * to float *
__global__ void inv_cast(uchar *input, float *output, int len){
  
  uint i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < len){
    output[i] = (float) (input[i]/255.0); 
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  
  //@@ Insert more code here
  float *deviceInputImageData;
  uchar *deviceUcharImage;
  uchar *deviceGreyImage;
  uint *deviceHistogram;
  float *deviceCDF;
  uchar *deviceCorrectImage;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  hostInputImageData = wbImage_getData(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  wbLog(TRACE, "The width of the image is ", imageWidth);
  wbLog(TRACE, "The height of the image is ", imageHeight);
  
  int numElements = imageWidth * imageHeight * imageChannels;
  int numGreyElements = imageWidth * imageHeight;
  
  //@@ insert code here
  wbTime_start(GPU, "Allocating memory on device.");
  wbCheck(cudaMalloc((void **)&deviceInputImageData, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceUcharImage, numElements * sizeof(uchar)));
  wbCheck(cudaMalloc((void **)&deviceGreyImage, numGreyElements * sizeof(uchar)));
  wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(uint)));
  wbCheck(cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceCorrectImage, numElements * sizeof(uchar)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating memory on device.");
  
  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(uint)));
  wbTime_stop(GPU, "Clearing output memory.");
  
  //@@ copy memory from host to device
  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, numElements*sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  //@@ initialize threads
  int numBlocks1 = (numElements + BLOCK_SIZE - 1)/BLOCK_SIZE;
  int numBlocks2 = (numGreyElements + BLOCK_SIZE - 1)/BLOCK_SIZE;
  dim3 dimGrid1(numBlocks1,1,1);
  dim3 dimGrid2(numBlocks2,1,1);
  dim3 dimGrid3(1,1,1);
  dim3 dimBlock1(BLOCK_SIZE, 1, 1);
  dim3 dimBlock2(HISTOGRAM_LENGTH/2, 1, 1); 
  
  //@@ launch the kernels to perform computations
  wbTime_start(Compute, "Performing CUDA computation");
  
  //cast image from float to char
  cast <<< dimGrid1, dimBlock1 >>> (deviceInputImageData, deviceUcharImage, numElements);
  
  //convert RGB to greyscale
  convert <<< dimGrid2, dimBlock1 >>> (deviceUcharImage, deviceGreyImage, numElements, numGreyElements);
  
  //calculate the histogram of greyimage using atomic operation
  histogram <<< dimGrid2, dimBlock1 >>> (deviceGreyImage, deviceHistogram, numGreyElements);

  
  //calculate Cumulative Distribution Function given the histogram
  Brent_Kohn_scan <<< dimGrid3, dimBlock2 >>> (deviceHistogram, deviceCDF, HISTOGRAM_LENGTH, numGreyElements);
  //Kogge_Stone_scan <<< dimGrid3, dimBlock1 >>> (deviceHistogram, deviceCDF, HISTOGRAM_LENGTH, numGreyElements);
  
  //map the original CDF to the desired CDF
  correct <<< dimGrid1, dimBlock1 >>> (deviceUcharImage, deviceCDF, deviceCorrectImage, numElements);
  
  //convert the image from char to float
  inv_cast <<< dimGrid1, dimBlock1 >>> (deviceCorrectImage, deviceOutputImageData, numElements);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");
  
  //@@ copy output from device to host;
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, numElements*sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  wbSolution(args, outputImage);

  //@@ free device and host memory here
  wbTime_start(GPU, "Freeing GPU memory");
  cudaFree(deviceInputImageData);
  cudaFree(deviceUcharImage);
  cudaFree(deviceGreyImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);
  cudaFree(deviceCorrectImage);
  cudaFree(deviceOutputImageData);
  wbTime_stop(GPU, "Freeing GPU memory");
  
  free(hostInputImageData);
  free(hostOutputImageData);
  
  return 0;
}
