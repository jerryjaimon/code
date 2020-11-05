#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_programs.h"
//#include "sobelFilter_CUDA.h"
#include <stdio.h>


#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3      
#define M_PI      3.14159265358979323846   // pi


using namespace cv;
using namespace std;

__global__ void Inversion_CUDA(unsigned char* Image, int Channels);
__global__ void boxFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel);
__global__ void Histogram_CUDA(unsigned char* Image, int* Histogram);


__global__ void Inversion_CUDA(unsigned char* Image, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) {
		Image[idx + i] = 255 - Image[idx + i];
	}
}

__global__ void color_Correction(unsigned char* Image, int Channels,int channels[3][2]) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) {
		if (Image[idx + i] < channels[3][0]) Image[idx + i] = channels[3][0];
		if (Image[idx + i] > channels[3][1]) Image[idx + i] = channels[3][1];
		Image[idx + i] = 255 - Image[idx + i];
	}
}

__global__ void color(unsigned char* Image, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) {
		Image[idx + i] = 255 - Image[idx + i];
	}
}


__global__ void boxFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		for (int c = 0; c < channel; c++)
		{
			float sum = 0;
			float kS = 0;
			for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
				for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
					float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
					sum += fl;
					kS += 1;
				}
			}
			dstImage[(y * width + x) * channel + c] = sum / kS;
		}
	}
}


__global__ void Histogram_CUDA(unsigned char* Image, int* Histogram) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = x + y * gridDim.x;

	atomicAdd(&Histogram[Image[Image_Idx]], 1);
}




__device__ void useFilterCuda(uchar* img_in, uchar* img_out, double* filterIn, int sizeFilter, int rows, int cols) {
int index = threadIdx.x;
int stride = blockDim.x;
for (int i = index; i < rows * cols; i += stride)
{
	int x = (int)(i / cols);
	int y = i % cols;
	if (x + sizeFilter < rows && y + sizeFilter < cols) {
		double sum = 0;
		for (int xf = 0; xf < sizeFilter; xf++)
			for (int yf = 0; yf < sizeFilter; yf++)
			{
				int index = ((x + xf) * cols) + (y + yf);
				int fIndex = xf * sizeFilter + yf;
				sum += img_in[index] * filterIn[fIndex];
			}
		img_out[i] = sum;
	}
	}
}

__device__ void sobelCuda(uchar* imgIn, uchar* imgOut, float* imgAngles, int rows, int cols) {
	//Sobel X Filter
	double xFilter[] = { -1.0f, 0, 1.0f, -2.0f, 0, 2.0f, -1.0f, 0, 1.0f };

	//Sobel Y Filter
	double yFilter[] = { 1.0f, 2.0f, 1.0f, 0, 0, 0, -1.0f, -2.0f, -1.0f };

	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < rows * cols; i += stride)
	{
		double sumx = 0;
		double sumy = 0;
		int xIndex = i / cols;
		int yIndex = i % cols;
		if (xIndex + 3 < rows && yIndex + 3 < cols) {
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
				{
					int fIndex = x * 3 + y;
					int index = (xIndex + x) * cols + yIndex + y;
					sumx += xFilter[fIndex] * (imgIn[index]); //Sobel_X Filter Value
					sumy += yFilter[fIndex] * (imgIn[index]); //Sobel_Y Filter Value
				}
			double sumxsq = sumx * sumx;
			double sumysq = sumy * sumy;

			double sq2 = sqrt(sumxsq + sumysq);

			if (sq2 > 255) //Unsigned Char Fix
				sq2 = 255;
			imgOut[i] = sq2;

			if (sumx == 0) //Arctan Fix
				imgAngles[i] = 90.0f;
			else
				imgAngles[i] = atan(sumy / sumx);
		}
		else
			imgAngles[i] = 90.0f;
	}
}

__device__ void nonMaxSupp(uchar* img_in, uchar* img_out, float* angles, int rows, int cols)
{	
	int index = threadIdx.x;
	int stride = blockDim.x;
	int max = index * stride > rows * cols - cols - rows ? index * stride : rows * cols - cols - rows;
	int start = index > cols ? index : cols;
	int mSize = rows * cols;
	for (int i = index; i < mSize; i += stride) {
		float targentData = angles[i];
		int x = i / cols;
		int y = i % cols;
		int index = (x - 1) * cols + y - 1;
		if (index > 0 && index < mSize) {
			img_out[index] = img_in[i];
			//Horizontal Edge
			if ((targentData > -22.5 && targentData <= 22.5) || (targentData > 157.5 && targentData <= -157.5)) {
				if (img_in[i] < img_in[i + 1] || img_in[i] < img_in[i - 1])
					img_out[index] = 0;
			}
			//Vertical Edge
			if (((-112.5 < targentData) && (targentData <= -67.5)) || ((67.5 < targentData) && (targentData <= 112.5)))
			{
				if (x + 1 < rows && x - 1 > 0 && (img_in[i] < img_in[(x + 1) * cols + (y)] || img_in[i] < img_in[(x - 1) * cols + y]))
					img_out[index] = 0;
			}

			//-45 Degree Edge
			if (((-67.5 < targentData) && (targentData <= -22.5)) || ((112.5 < targentData) && (targentData <= 157.5)))
			{
				if (y + 1 < cols && x - 1 > 0 && x + 1 < rows && y - 1 > 0 &&
					(img_in[i] < img_in[(x - 1) * cols + (y + 1)] || img_in[i] < img_in[(x + 1) * cols + (y - 1)]))
					img_out[index] = 0;
			}

			//45 Degree Edge
			if (((-157.5 < targentData) && (targentData <= -112.5)) || ((22.5 < targentData) && (targentData <= 67.5)))
			{
				if (y + 1 < cols && x - 1 > 0 && x + 1 < rows && y - 1 > 0 &&
					(img_in[i] < img_in[(x + 1) * cols + (y + 1)] || img_in[i] < img_in[(x - 1) * cols + (y - 1)]))
					img_out[index] = 0;
			}
		}
	}
}

__device__ void threshold(uchar* img_in, uchar* img_out, int low, int high, int rows, int cols)
{
	if (low > 255)
		low = 255;
	if (high > 255)
		high = 255;

	int index = threadIdx.x;
	int stride = blockDim.x;
	int max = index * stride > rows * cols ? index * stride : rows * cols;
	for (int i = index; i < rows * cols; i += stride)
	{
		img_out[i] = img_in[i];
		if (img_out[i] > high)
			img_out[i] = 255;
		else if (img_out[i] < low)
			img_out[i] = 0;
		else
		{
			bool anyHigh = false;
			bool anyBetween = false;
			for (int x = 0; x < 3 * 3; x++)
			{
				int index = (i + ((x / 3) - 1) * cols) + ((i % cols) + (x % 3) - 1);
				if (index < 0 || index >= cols * rows) //Out of bounds
					continue;
				else
				{
					if (img_out[index] > high)
					{
						img_out[i] = 255;
						anyHigh = true;
						break;
					}
					else if (img_out[index] <= high && img_out[index] >= low)
						anyBetween = true;
				}
				if (anyHigh)
					break;
			}
			if (!anyHigh && anyBetween)
				for (int x = 0; x < 5 * 5; x++)
				{
					int index = (i + ((x / 5) - 2) * cols) + ((i % cols) + (x % 5) - 2);
					if (index < 0 || index >= cols * rows) //Out of bounds
						continue;
					else
					{
						if (img_out[index] > high)
						{
							img_out[i] = 255;
							anyHigh = true;
							break;
						}
					}

					if (anyHigh)
						break;
				}
			if (!anyHigh)
				img_out[i] = 0;
		}
	}
}

__global__ void canny(uchar* imgIn, uchar* imgGaussian, uchar* imgSober, uchar* imgNonMax, uchar* imgFinal,
	float* angles, double* filterIn, int sizeFilter, int rows, int cols) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	useFilterCuda(imgIn, imgGaussian, filterIn, sizeFilter, rows, cols);//Gaussian filter
	sobelCuda(imgGaussian, imgSober, angles, rows, cols);//Finding the intensity (Sobel) 
	nonMaxSupp(imgSober, imgNonMax, angles, rows, cols);//Non-maximum suppression
	threshold(imgNonMax, imgFinal, 20, 40, rows, cols);//Double threshold
}



double* createFilter(int row, int column, double sigmaIn)
{
	vector<vector<double>> filter;

	for (int i = 0; i < row; i++)
	{
		vector<double> col;
		for (int j = 0; j < column; j++)
		{
			col.push_back(-1);
		}
		filter.push_back(col);
	}

	float coordSum = 0;
	float constant = 2.0 * sigmaIn * sigmaIn;

	// Sum is for normalization
	float sum = 0.0;

	for (int x = -row / 2; x <= row / 2; x++)
	{
		for (int y = -column / 2; y <= column / 2; y++)
		{
			coordSum = (x * x + y * y);
			filter[x + row / 2][y + column / 2] = (exp(-(coordSum) / constant)) / (M_PI * constant);
			sum += filter[x + row / 2][y + column / 2];
		}
	}

	// Normalize the Filter
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			filter[i][j] /= sum;
	double* mFilter = new double[row * column];
	for (int i = 0; i < filter.size(); i++)
	{
		for (int j = 0; j < filter[i].size(); j++)
		{
			int index = i * row + j;
			mFilter[index] = filter[i][j];
		}
	}
	return mFilter;

}

__global__ void sobel_mainCUDA(uchar* imgIn, uchar* imgGaussian, uchar* imgSober,
	float* angles, double* filterIn, int sizeFilter, int rows, int cols) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	useFilterCuda(imgIn, imgGaussian, filterIn, sizeFilter, rows, cols);//Gaussian filter
	sobelCuda(imgGaussian, imgSober, angles, rows, cols);//Finding the intensity (Sobel) 
}

void sobel_CUDA(Mat img, Mat& output, Mat filterImage, Mat sobelImage) {
	uchar* img_sober, * img_non, * img_final, * img_filter, * img_in;
	uchar* cSoberImg, * cNonImg, * cFinalImg, * cFilterImg, * cInImg;
	float* angles, * cAngles;
	double* filterIn, * cFilter;
	int sizeFilter = 5;

	long size = sizeof(uchar) * img.rows * img.cols;
	int cSizeFilter = sizeof(double) * sizeFilter * sizeFilter;
	int sizeAngles = sizeof(float) * img.rows * img.cols;
	angles = new float[img.rows * img.cols];

	filterIn = createFilter(sizeFilter, sizeFilter, 1);

	// Allocate host memory
	img_sober = (uchar*)malloc(size);
	img_filter = (uchar*)malloc(size);

	cudaMalloc((void**)&cInImg, size);
	cudaMalloc((void**)&cSoberImg, size);
	cudaMalloc((void**)&cFilterImg, size);
	cudaMalloc((void**)&cAngles, sizeAngles);
	cudaMalloc((void**)&cFilter, cSizeFilter);


	cudaMemcpy(cInImg, img.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cFilterImg, filterImage.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cSoberImg, sobelImage.data, size, cudaMemcpyHostToDevice);

	cudaMemcpy(cFilter, filterIn, cSizeFilter, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start); {
		sobel_mainCUDA << <1, 256 >> > (cInImg, cFilterImg, cSoberImg, cAngles, cFilter, sizeFilter, img.rows, img.cols);
	}
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("\n\tElapsed GPU time : %fms", ms);
	// Transfer data back to host memory
	cudaMemcpy(img_filter, cFilterImg, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(img_sober, cSoberImg, size, cudaMemcpyDeviceToHost);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}
	output.data = img_sober;
	// Deallocate device memory
	cudaFree(cInImg);
	cudaFree(cFilterImg);
	cudaFree(cSoberImg);
	cudaFree(cFilter);
	cudaFree(cAngles);
}

void Histogram_Calculation_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram) {
	unsigned char* Dev_Image = NULL;
	int* Dev_Histogram = NULL;

	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Histogram, 256 * sizeof(int));

	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram, Histogram, 256 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Histogram_CUDA << <Grid_Image, 1 >> > (Dev_Image, Dev_Histogram);

	//copy memory back to CPU from GPU
	cudaMemcpy(Histogram, Dev_Histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	//free up the memory of GPU
	cudaFree(Dev_Histogram);
	cudaFree(Dev_Image);
}

void canny_CUDA(Mat img, Mat& output,Mat filterImage,Mat sobelImage,Mat nonMaxImage,Mat finalImage) {
	uchar* img_sober, * img_non, * img_final, * img_filter, * img_in;
	uchar* cSoberImg, * cNonImg, * cFinalImg, * cFilterImg, * cInImg;
	float* angles, * cAngles;
	double* filterIn, * cFilter;
	int sizeFilter = 5;

	long size = sizeof(uchar) * img.rows * img.cols;
	int cSizeFilter = sizeof(double) * sizeFilter * sizeFilter;
	int sizeAngles = sizeof(float) * img.rows * img.cols;
	angles = new float[img.rows * img.cols];

	filterIn = createFilter(sizeFilter, sizeFilter, 1);

	// Allocate host memory
	img_sober = (uchar*)malloc(size);
	img_filter = (uchar*)malloc(size);
	img_final = (uchar*)malloc(size);
	img_non = (uchar*)malloc(size);

	// Allocate device memory 
	cudaMalloc((void**)&cInImg, size);
	cudaMalloc((void**)&cSoberImg, size);
	cudaMalloc((void**)&cNonImg, size);
	cudaMalloc((void**)&cFilterImg, size);
	cudaMalloc((void**)&cFinalImg, size);
	cudaMalloc((void**)&cAngles, sizeAngles);
	cudaMalloc((void**)&cFilter, cSizeFilter);

	// Transfer data from host to device memory
	cudaMemcpy(cInImg, img.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cFilterImg, filterImage.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cSoberImg, sobelImage.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cNonImg, nonMaxImage.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cFinalImg, finalImage.data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cFilter, filterIn, cSizeFilter, cudaMemcpyHostToDevice);

	// Executing kernel 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start); {
		canny << <1, 256 >> > (cInImg, cFilterImg, cSoberImg, cNonImg, cFinalImg, cAngles, cFilter, sizeFilter, img.rows, img.cols);
	}
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("\n\tElapsed GPU time : %fms", ms);

	// Transfer data back to host memory
	cudaMemcpy(img_filter, cFilterImg, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(img_sober, cSoberImg, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(img_non, cNonImg, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(img_final, cFinalImg, size, cudaMemcpyDeviceToHost);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}

	output.data = img_final;
	cudaFree(cInImg);
	cudaFree(cFilterImg);
	cudaFree(cSoberImg);
	cudaFree(cNonImg);
	cudaFree(cFinalImg);
	cudaFree(cFilter);
	cudaFree(cAngles);
}

void Image_Inversion_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels) {
	unsigned char* Dev_Input_Image = NULL;
	cudaMalloc((void**)&Dev_Input_Image, Height * Width * Channels);
	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 Grid_Image(Width, Height);

	cudaEventRecord(start); {
	Inversion_CUDA << <Grid_Image, 1 >> > (Dev_Input_Image, Channels);
	}
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("\n\tElapsed GPU time : %fms", ms);

	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);
	cudaFree(Dev_Input_Image);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}
}

void boxFilter_CUDA(Mat& input,Mat& output)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int channel = input.step / input.cols;

	const int inputSize = input.cols * input.rows * channel;
	const int outputSize = output.cols * output.rows * channel;
	unsigned char* d_input, * d_output;

	cudaMalloc<unsigned char>(&d_input, inputSize);
	cudaMalloc<unsigned char>(&d_output, outputSize);

	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

	cudaEventRecord(start);

	boxFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);

	cudaEventRecord(stop);

	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventSynchronize(stop);
	float milliseconds = 0;

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\n\tElapsed GPU time : %fms", milliseconds);
}

void colorCorrection_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels,int channels[3][2]) {
	unsigned char* Dev_Input_Image = NULL;
	cudaMalloc((void**)&Dev_Input_Image, Height * Width * Channels);
	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 Grid_Image(Width, Height);

	cudaEventRecord(start); {
		color_Correction << <Grid_Image, 1 >> > (Dev_Input_Image, Channels,channels);
	}
	cudaEventRecord(stop);
	float ms = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("\n\tElapsed GPU time : %fms", ms);

	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);
	cudaFree(Dev_Input_Image);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}
}

