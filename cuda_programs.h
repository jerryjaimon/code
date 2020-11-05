#ifndef _CUDA_PROGRAMS_
#define _CUDA_PROGRAMS_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
void Histogram_Calculation_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram);
void Image_Inversion_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels);
void colorCorrection_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels, int channels[3][2]);
void canny_CUDA(Mat img, Mat& output, Mat filterImage, Mat sobelImage, Mat nonMaxImage, Mat finalImage);
void sobel_CUDA(Mat img, Mat& output, Mat filterImage, Mat sobelImage);
void boxFilter_CUDA(Mat& input, Mat& output);
void sobelFilter_CUDA(unsigned char* inputMatrixPointer, unsigned char* outputMatrixPointer, int Width, int Height);
#endif

