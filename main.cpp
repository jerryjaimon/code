#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include "cuda_programs.h"
//#include "boxFilter_CUDA.h"
//#include "sobelFilter_CUDA.h"
//#include "canny_CUDA.h"

using namespace std;
using namespace cv;



Mat resizeImage(Mat input) {
    Mat output;
    resize(input, output, Size(), 0.5, 0.5);
    return output;
}

int invertImage(Mat input,string str) {
    Mat Input_Image = input;
    cout << "\n\tHeight: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
    Image_Inversion_CUDA(Input_Image.data, Input_Image.cols, Input_Image.rows, Input_Image.channels());
    Mat output = resizeImage(Input_Image);
    string loc = "_Inverted_Image.png";
    str.erase(str.end()-4,str.end());
    str.append(loc);
    imwrite(str, Input_Image);
    imshow("Your Image", output);
    int k = waitKey(0);
    destroyAllWindows();
    return 0;
}

int colorCorrect(Mat input, string str) {
    Mat Input_Image = input;
    cout << "\n\tHeight: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
    int percent = 5;
    Input_Image.convertTo(Input_Image, CV_32F);
    vector<Mat> channels(3);
    int channel[3][2];
    int chnls = Input_Image.channels();
    printf("%d", chnls);
    double halfPercent = percent / 200.0;
    if (chnls == 3) split(Input_Image, channels);
    vector<Mat> results;
    for (int i = 0; i < chnls; i++) {
        Mat flat;
        channels[i].reshape(1, 1).copyTo(flat);
        cv::sort(flat, flat, SORT_ASCENDING);
        double lowVal = flat.at<int>(0, (int)floor(flat.cols * halfPercent));
        double topVal = flat.at<int>(0, (int)ceil(flat.cols * (1.0 - halfPercent)));
        channel[i][0]=lowVal;
        channel[i][1]=topVal;
     }
    colorCorrection_CUDA(Input_Image.data, Input_Image.cols, Input_Image.rows, Input_Image.channels(),channel);
    for (int i = 0; i < chnls; i++) {
        Mat channel = channels[i];
        normalize(channel, channel, 0.0, 255.0, NORM_MINMAX);
        channel.convertTo(channel, CV_32F);
        results.push_back(channel);
    }
    Mat outval;
    merge(results, outval);
    outval.convertTo(outval, CV_8UC1);
    Mat output = resizeImage(outval);
    string loc = "_Color_Corrected.png";
    str.erase(str.end() - 4, str.end());
    str.append(loc);
    imwrite(str, Input_Image);
    imshow("Your Image", output);
    int k = waitKey(0);
    destroyAllWindows();
    return 0;
}

int boxFilter(Mat input,string str) {
    Mat Input_Image = input;
    cout << "\n\tHeight: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
    Mat Output_Image(Input_Image.size(),Input_Image.type());
    boxFilter_CUDA(Input_Image, Output_Image);
    Mat output = resizeImage(Output_Image);
    string loc = "_boxFilter.png";
    str.erase(str.end() - 4, str.end());
    str.append(loc);
    imwrite(str, Output_Image);
    imshow("Your Image", output);
    int k = waitKey(0);
    destroyAllWindows();
    return 0;
}

int histogramValue(Mat input, string str) {
    Mat Input_Image;
    cvtColor(input, Input_Image, COLOR_BGR2GRAY);
    cout << "\n\tHeight: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
    Mat Output_Image(Input_Image.size(), Input_Image.type());
    int Histogram_GrayScale[256] = { 0 };
    Histogram_Calculation_CUDA(Input_Image.data, Input_Image.cols, Input_Image.rows, Input_Image.channels(),Histogram_GrayScale);
    for (int i = 0; i < 256; i++) {
        cout << "\tHistogram_GrayScale[" << i << "]: " << Histogram_GrayScale[i] << endl;
    }
    return 0;
}



int sobelFilter(Mat img, string str) {
    Mat gray;
    cout << "\n\tHeight: " << img.rows << ", Width: " << img.cols << ", Channels: " << img.channels() << endl;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat Output_Image(gray.size(), gray.type());
    Mat filterImage = Mat(img.rows, img.cols, CV_8UC1);
    Mat sobelImage = Mat(img.rows, img.cols, CV_8UC1);
    sobel_CUDA(gray, Output_Image, filterImage, sobelImage);
    Mat output = resizeImage(Output_Image);
    string loc = "_sobel.png";
    str.erase(str.end() - 4, str.end());
    str.append(loc);
    imwrite(str, Output_Image);
    imshow("Your Image", output);
    int k = waitKey(0);
    destroyAllWindows();
    return 0;

}

int canny(Mat img, string str) {
   
    Mat gray;
    cout << "\n\tHeight: " << img.rows << ", Width: " << img.cols << ", Channels: " << img.channels() << endl;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat Output_Image(gray.size(), gray.type());
    Mat filterImage = Mat(img.rows, img.cols, CV_8UC1);
    Mat sobelImage = Mat(img.rows, img.cols, CV_8UC1);
    Mat nonMaxImage = Mat(img.rows, img.cols, CV_8UC1);
    Mat finalImage = Mat(img.rows, img.cols, CV_8UC1);
    canny_CUDA(gray,Output_Image, filterImage, sobelImage,nonMaxImage,finalImage);
    Mat output = resizeImage(Output_Image);
    string loc = "_canny.png";
    str.erase(str.end() - 4, str.end());
    str.append(loc);
    imwrite(str, Output_Image);
    imshow("Your Image", output);
    int k = waitKey(0);
    destroyAllWindows();
    return 0;

}

Mat readImage(string location) {
    Mat InputImage = imread(location);
    return InputImage;
}

int menu() {
    int x;
    cout << "\n\t\tMenu";
    cout << "\n\tBasic Operations";
    cout << "\n\t  1.Invert Image";
    cout << "\n\t  2.Get Histogram Values:Grayscale";
    //cout << "\n\t  3.Get Histogram Value:Color";
    cout << "\n\tOther Operations:";
    cout << "\n\t  3.Box Filter";
    cout << "\n\t  4.Sobel Filter";
    //cout << "\n\t  6.Histogram Equalization";
    cout << "\n\t  5.Canny Edge Detection";
    cout << "\n\t  6.Underwater Image Enhancement";
    cout << "\n\tEnter your options:";
    cin >> x;
    return x;
}

int main()
{
    string location;
   
    while (1) {
        cout << "\n\tInput Image:";
        cin >> location;
        Mat InputImage = readImage(location);
        if (InputImage.empty())
        {
            cout << "Failed imread(): Image not found" << std::endl;
            continue;
        }
        int x = menu();
        if (x == 0) {
            cout << "\n\tEnding program";
            break;
        }
       
        switch (x) {
        case 1: invertImage(InputImage,location);
                break;
        case 2: histogramValue(InputImage, location);
                break;
        case 3: boxFilter(InputImage, location);
                break;
        case 4: sobelFilter(InputImage, location);
            break;
        case 5: canny(InputImage, location);
            break;
        case 6: colorCorrect(InputImage, location);
            break;
        default: continue;
        }
    }
    return 0;
}