#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;


//½·ÑÎÔëÉù
void salt(Mat &image, int n) {
	for (size_t k = 0; k < n; k++)
	{
		int i = rand() % image.cols;
		int j = rand() % image.rows;
		if (image.channels() == 1) {
			image.at<uchar>(j, i) = 255;
		}
		else if (image.channels() == 3) {
			image.at<Vec3b>(j, i)[0] = 255;
			image.at<Vec3b>(j, i)[1] = 255;
			image.at<Vec3b>(j, i)[2] = 255;
		}
	}
}

//ÑÕÉ«Ëõ¼õº¯Êý
void colorReduce(Mat &image, int div = 64) {
	int nl = image.rows;
	int nc = image.cols*image.channels();
	for (size_t j = 0; j < nl; j++)
	{
		uchar * data = image.ptr<uchar>(j);
		for (size_t i = 0; i < nc; i++)
		{
			data[i] = data[i] / div * div + div / 2;
		}
	}
}

void colorIteratorReduce(cv::Mat& image, cv::Mat& result, int div = 64)
{
	result.create(image.rows, image.cols, image.type());
	cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator rit = result.begin<cv::Vec3b>();
	for (; it != itend; it++) {
		(*rit)[0] = (*it)[0] / div * div + div / 2;
		(*rit)[1] = (*it)[1] / div * div + div / 2;
		(*rit)[2] = (*it)[2] / div * div + div / 2;
		rit++; 
	}
}

