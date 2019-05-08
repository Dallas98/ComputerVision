#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "Experiment.h"
using namespace std;
using namespace cv;

void findlp(Mat image, int &x, int&y);
void draw(vector<vector<Point>> contours, Mat image, Point *point);

void Experiment1() {
	// read source img
	Mat srcimg = imread("tree.jpg", IMREAD_COLOR);

	// crop to a region covering the tree
	Mat tree = srcimg;
	tree = tree(Rect(525, 0, 300, tree.rows*0.64));

	// split blue channel as a mask
	vector<Mat> channels;
	split(tree, channels);
	Mat imageBlueChannel = channels.at(0);
	Mat mask1, mask2;
	threshold(imageBlueChannel, mask1, 223, 255, CV_THRESH_BINARY);
	threshold(imageBlueChannel, mask2, 160, 255, CV_THRESH_BINARY);
	mask1 = 255 - mask1; // inverse color
	mask2 = 255 - mask2;

	// adujst details
	mask2(Rect(0, tree.rows - 14, tree.cols, 14)).copyTo(mask1(Rect(0, tree.rows - 14, tree.cols, 14)));
	mask1(Rect(175, tree.rows - 10, tree.cols - 175, 10)) = 0;
	mask1(Rect(0, tree.rows - 10, 145, 10)) = 0;

	// use mask to split tree out
	tree.copyTo(srcimg(Rect(100, 5, tree.cols, tree.rows)), mask1);
	imshow("", srcimg);
	waitKey(0);
}



void Experiment2() {
	int sum_x = 0, sum_y = 0;
	Point point[4];
	Mat image1 = imread("animals.jpg");
	Mat image = imread("animals.jpg", 0);
	Mat thresholded;
	//threshold用来进行对图像（二维数组）的二值化阈值处理，thresholded为输出图像
	threshold(image, thresholded, 60, 255, THRESH_BINARY_INV);
	//imshow("", thresholded);
	Mat res1;
	dilate(thresholded, res1, Mat());
	dilate(res1, res1, Mat());
	Mat res;
	erode(res1, res, Mat());

	vector<vector<Point>> contours;
	findContours(res, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//Mat result(res.size(), CV_8U, Scalar(255));
	int cmin = 100;
	int cmax = 1000;
	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end()) {
		if (itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else
			++itc;
	}
	drawContours(image1, contours, -1, Scalar(255), 2);
	for (int i = 0; i < contours.size(); i++)
	{
		sum_x = sum_y = 0;
		for (int j = 0; j < contours[i].size(); j++)
		{
			sum_x += contours[i][j].x;
			sum_y += contours[i][j].y;
		}
		point[i].x = sum_x / contours[i].size();
		point[i].y = sum_y / contours[i].size();
	}
	for (int i = 0; i < contours.size(); i++)
	{
		circle(image1, point[i], 5, Scalar(255, 0, 0), -1);
	}

	imshow("result", image1);
	waitKey(0);
	//imwrite("animalsres.bmp", image1);
}

void Experiment3()
{
	Mat image1 = imread("elephant.jpg");
	Mat image = imread("elephant.jpg", 0);
	vector<Mat> planes;
	split(image, planes);
	Mat image2 = planes[0];
	Mat thresholded;
	threshold(image2, thresholded, 190, 255, THRESH_BINARY_INV);
	Mat res = thresholded;
	vector<vector<Point>> contours;
	findContours(res, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	int cmin = 500;
	int cmax = 10000;
	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end()) {
		if (itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else
			++itc;
	}
	//drawContours(image1, contours, -1, Scalar(255, 0, 0), 2);
	//得到每个轮廓的宽度length和最小横坐标x
	int num = contours.size();
	Point point[8];
	//每一个轮廓的最大最小横坐标
	for (int i = 0; i < num; i++)
	{
		int min_y = 10000;
		int max_y = -10000;
		for (int j = 0; j < contours[i].size(); j++)
		{
			if (contours[i][j].x < min_y)
				min_y = contours[i][j].x;
			if (contours[i][j].x > max_y)
				max_y = contours[i][j].x;
		}
		point[i].x = min_y;
		point[i].y = max_y;
	}
	for (int i = 0; i < num; i++)
		cout << '(' << point[i].x << ',' << point[i].y << ')' << endl;
	cout << contours[0].size();
	imshow("res", image1);

	draw(contours, image1, point);
	waitKey(0);
}


void findlp(Mat image, int &x, int&y)
{
	vector<Mat> planes;
	split(image, planes);
	Mat image2 = planes[0];
	for (int i = 0; i < image2.rows; i++) {
		for (int j = 0; j < image2.cols; j++) {
			if (image2.at<uchar>(i, j) < 80) {
				x = i;
				i = image2.rows;
				break;
			}
		}
	}
	for (int i = 0; i < image2.rows; i++) {
		for (int j = 0; j < 100; j++) {
			if (image2.at<uchar>(i, j) < 80) {
				y = j;
				i = image2.rows;
				break;
			}
		}
	}
}

void draw(vector<vector<Point>> contours, Mat image, Point *point)
{
	//生成与原图大小相同的空白图片
	Mat mask = Mat::zeros(image.size(), image.type());
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			for (int k = 0; k < 3; k++) {
				mask.at<Vec3b>(i, j)[k] = 255;
			}
		}
	}
	//实际大象块顺序：2，8，6，5，1，4，7，3
	int order[8] = { 0, 2, 4, 5, 3, 7, 6, 1 };
	int start[8] = { 88, 148, 213, 294, 394, 474, 540, 636 };
#pragma omp parallel for schedule(dynamic) num_threads(88)
	for (int i = 7; i >= 0; i--)
	{
		int num = order[i];//第几个块
		for (int j = 0; j < image.rows; j++) {
			for (int k = 0; k < image.cols; k++)
			{

				if (pointPolygonTest(contours[num], Point(k, j), false) >= 0)
				{
					int y = k - point[num].x + start[i];//减去最小的列坐标
					for (int l = 0; l < 3; l++) {
						mask.at<Vec3b>(j, y)[l] = image.at<Vec3b>(j, k)[l];
					}
				}
			}
		}
	}

	imshow("mask", mask);
	waitKey(-1);
}