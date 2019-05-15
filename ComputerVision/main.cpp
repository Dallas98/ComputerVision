#include <iostream>  
#include "Function.h"
#include "Experiment.h"
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace cv;

vector<Point> GetLinePoints(Point p1, Point p2)
{
	vector<Point> points;
	int steps = max(abs(p2.x - p1.x), abs(p2.y - p1.y));
	float dx = float(steps != 0 ? (p2.x - p1.x) / steps : 0);
	float dy = float(steps != 0 ? (p2.y - p1.y) / steps : 0);
	for (int i = 0; i <= steps; i++) {
		Point point;
		point.x = int(round(p1.x + dx * i));
		point.y = int(round(p1.y + dy * i));
		points.push_back(point);
	}
	return points;
}


void BlurContour(Mat& src, vector<Point> contour, int xof = 0, int yof = 0, int method = 0)
{
	Mat ref;
	src.copyTo(ref);
	int num_points = int(contour.size());
	for (int i = 0; i < num_points; i++) {
		Point point1 = contour[i], point2 = contour[(i + 1) % (num_points)];
		point1.x = point1.x + xof;
		point2.x = point2.x + xof;
		point1.y = point1.y + yof;
		point2.y = point2.y + yof;
		vector<Point> points = GetLinePoints(point1, point2);
		for (int pi = 0; pi < points.size(); pi++) {
			Point point = points[pi];
			uchar* s = src.ptr<uchar>(point.y);
			uchar* r = ref.ptr<uchar>(point.y);
			switch (method)
			{
			case 0:
			{
				for (int channel = 0; channel < 3; channel++)
					s[point.x * 3 + channel] = (r[(point.x - 1) * 3 + channel] + r[(point.x + 1) * 3 + channel]) / 2;
				break;
			}
			case 1:
			{
				for (int channel = 0; channel < 3; channel++)
					s[point.x * 3 + channel] = min(r[(point.x - 1) * 3 + channel], r[(point.x + 1) * 3 + channel]);
				break;
			}
			}
		}
	}
}


void CopyTo(Mat& src, Mat& dst, int x, int y, int width, int height, int thresh = 250, bool blur_contour = true, int blur_iterate = 30)
{
	Mat mask, gray;
	cvtColor(src, gray, CV_BGR2GRAY);
	threshold(gray, mask, thresh, 255, THRESH_BINARY_INV);
	copyMakeBorder(dst, dst, y < 0 ? -y : 0, height + y>dst.rows ? height + y - dst.rows : 0, x < 0 ? -x : 0, width + x>dst.cols ? width + x - dst.cols : 0, BORDER_CONSTANT, Scalar(255, 255, 255));
	src.copyTo(dst(Rect(x > 0 ? x : 0, y > 0 ? y : 0, width, height)), mask);

	if (blur_contour) {
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size(); i++) {
			vector<Point> contour = contours[i];
			if (contour.size() > 2) {
				for (int i = 0; i < blur_iterate; i++) {
					BlurContour(dst, contour, x - 1 > 0 ? x - 1 : 0, y > 0 ? y : 0, 1);
					BlurContour(dst, contour, x + 1 > 0 ? x + 1 : 0, y > 0 ? y : 0, 1);
					BlurContour(dst, contour, x > 0 ? x : 0, y > 0 ? y : 0, 0);
				}
			}
		}
	}
}


bool Comp(Rect rec1, Rect rec2)
{
	return rec1.x < rec2.x;
}


vector<Mat> Split(Mat& img, int thresh_contour = 80, bool write = false)
{
	vector<Mat> img_parts;
	Mat img_gray, img_bin;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	threshold(img_gray, img_bin, 230, 255, THRESH_BINARY_INV);
	vector<vector<Point> > contours;
	findContours(img_bin, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<Rect> rects;
	for (size_t i = 0; i < contours.size(); i++)
		if (contours[i].size() > thresh_contour)
			rects.push_back(boundingRect(contours[i]));
	sort(rects.begin(), rects.end(), Comp);
	for (size_t i = 0; i < rects.size(); i++) {
		Mat part;
		img(rects[i]).copyTo(part);
		img_parts.push_back(part);
		if (write)
			imwrite("part" + to_string(i + 1) + ".jpg", part);
	}
	return img_parts;
}


int main()
{
	Experiment3();
	/*Mat img = imread("elephant.jpg");
	vector<Mat> elephant;
	elephant = Split(img);
	imshow("elephant[2]", elephant[2]);
	waitKey();
	imshow("elephant[3]", elephant[3]);
	waitKey();
	Mat temp;
	elephant[2].copyTo(temp);
	CopyTo(elephant[3], temp, -70, 9, elephant[3].cols, elephant[3].rows, 220);
	elephant[3].copyTo(temp);
	CopyTo(elephant[3], temp, -70, 9, elephant[3].cols, elephant[3].rows, 220);
	imshow("ret", temp);
	waitKey();*/
	return 0;
}
