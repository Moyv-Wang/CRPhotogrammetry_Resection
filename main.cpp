#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include<fstream>
#include<sstream>
#include<vector>
#include<iostream>
using namespace std;
using namespace cv;

//  控制场平面坐标系
//  ^ X
//  |
//  |
//  |——————> Y

//  标志点量测坐标定义       标志点像平面坐标定义
//  --------> x(pixel)            ^ y(mm)
//  |                             |    
//  |                             |            
//  |                     -----------------> x
//  |                             |
//   y                            |    
// 

#define Pi 3.14159265
#define ROWS 2848
#define COLS 4272
#define PIXSIZE 0.00519663 //22.2mm × 14.8mm => 4272pixel × 2848pixel

struct ControlPoint
{
	int flag;
	double X;
	double Y;
	double Z;
};

struct imgPoint
{
	int flag;
	double x;
	double y;
};

struct PointPair
{
	int flag;
	double X;
	double Y;
	double Z;
	double x;
	double y;
};

//外方位元素结构
struct orienParam {
	//内方位元素
	double x0 = 0;
	double y0 = 0;
	double f = 0;
	//畸变系数
	double k1 = 0;
	double k2 = 0;
	double p1 = 0;
	double p2 = 0;
	//外方位元素
	double Phi = 0;
	double Omega = 0;
	double Kappa = 0;
	double Xs = 0;
	double Ys = 0;
	double Zs = 0;
};

void readClptData(char* file, vector<ControlPoint>& ControlPoints)
{
	ifstream inFile(file);
	if (!inFile)
	{
		cout << "文件读取失败，请检查文件路径" << endl;
		return;
	}
	string line;
	string firstLine;
	getline(inFile, firstLine);
	while (getline(inFile, line)) {
		ControlPoint clpt;
		istringstream iss(line);
		iss >> clpt.flag;
		iss >> clpt.Z;
		iss >> clpt.X;
		iss >> clpt.Y;
		clpt.Z = -1.0 * clpt.Z;
		ControlPoints.push_back(clpt);
	}
	inFile.close();
}

void readImgData(char* file, vector<imgPoint>& imgPoints)
{
	ifstream inFile(file);
	if (!inFile)
	{
		cout << "文件读取失败，请检查文件路径" << endl;
		return;
	}
	string line;
	while (getline(inFile, line)) {
		imgPoint imgpt;
		istringstream iss(line);
		iss >> imgpt.flag;
		iss >> imgpt.x;
		iss >> imgpt.y;
		imgPoints.push_back(imgpt);
	}
	inFile.close();
}

void GeneratePairs(vector<imgPoint>& imgPoints, vector<ControlPoint>& ControlPoints, vector<PointPair>& PointPairs)
{
	PointPair pair;
	for (int i = 0; i < imgPoints.size(); i++)
	{
		for (int j = 0; j < ControlPoints.size(); j++)
		{
			if (imgPoints[i].flag == ControlPoints[j].flag)
			{
				pair.flag = imgPoints[i].flag;
				pair.X = ControlPoints[j].X;
				pair.Y = ControlPoints[j].Y;
				pair.Z = ControlPoints[j].Z;
				pair.x = imgPoints[i].x;
				pair.y = imgPoints[i].y;
				PointPairs.push_back(pair);
				break;
			}
		}
	}
}


void cvtPix2Img(vector<imgPoint>& imgPoints)
{
	for (int i = 0; i < imgPoints.size(); i++)
	{
		imgPoints[i].x = (imgPoints[i].x - COLS / 2) * PIXSIZE;
		imgPoints[i].y = -1.0 * (imgPoints[i].y - ROWS / 2) * PIXSIZE;
	}
}

void cal_Coefficient(Mat& A, Mat& l, vector<PointPair>& PointPairs, orienParam& orien)
{
	double x0 = orien.x0;
	double y0 = orien.y0;
	double f = orien.f;
	double k1 = orien.k1;
	double k2 = orien.k2;
	double p1 = orien.p1;
	double p2 = orien.p2;
	double Phi = orien.Phi;
	double Omega = orien.Omega;
	double Kappa = orien.Kappa;
	double Xs = orien.Xs;
	double Ys = orien.Ys;
	double Zs = orien.Zs;
	//计算旋转矩阵
	Mat R = Mat::zeros(3, 3, CV_64FC1);
	R.at<double>(0, 0) = cos(Phi) * cos(Kappa) - sin(Phi) * sin(Omega) * sin(Kappa);
	R.at<double>(0, 1) = -1.0 * cos(Phi) * sin(Kappa) - sin(Phi) * sin(Omega) * cos(Kappa);
	R.at<double>(0, 2) = -1.0 * sin(Phi) * cos(Omega);
	R.at<double>(1, 0) = cos(Omega) * sin(Kappa);
	R.at<double>(1, 1) = cos(Omega) * cos(Kappa);
	R.at<double>(1, 2) = -1.0 * sin(Omega);
	R.at<double>(2, 0) = sin(Phi) * cos(Kappa) + cos(Phi) * sin(Omega) * sin(Kappa);
	R.at<double>(2, 1) = -1.0 * sin(Phi) * sin(Kappa) + cos(Phi) * sin(Omega) * cos(Kappa);
	R.at<double>(2, 2) = cos(Phi) * cos(Omega);
	for (int i = 0; i < PointPairs.size(); i++)
	{
		double X = PointPairs[i].X;
		double Y = PointPairs[i].Y;
		double Z = PointPairs[i].Z;
		double x = PointPairs[i].x;
		double y = PointPairs[i].y;
		double r_2 = pow(x - x0, 2) + pow(y - y0, 2);
		double delta_x = (x - x0) * (k1 * r_2 + k2 * r_2 * r_2) + p1 * (r_2 + pow((x - x0), 2)) + 2 * p2 * (x - x0) * (y - y0);
		double delta_y = (y - y0) * (k1 * r_2 + k2 * r_2 * r_2) + p2 * (r_2 + pow((y - y0), 2)) + 2 * p1 * (x - x0) * (y - y0);
		double Xbar = R.at<double>(0, 0) * (X - Xs) + R.at<double>(1, 0) * (Y - Ys) + R.at<double>(2, 0) * (Z - Zs);
		double Ybar = R.at<double>(0, 1) * (X - Xs) + R.at<double>(1, 1) * (Y - Ys) + R.at<double>(2, 1) * (Z - Zs);
		double Zbar = R.at<double>(0, 2) * (X - Xs) + R.at<double>(1, 2) * (Y - Ys) + R.at<double>(2, 2) * (Z - Zs);
		//外方位元素系数->线元素
		double a11 = (R.at<double>(0, 0) * f + R.at<double>(0, 2) * (x - x0 + delta_x)) / Zbar;
		double a12 = (R.at<double>(1, 0) * f + R.at<double>(1, 2) * (x - x0 + delta_x)) / Zbar;
		double a13 = (R.at<double>(2, 0) * f + R.at<double>(2, 2) * (x - x0 + delta_x)) / Zbar;
		double a21 = (R.at<double>(0, 1) * f + R.at<double>(0, 2) * (y - y0 + delta_y)) / Zbar;
		double a22 = (R.at<double>(1, 1) * f + R.at<double>(1, 2) * (y - y0 + delta_y)) / Zbar;
		double a23 = (R.at<double>(2, 1) * f + R.at<double>(2, 2) * (y - y0 + delta_y)) / Zbar;
		//外方位元素系数->角元素
		double a14 = (y - y0 + delta_y) * sin(Omega) - (((x - x0 + delta_x) / f) * ((x - x0 + delta_x) * cos(Kappa) - (y - y0 + delta_y) * sin(Kappa)) + f * cos(Kappa)) * cos(Omega);
		double a15 = -f * sin(Kappa) - ((x - x0 + delta_x) / f) * ((x - x0 + delta_x) * sin(Kappa) + (y - y0 + delta_y) * cos(Kappa));
		double a16 = y - y0 + delta_y;
		double a24 = -1.0 * (x - x0 + delta_x) * sin(Omega) - (((y - y0 + delta_y) / f) * ((x - x0 + delta_x) * cos(Kappa) - (y - y0 + delta_y) * sin(Kappa)) - f * sin(Kappa)) * cos(Omega);
		double a25 = -f * cos(Kappa) - ((y - y0 + delta_y) / f) * ((x - x0 + delta_x) * sin(Kappa) + (y - y0 + delta_y) * cos(Kappa));
		double a26 = -1.0 * (x - x0 + delta_x);
		//内方位元素系数
		double a17 = (x - x0 + delta_x) / f;
		//double a17 = -1.0 * Xbar / Zbar;
		double a18 = 1;
		double a19 = 0;
		double a27 = (y - y0 + delta_y) / f;
		//double a27 = -1.0 * Ybar / Zbar;
		double a28 = 0;
		double a29 = 1;

		double a110 = -1.0 * (x - x0 + delta_x) * r_2;                      //k1
		double a111 = -1.0 * (x - x0 + delta_x) * r_2 * r_2;				//k2
		double a112 = -1.0 * (2 * pow((x - x0 + delta_x), 2) + r_2);	    //p1
		double a113 = -2.0 * (y - y0 + delta_y) * (x - x0 + delta_x);		//p2

		double a210 = -1.0 * (y - y0 + delta_y) * r_2;                      //k1
		double a211 = -1.0 * (y - y0 + delta_y) * r_2 * r_2;                //k2        
		double a212 = -2.0 * (y - y0 + delta_y) * (x - x0 + delta_x);	    //p1
		double a213 = -1.0 * (2 * pow((y - y0 + delta_y), 2) + r_2);        //p2

		A.at<double>(2 * i, 0) = a11; A.at<double>(2 * i + 1, 0) = a21;
		A.at<double>(2 * i, 1) = a12; A.at<double>(2 * i + 1, 1) = a22;
		A.at<double>(2 * i, 2) = a13; A.at<double>(2 * i + 1, 2) = a23;
		A.at<double>(2 * i, 3) = a14; A.at<double>(2 * i + 1, 3) = a24;
		A.at<double>(2 * i, 4) = a15; A.at<double>(2 * i + 1, 4) = a25;
		A.at<double>(2 * i, 5) = a16; A.at<double>(2 * i + 1, 5) = a26;
		A.at<double>(2 * i, 6) = a17; A.at<double>(2 * i + 1, 6) = a27;
		A.at<double>(2 * i, 7) = a18; A.at<double>(2 * i + 1, 7) = a28;
		A.at<double>(2 * i, 8) = a19; A.at<double>(2 * i + 1, 8) = a29;
		A.at<double>(2 * i, 9) = a110; A.at<double>(2 * i + 1, 9) = a210;
		A.at<double>(2 * i, 10) = a111; A.at<double>(2 * i + 1, 10) = a211;
		A.at<double>(2 * i, 11) = a112; A.at<double>(2 * i + 1, 11) = a212;
		A.at<double>(2 * i, 12) = a113; A.at<double>(2 * i + 1, 12) = a213;

		l.at<double>(2 * i, 0) = x - (x0 - f * Xbar / Zbar - delta_x);
		l.at<double>(2 * i + 1, 0) = y - (y0 - f * Ybar / Zbar - delta_y);
	}
}

int main()
{
	//读取控制点数据和左右片标志点的像素坐标
	vector<ControlPoint> ControlPoints;
	vector<imgPoint> left_imgPoints;
	vector<imgPoint> right_imgPoints;
	char clptfile[] = "./data/clpts.txt";
	char left_file[] = "./data/left.txt";
	char right_file[] = "./data/right.txt";
	readClptData(clptfile, ControlPoints);
	readImgData(left_file, left_imgPoints);
	readImgData(right_file, right_imgPoints);

	//定义两张相片的内外方位元素
	orienParam left_orien;
	left_orien.x0 = 0;
	left_orien.y0 = 0;
	left_orien.f = 28;
	left_orien.Xs = 1500;
	left_orien.Ys = -200;
	left_orien.Zs = -1000;
	left_orien.Kappa = 0;
	left_orien.Omega = 0;
	left_orien.Phi = 0.3;
	left_orien.k1 = 0;
	left_orien.k2 = 0;
	left_orien.p1 = 0;
	left_orien.p2 = 0;

	orienParam right_orien;
	right_orien.x0 = 0;
	right_orien.y0 = 0;
	right_orien.f = 28;
	right_orien.Xs = 4500;
	right_orien.Ys = 200;
	right_orien.Zs = -1000;
	right_orien.Kappa = 0;
	right_orien.Omega = 0;
	right_orien.Phi = -0.2678;
	right_orien.k1 = 0;
	right_orien.k2 = 0;
	right_orien.p1 = 0;
	right_orien.p2 = 0;

	//将像素坐标(pixel)转换为图像坐标(mm)
	cvtPix2Img(left_imgPoints);
	cvtPix2Img(right_imgPoints);

	//生成点对
	vector<PointPair> left_PointPairs;
	vector<PointPair> right_PointPairs;
	GeneratePairs(left_imgPoints, ControlPoints, left_PointPairs);
	GeneratePairs(right_imgPoints, ControlPoints, right_PointPairs);

	////构建系数阵A和常数项l
	//Mat A(2 * right_PointPairs.size(), 13, CV_64FC1);
	//Mat l(2 * right_PointPairs.size(), 1, CV_64FC1);
	//while (true)
	//{
	//	//填充系数阵
	//	cal_Coefficient(A, l, right_PointPairs, right_orien);
	//	//最小二乘
	//	Mat X = (A.t() * A).inv() * A.t() * l;
	//	//更新外方位元素
	//	right_orien.Xs += X.at<double>(0, 0);
	//	right_orien.Ys += X.at<double>(1, 0);
	//	right_orien.Zs += X.at<double>(2, 0);
	//	right_orien.Phi += X.at<double>(3, 0);
	//	right_orien.Omega += X.at<double>(4, 0);
	//	right_orien.Kappa += X.at<double>(5, 0);
	//	right_orien.f += X.at<double>(6, 0);
	//	right_orien.x0 += X.at<double>(7, 0);
	//	right_orien.y0 += X.at<double>(8, 0);
	//	right_orien.k1 += X.at<double>(9, 0);
	//	right_orien.k2 += X.at<double>(10, 0);
	//	right_orien.p1 += X.at<double>(11, 0);
	//	right_orien.p2 += X.at<double>(12, 0);
	//	cout << X << endl;
	//	cout << l << endl;
	//	system("cls");
	//	if (abs(X.at<double>(3, 0)) < 0.0001 && abs(X.at<double>(4, 0)) < 0.0001 && abs(X.at<double>(5, 0)) < 0.0001)
	//		break;
	//}
	////输出定位参数right_orien
	//cout << "Xs:" << right_orien.Xs << endl;
	//cout << "Ys:" << right_orien.Ys << endl;
	//cout << "Zs:" << right_orien.Zs << endl;
	//cout << "Phi:" << right_orien.Phi << endl;
	//cout << "Omega:" << right_orien.Omega << endl;
	//cout << "Kappa:" << right_orien.Kappa << endl;
	//cout << "f:" << right_orien.f << endl;
	//cout << "x0:" << right_orien.x0 << endl;
	//cout << "y0:" << right_orien.y0 << endl;
	//cout << "k1:" << right_orien.k1 << endl;
	//cout << "k2:" << right_orien.k2 << endl;
	//cout << "p1:" << right_orien.p1 << endl;
	//cout << "p2:" << right_orien.p2 << endl;

	//构建系数阵A和常数项l
	Mat A(2 * left_PointPairs.size(), 13, CV_64FC1);
	Mat l(2 * left_PointPairs.size(), 1, CV_64FC1);
	while (true)
	{
		//填充系数阵
		cal_Coefficient(A, l, left_PointPairs, left_orien);
		//cout << "L = " << l << endl;
		//最小二乘
		Mat X = (A.t() * A).inv() * A.t() * l;
		//更新外方位元素
		left_orien.Xs += X.at<double>(0, 0);
		left_orien.Ys += X.at<double>(1, 0);
		left_orien.Zs += X.at<double>(2, 0);
		left_orien.Phi += X.at<double>(3, 0);
		left_orien.Omega += X.at<double>(4, 0);
		left_orien.Kappa += X.at<double>(5, 0);
		left_orien.f += X.at<double>(6, 0);
		left_orien.x0 += X.at<double>(7, 0);
		left_orien.y0 += X.at<double>(8, 0);
		left_orien.k1 += X.at<double>(9, 0);
		left_orien.k2 += X.at<double>(10, 0);
		left_orien.p1 += X.at<double>(11, 0);
		left_orien.p2 += X.at<double>(12, 0);
		//cout << X << endl;
		//cout << l << endl;
		//system("cls");
		if (abs(X.at<double>(3, 0)) < 0.0001 && abs(X.at<double>(4, 0)) < 0.0001 && abs(X.at<double>(5, 0)) < 0.0001)
			break;
	}
	//输出定位参数left_orien
	cout << "Xs: " << left_orien.Xs << endl;
	cout << "Ys: " << left_orien.Ys << endl;
	cout << "Zs: " << left_orien.Zs << endl;
	cout << "Phi: " << left_orien.Phi << endl;
	cout << "Omega: " << left_orien.Omega << endl;
	cout << "Kappa: " << left_orien.Kappa << endl;
	cout << "f: " << left_orien.f << endl;
	cout << "x0: " << left_orien.x0 << endl;
	cout << "y0: " << left_orien.y0 << endl;
	cout << "k1: " << left_orien.k1 << endl;
	cout << "k2: " << left_orien.k2 << endl;
	cout << "p1: " << left_orien.p1 << endl;
	cout << "p2: " << left_orien.p2 << endl;
	
	//精度统计

	return 0;
}