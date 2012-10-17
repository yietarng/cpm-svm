#include "functions.h"
//#include "libqp.h"

#include <iostream>

using namespace cv;
using namespace std;

/*

float Max(float a, float b);

const vector<Mat>* Aptr;



float Regularizer::Value(const cv::Mat &w) const
{
	return -1;
}



float Regularizer::MinJt(const std::vector<cv::Mat> &A, const std::vector<float> &b, 
				   cv::Mat &wMin) const
{
	Aptr = &A;
	//libqp_gsmo_solver(
	return -1;
}

EmpiricalRisk::EmpiricalRisk(const cv::Mat &_samples, const cv::Mat &_responses)
{
	samples = _samples;
	responses = _responses;
}

float EmpiricalRisk::LossFunction(int idx, const cv::Mat &w, cv::Mat& subgradient) const
{
	Mat x = samples.row(idx);
	x = x.t();
	float y = responses.at<float>(idx);
	cv::Size size = w.size();
	float distance = float( 1-y*x.dot(w) );
	subgradient.create(w.rows, 1, CV_32FC1);
	if(distance>=0)
	{
		subgradient = -y*x;
		return distance;
	}
	else
	{
		subgradient = 0;
		return 0;
	}
}

void EmpiricalRisk::GetSubgradient(const cv::Mat &w, cv::Mat& subgradient) const
{
	
}

float EmpiricalRisk::Value(const cv::Mat &w, cv::Mat& subgradient) const
{
	int m = samples.rows;
	float risk = 0;
	subgradient.create(w.rows, 1, CV_32FC1);
	subgradient = 0;
	Mat s(w.rows, 1, CV_32FC1);
	for(int i = 0;i<m;i++)
	{
		risk += LossFunction(i, w, s);
		subgradient += s;
	}
	risk /= m;
	return risk;
}


float Max(float a, float b)
{
	if(a>=b)
	{
		return a;
	}
	else
	{
		return b;
	}
}

*/