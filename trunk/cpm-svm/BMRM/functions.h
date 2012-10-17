#pragma once

#include <cxcore.h>
#include <vector>

//==============================
typedef float real;
#define matElementType CV_32FC1 
//==============================



class Regularizer
{
public:
	virtual real Value(const cv::Mat& w) const = 0;
	virtual cv::Mat ArgMinJt(const std::vector<cv::Mat>& A, const std::vector<real>& b,
		real lambda) const = 0;
};


class LossFunction
{
public:
	virtual real Value(const cv::Mat& x, const real y, 
		const cv::Mat& w) const = 0;

	virtual cv::Mat Subgradient(const cv::Mat& x, const real y, 
		const cv::Mat& w) const = 0;
};


class EmpiricalRisk
{
public:

	EmpiricalRisk(const cv::Mat& _samples, const cv::Mat& _responses, 
		const LossFunction* _lossF);
	real Value(const cv::Mat& w) const;
	cv::Mat Subgradient(const cv::Mat& w) const;
	int Dim() const;
private:
	
	const LossFunction* lossF;
	cv::Mat samples;
	cv::Mat responses;
};