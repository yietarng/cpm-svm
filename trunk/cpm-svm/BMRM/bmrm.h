#pragma once


#include "functions.h"


#include <ml.h>


class BMRM
{
public:
	BMRM(const Regularizer* _reg, const EmpiricalRisk* _risk, real _lambda);
	~BMRM();

	cv::Mat FindMin(real epsilon, const cv::Mat& w0);

private:
	int dim;
	real lambda;
	const Regularizer* reg;
	const EmpiricalRisk* risk;
};





