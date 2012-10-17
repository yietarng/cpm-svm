#include "bmrm.h"

using namespace cv;

BMRM::BMRM(const Regularizer *_reg, const EmpiricalRisk *_risk, real _lambda)
: reg(_reg), risk(_risk), lambda(_lambda) 
{
	dim = risk->Dim();
}


cv::Mat BMRM::FindMin(real epsilon, const cv::Mat &w0)
{
	Mat wt(1, dim, matElementType);

	

	return wt;
}