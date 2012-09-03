#include "SVM.h"
//#include <opencv2\core\core.hpp>

using namespace cv;

#define IsNotImplemented() CV_Error(CV_StsNotImplemented, "");

LSVM::LSVM()
{
}

LSVM::~LSVM()
{

}

bool LSVM::train(const Mat& trainData, const Mat& responses, const Mat& varIdx, 
	const Mat& sampleIdx, LSVMParams params)
{
	
	return false;
}

float LSVM::predict(const Mat& sample) const
{
	IsNotImplemented();
	return 0;
}

void LSVM::clear()
{
	IsNotImplemented();
}

void LSVM::save(const char* filename, const char* name) const
{
	IsNotImplemented();
}

void LSVM::load(const char* filename, const char* name)
{
	IsNotImplemented();
}

void LSVM::write(CvFileStorage* storage, const char* name) const
{
	IsNotImplemented();
}

void LSVM::read(CvFileStorage* storage, CvFileNode* node)
{
	IsNotImplemented();
}


