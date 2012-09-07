#include "SVM.h"
#include "bmrm.h"

#include <map>
#include <iostream>

using namespace cv;
using namespace std;

#define IsNotImplemented() CV_Error(CV_StsNotImplemented, "");

void GetNewIdx(const Mat& oldIdx, int allCount, Mat& newIdx);

LSVM::LSVM()
{
}

LSVM::~LSVM()
{

}

bool LSVM::train(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx, 
		const CvMat* sampleIdx, LSVMParams params)
{
	bool result;
	result = train(Mat(trainData), Mat(responses), Mat(varIdx), Mat(sampleIdx), params);
	return result;
}

bool LSVM::train(const Mat& trainData, const Mat& responses, const Mat& _varIdx, 
	const Mat& _sampleIdx, LSVMParams params)
{
	clear();

	CV_Assert(	CV_MAT_TYPE(trainData.type())==CV_32FC1 && 
				CV_MAT_TYPE(responses.type())==CV_32FC1 &&
				CV_MAT_TYPE(_varIdx.type())==CV_32SC1 &&
				CV_MAT_TYPE(_sampleIdx.type())==CV_32SC1);
	CV_Assert(trainData.rows==responses.rows);
	CV_Assert(_varIdx.empty() || _varIdx.rows==1);
	CV_Assert(_sampleIdx.empty() || _sampleIdx.rows==1);
	CV_Assert(params.C>0);

	Mat varIdx;
	GetNewIdx(_varIdx, trainData.cols, varIdx);
	Mat sampleIdx;
	GetNewIdx(_sampleIdx, trainData.rows, sampleIdx);
	int sampleCount = sampleIdx.cols;
	int varCount = varIdx.cols;

	for(int i = 0;i<sampleCount;i++)
	{
		int idx = sampleIdx.at<int>(i);
		float label = responses.at<float>(idx);
		map<float,int>::iterator iter = classLabels.find(label);
		if(iter==classLabels.end())
		{
			pair<float,int> newPair(label, 1);
			classLabels.insert(newPair);
		}
		else
		{
			(*iter).second++;
		}
	}
	if(classLabels.size()!=2)
	{
		CV_Error(CV_StsBadArg, "At the moment only binary classification is supported");
	}
	


	Mat denseSamples(sampleCount, varCount, CV_32FC1);
	Mat denseResponses(sampleCount, 1, CV_32FC1);
	
	for(int i = 0;i<sampleCount;i++)
	{
		int row = sampleIdx.at<int>(i);

		float label = responses.at<float>(row);
		denseResponses.at<float>(i) = GetUnitLabel(label);
		for(int j = 0;j<varCount;j++)
		{
			int col = varIdx.at<int>(j);

			float value = trainData.at<float>(row, col);
			denseSamples.at<float>(i, j) = value;
		}
	}

	
	cout << sampleIdx << endl << endl;

	//временно
	normal = Mat(1, trainData.cols, CV_32FC1);
	normal = -1;
	
	return false;
}

bool LSVM::train(CvMLData* trainData, LSVMParams params)
{
	bool result;
	if(trainData->get_missing())
	{
		CV_Error(CV_StsNotImplemented, "LSVM does not support missing values");
	}
	result = train(trainData->get_values(), trainData->get_responses(), 
		trainData->get_var_idx(), trainData->get_train_sample_idx(), params);
	return result;
}

float LSVM::predict(const Mat& sample) const
{
	float result;
	/*
	проверка sample
	*/
	if(normal.dot(sample)>=0)
	{
		result = 1;
	}
	else
	{
		result = -1;
	}
	return result;
}

float LSVM::predict(const CvMat* sample) const
{
	Mat _sample(sample);
	return predict(_sample);
}

float LSVM::calc_error(CvMLData* data, int type, std::vector<float>* resp) const
{
	Mat values = data->get_values();
	Mat responses = data->get_responses();
	Mat sampleIdx;
	if(type==CV_TRAIN_ERROR)
	{
		sampleIdx = data->get_train_sample_idx();
	}
	else  // CV_TEST_ERROR
	{
		sampleIdx = data->get_test_sample_idx();
	}
	int sampleCount = sampleIdx.cols;

	float error = 0;
	if(resp)
	{
		resp->resize(sampleCount);
	}
	for(int i = 0;i<sampleIdx.cols;i++)
	{
		int idx = sampleIdx.at<int>(i);
		float predResp = predict(values.row(idx));
		if(resp)
		{
			(*resp)[i] = predResp;
		}
		float trueResp = GetUnitLabel(responses.at<float>(idx));
		error += predResp-trueResp!=0;
	}
	error /= sampleCount;
 	return error;
}

void LSVM::clear()
{
	type = -1;
	normal.release();
	classLabels.clear();
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

float LSVM::GetUnitLabel(float label) const
{
	map<float, int>::const_iterator iter = classLabels.begin();
	if((*iter).first==label)
	{
		return 1;
	}
	else
	{
		return -1;
	}
}


void GetNewIdx(const Mat& oldIdx, int allCount, Mat& newIdx)
{
	if(oldIdx.empty())
	{
		newIdx.create(1, allCount, CV_32SC1);
		for(int i = 0;i<allCount;i++)
		{
			newIdx.at<int>(i) = i;
		}
	}
	else
	{
		newIdx = oldIdx;
	}
}


