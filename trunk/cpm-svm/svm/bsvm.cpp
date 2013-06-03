#include "bsvm.h"
#include "bmrm.h"

#include <math.h>
#include <assert.h>
#include <map>

#include <iostream>


using namespace cv;
using namespace std;


float Sign(float value);
void GetLabels(const Mat& resp, const Mat& sampleIdx, vector<float>& labels); 



BSVM::BSVM()
{
}


BSVM::~BSVM()
{
	clear();
}


float BSVM::predict(const cv::Mat& sample) const
{
	assert(sampleDim!=0 && sample.cols==sampleDim && sample.rows==1 && sample.type()==CV_32FC1);
	float res = 0;
	for(int i = 0;i<var_idx.size();i++)
	{
		int idx = var_idx[i];
		res += betta[i]*sample.at<float>(idx);
	}
	float sign = Sign(res);

	float resp;
	if(sign==1)
	{
		resp = labels[0];
	}
	else // sign==-1
	{
		resp = labels[1];
	}
	return resp;
}


float BSVM::calc_error(CvMLData* trainData, int type, std::vector<float>* resp)
{
	Mat sampleIdx;
	if(type==CV_TRAIN_ERROR)
	{
		sampleIdx = trainData->get_train_sample_idx();
	}
	else // type==CV_TEST_ERROR
	{
		sampleIdx = trainData->get_test_sample_idx();
	}

	Mat values = trainData->get_values();
	Mat responses = trainData->get_responses();

	float error = 0;
	for(int i = 0;i<sampleIdx.cols;i++)
	{
		int idx = sampleIdx.at<int>(i);
		Mat sample = values.row(idx);
		float response = responses.at<float>(idx);

		float pred = predict(sample);
		if(resp!=0)
		{
			resp->push_back(pred);
		}
		if(pred!=response)
		{
			error += 1;
		}
	}
	error /= sampleIdx.cols;

	return error;
}


void BSVM::clear()
{
	labels.clear();
	betta.clear();
	var_idx.clear();
	sampleDim = 0;
}


bool BSVM::train(const cv::Mat& trainData, const cv::Mat& responses, const cv::Mat& varIdx, 
		const cv::Mat& sampleIdx, BSVMParams params)
{
	//Здесь должна быть проверка исходных данных

	sampleDim = trainData.cols;

	int n = varIdx.cols;
	betta.resize(n);
	var_idx.resize(n);
	for(int i = 0;i<n;i++)
	{
		var_idx[i] = varIdx.at<int>(i);
	}

	
	GetLabels(responses, sampleIdx, labels);
	assert(labels.size()==2);

	int m = sampleIdx.cols;
	float* y = new float[m];
	for(int i = 0;i<m;i++)
	{
		int idx = sampleIdx.at<int>(i);
		float resp = responses.at<float>(idx);
		if(labels[resp]==0)
		{
			y[i] = 1;
		}
		else
		{
			y[i] = -1;
		}
	}

	float** x = new float*[m];
	for(int i = 0;i<m;i++)
	{
		x[i] = new float[n];
		
		int row = sampleIdx.at<int>(i);
		for(int j = 0;j<n;j++)
		{
			int col = varIdx.at<int>(j);
			x[i][j] = trainData.at<float>(row, col);
			//cout << x[i][j] << '\t';
		}
		//cout << y[i] << endl;
	}

	if(params.solver==BSVMParams::BMRM_SOLVER)
	{
		BMRMSolver bmrm((const float**)x, y, n, m, params.lambda);
		bmrm.Solve(params.epsilon, params.maxIter, &betta[0]);
	}
	else
	{
		assert(0);
	}

	delete[] y;
	for(int i = 0;i<m;i++)
	{
		delete[] x[i];
	}
	delete[] x;

	return true;
}


bool BSVM::train(CvMLData* trainData, BSVMParams params)
{
	const CvMat* ptr = trainData->get_train_sample_idx();
//	CvMat(
	train(trainData->get_values(), trainData->get_responses(), trainData->get_var_idx(),
		trainData->get_train_sample_idx(), params);
	return true;
}


//====================================================================

float Sign(float value)
{
	if(value<0)
	{
		return -1;
	}
	else // value>=0
	{
		return 1;
	}
}


void GetLabels(const Mat& resp, const Mat& sampleIdx, vector<float>& labels)
{
	labels.clear();
	for(int i = 0;i<sampleIdx.cols;i++)
	{
		int idx = sampleIdx.at<int>(i);
		float label = resp.at<float>(idx);
	}
}