#include "svm.h"
#include "bmrm.h"

#include <math.h>
#include <assert.h>
#include <map>

#include <iostream>


using namespace cv;
using namespace std;


float Sign(float value);
void GetLabels(const Mat& resp, const Mat& sampleIdx, map<float, int>& labels); 



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
	return Sign(res);
}


void BSVM::clear()
{
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

	map<float,int> labels;
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
		//cout << endl;
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


void GetLabels(const Mat& resp, const Mat& sampleIdx, map<float, int>& labels)
{
	labels.clear();
	for(int i = 0;i<sampleIdx.cols;i++)
	{
		int idx = sampleIdx.at<int>(i);
		float key = resp.at<float>(idx);
		if(labels.find(key)==labels.end())
		{
			int value = labels.size();
			labels[key] = value;
		}
	}
}