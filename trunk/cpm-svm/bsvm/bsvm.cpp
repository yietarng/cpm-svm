#include "bsvm.h"
#include "bmrm.h"

#include <math.h>
#include <assert.h>
#include <map>

#include <iostream>


using namespace cv;
using namespace std;



//#define SHOW_DATA


float Sign(float value);
void GetLabels(const Mat& resp, const Mat& sampleIdx, vector<float>& labels); 
double Normalize(VarInfo info, double value);


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
	double res = 0;
	for(int i = 0;i<var_idx.size();i++)
	{
		int idx = var_idx[i];
		double value = double( sample.at<float>(idx) );
		res += betta[i]*Normalize(varInfo[i], value);
	}
	float sign = Sign( float(res) );

	float resp;
	if(sign==-1)
	{
		resp = labels[0];
	}
	else // sign==1
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
	varInfo.clear();
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
	double* y = new double[m];
	for(int i = 0;i<m;i++)
	{
		int idx = sampleIdx.at<int>(i);
		float resp = responses.at<float>(idx);
		if(labels[0]==resp)
		{
			y[i] = -1;
		}
		else
		{
			y[i] = 1;
		}
	}

	double** x = new double*[m];
	for(int i = 0;i<m;i++)
	{
		x[i] = new double[n];
		
		int row = sampleIdx.at<int>(i);
		for(int j = 0;j<n;j++)
		{
			int col = varIdx.at<int>(j);
			x[i][j] = double( trainData.at<float>(row, col) );

			#ifdef SHOW_DATA
			cout << x[i][j] << '\t';
			#endif SHOW_DATA
		}
		#ifdef SHOW_DATA
		cout << y[i] << endl;
		#endif SHOW_DATA
	}


	varInfo.resize(n);
	for(int j = 0;j<n;j++)
	{
		VarInfo& info = varInfo[j];
		for(int i = 0;i<m;i++)
		{
			double value = x[i][j];

			info.mean += value;
			if(info.max<value)
			{
				info.max = value;
			}
			if(info.min>value)
			{
				info.min = value;
			}
		}
		info.mean /= m;
	}

	for(int i = 0;i<m;i++)
	{
		for(int j = 0;j<n;j++)
		{
			double& value = x[i][j];
			value = Normalize(varInfo[j], value);
		}
	}

	if(params.solver==BSVMParams::BMRM_SOLVER)
	{
		BMRMSolver bmrm((const double**)x, y, n, m, params.lambda);
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

		bool endFlag = true;
		vector<float>::iterator iter = labels.begin();
		for(int i = 0;i<labels.size();i++)
		{
			if(label<=labels[i])
			{
				if(label<labels[i])
				{
					labels.insert(iter, label);
				}
				endFlag = false;
				break;
			}
			iter++;
		}

		if(endFlag)
		{
			labels.insert(iter, label);
		}
	}
}

double Normalize(VarInfo info, double value)
{
	return (value-info.min)/(info.max-info.min)*2-1;
}

//================================================================
VarInfo::VarInfo() : mean(0), max(-DBL_MAX), min(DBL_MAX) {}