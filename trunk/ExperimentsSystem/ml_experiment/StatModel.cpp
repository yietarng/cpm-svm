#include "StatModel.h"
#include "ErrorHandling.h"


#include <time.h>
#include <list>


using namespace std;
using namespace mle;


void MatToVector(const cv::Mat& mat, const cv::Mat& mask, std::vector<float>& v);
cv::Mat GetTestMask(const CvMLData* data);
void MixArray(int* array, int n);
void GetMasks(const CvMLData* data, 
			  std::vector<cv::Mat>& trainMasks, std::vector<cv::Mat>& validMasks, 
			  cv::Mat& testMask, 
			  int cvFolds);


StatModel::StatModel()
{
	algorithm = -1;
	task = -1;
	model = 0;
}

void StatModel::Clear()
{
	algorithm = -1;
	task = -1;
	classes.clear();
	delete model;
}

StatModel::~StatModel()
{
	Clear();
}

void StatModel::Train(MLData *data, const StatModelParams& params)
{
	Clear();
	algorithm = params.Algorithm();
	task = data->get_var_type(data->get_response_idx());
	switch(algorithm)
	{
	case DECISION_TREE :
		model = new CvDTree();
		dynamic_cast<CvDTree*>(model)->train(data, params); 
		break;

	case GRADIENT_BOOSTED_TREES : 
		model = new CvGBTrees;
		dynamic_cast<CvGBTrees*>(model)->train(data, params); 
		break;

	case SUPPORT_VECTOR_MACHINE : 
		CV_Assert(data->get_response_idx()==data->get_values()->cols-1 && 
			data->get_missing()==0);

		model = new CvSVM;
		if(data->get_missing())
		{
			ES_Error("SVM не поддерживает пропущенные значения");
		}
		CvMat trainData(*data->get_values());
		trainData.cols = data->get_values()->cols-1;
		dynamic_cast<CvSVM*>(model)->train(&trainData, data->get_responses(), 
			data->get_var_idx(), data->get_train_sample_idx(), params); 
		break;

	case RANDOM_TREES :
		model = new CvRTrees;
		dynamic_cast<CvRTrees*>(model)->train(data, params); 
		break;

	case EXTREMELY_RANDOMIZED_TREES : 
		model = new CvERTrees;
		dynamic_cast<CvERTrees*>(model)->train(data, params); 
		break;

	case MULTI_LAYER_PERCEPTRON:
		{
			CV_Assert(data->get_response_idx()==data->get_values()->cols-1 && 
				data->get_missing()==0);


			//подготовка к созданию и обучению нейронной сети

			MLP_Params mlpParams = params;

			CvMat trainData(*data->get_values());
			trainData.cols = data->get_values()->cols-1;

			cv::Mat layerSizes(1, mlpParams.hiddenLayersCount+2, CV_32SC1);
			const int inputsCount = data->get_values()->cols-1;
			layerSizes.at<int>(0) = inputsCount;
			for(int i = 0;i<mlpParams.hiddenLayersCount;i++)
			{
				layerSizes.at<int>(1+i) = mlpParams.hiddenLayerSize;
			}

			cv::Mat responses;

			if(task==Classification)
			{
				cv::Mat originalResp(data->get_responses());
				vector<float> v(originalResp.rows);
				for(int i = 0;i<v.size();i++)
				{
					v[i] = originalResp.at<float>(i);
				}
				AddClasses(v, classes);
				
				responses = cv::Mat(originalResp.rows, classes.size(), CV_32FC1);
				responses = 0;
				for(int i = 0;i<responses.rows;i++)
				{
					int classIdx = classes[originalResp.at<float>(i)];
					responses.at<float>(i, classIdx) = 1;
				}
			}
			else // task==Regression
			{
				responses = data->get_responses();
			}
			layerSizes.at<int>(layerSizes.cols-1) = responses.cols;


			//создание и обучение нейронной сети

			model = new CvANN_MLP(layerSizes, 
				mlpParams.activateFunc, 
				mlpParams.fParam1, mlpParams.fParam2);

			CvMat temp = responses;
			dynamic_cast<CvANN_MLP*>(model)->train(&trainData, &temp, 0, 
				data->get_train_sample_idx(), mlpParams.trainParams);

		}
		break;

	default: 
		ES_Assert(0);
		task = -1;
	}
}

Measures* StatModel::CalcMeasures(MLData* data, int type, Time* predictingTime, 
								  std::vector<float>* resp) const
{
	ES_Assert(type==CV_TEST_ERROR || type==CV_TRAIN_ERROR);
	vector<float>* predicted;
	if(resp!=0)
	{
		predicted = resp;
	}
	else
	{
		predicted = new vector<float>;
	}
	Time* time;
	if(predictingTime!=0)
	{
		time = predictingTime;
	}
	else
	{
		time = new Time;
	}
	switch(algorithm)
	{
	case DECISION_TREE :
		time->Start();
		dynamic_cast<CvDTree*>(model)->calc_error(data, type, predicted); 
		time->Finish();
		break;

	case GRADIENT_BOOSTED_TREES : 
		time->Start();
		dynamic_cast<CvGBTrees*>(model)->calc_error(data, type, predicted); 
		time->Finish();
		break;

	case SUPPORT_VECTOR_MACHINE : 
		{
			CvMat samples = data->SubData(type);
			predicted->resize(samples.rows);
			CvMat results = cvMat(1, samples.rows, CV_32FC1, &(*predicted)[0]);
			
			time->Start();
			dynamic_cast<CvSVM*>(model)->predict(&samples, &results);
			time->Finish();
		}
		break;

	case RANDOM_TREES :
		time->Start();
		dynamic_cast<CvRTrees*>(model)->calc_error(data, type, predicted); 
		time->Finish();
		break;

	case EXTREMELY_RANDOMIZED_TREES :
		time->Start();
		dynamic_cast<CvERTrees*>(model)->calc_error(data, type, predicted); 
		time->Finish();
		break;

	case MULTI_LAYER_PERCEPTRON:
		{
			
			cv::Mat samples = data->SubData(type);
			cv::Mat results;
			predicted->resize(samples.rows);

			if(task==Classification)
			{
				results = cv::Mat(samples.rows, classes.size(), CV_32FC1);
			}
			else // task==Regression
			{
				results = cv::Mat(samples.rows, 1, CV_32FC1, &(*predicted)[0]);
			}
			
			time->Start();
			dynamic_cast<CvANN_MLP*>(model)->predict(samples, results);
			time->Finish();
			
			if(task==Classification)
			{
				for(int i = 0;i<samples.rows;i++)
				{
					float maxValue = FLT_MIN;
					int maxIdx = 0;
					for(int k = 0;k<classes.size();k++)
					{
						float value = results.at<float>(i,k);
						if(value>maxValue)
						{
							maxValue = value;
							maxIdx = k;
						}
					}

					map<float, int>::const_iterator iter = classes.begin();
					for(int k = 0;k<maxIdx;k++)
					{
						iter++;
					}
					(*predicted)[i] = iter->first;
				}
			}
		}
		break;

	default: ES_Error("");
	}

	const CvMat* mask;
	switch(type)
	{
		case CV_TEST_ERROR	: mask = data->get_test_sample_idx();	break;
		case CV_TRAIN_ERROR	: mask = data->get_train_sample_idx();	break;
	}
	vector<float> actual;
	MatToVector(data->get_responses(), mask, actual);
	Measures* mes;
	switch(task)
	{
		case Classification	: mes = new ClMeasures(); break;
		case Regression		: mes = new ReMeasures(); break;
	}
	mes->Calculate(actual, *predicted);
	if(resp==0)
	{
		delete predicted;
	}
	return mes;
}

int StatModel::Task() const
{
	return task;
}


void StatModel::Fit(MLData* data, StatModelParams& params, std::ostream* stream)
{
	ES_Assert(params.isCVParams());

	const int cvFolds = params.CVFolds();
	
	cv::Mat testMask;
	vector<cv::Mat> trainMasks, validMasks;
	GetMasks(data, trainMasks, validMasks, testMask, cvFolds);
	
	if(stream!=0)
	{
		*stream << "перебор параметров..." << endl;
	}

	params.InitVaryingByGrid();
	float minError = FLT_MAX;
	do
	{
		float error = 0;
		for(int i = 0;i<cvFolds;i++)
		{
			data->set_train_test_split(trainMasks[i]);
			Train(data, params);
			data->set_train_test_split(validMasks[i]);
			error += CalcError(data, CV_TEST_ERROR);
		}
		error /= cvFolds;

		if(stream!=0)
		{
			*stream << endl;
			params.PrintGridValues(*stream);
			Print("ошибка", error, *stream);
		}

		if(minError>error)
		{
			minError = error;
			params.SaveGridValues();
		}
	}
	while(params.VaryByGrid());

	params.SetSavedGridValues();
	data->set_train_test_split(testMask);
	Train(data, params);
}



const CvStatModel* StatModel::CvPointer() const
{
	return model;
}

float StatModel::CalcError(MLData* data, int type) const
{
	Measures* mes = CalcMeasures(data, type);
	float error = mes->Error();
	delete mes;
	return error;
}

//===============================================================================

void MatToVector(const cv::Mat& mat, const cv::Mat& mask, std::vector<float>& v)
{
	v.clear();
	for(int i = 0;i<mask.cols;i++)
	{
		int idx = mask.at<int>(i);
		v.push_back(mat.at<float>(idx));
	}
}


cv::Mat GetTestMask(const CvMLData* data)
{
	const int sampleCount = data->get_values()->rows;
	cv::Mat testMask(1, sampleCount, CV_8UC1);
	testMask = 0;

	const cv::Mat testSampleIdx = data->get_test_sample_idx();
	for(int j = 0;j<testSampleIdx.cols;j++)
	{
		testMask.at<uchar>(testSampleIdx.at<int>(j)) = 1;
	}
	return testMask;
}

void MixArray(int* array, int n)
{
	cv::RNG* rng = &cv::theRNG();
    for (int i = 0;i<n;i++)
    {
        int a = (*rng)(n);
        int b = (*rng)(n);
        int t;
        CV_SWAP(array[a], array[b], t);
    }
	
}

void GetMasks(const CvMLData* data, 
						 std::vector<cv::Mat>& trainMasks, 
						 std::vector<cv::Mat>& validMasks, 
						 cv::Mat& testMask, int cvFolds)
{
	testMask = GetTestMask(data);

	trainMasks.resize(cvFolds);
	validMasks.resize(cvFolds);

	const CvMat* trainSampleIdx = data->get_train_sample_idx();
	const int trainSampleCount = trainSampleIdx->cols;
	const int sampleCount = data->get_values()->rows; 

	int* mixedTrainSampleIdx = new int[trainSampleCount];
	memcpy(mixedTrainSampleIdx, trainSampleIdx->data.i, trainSampleCount*sizeof(int));
	MixArray(mixedTrainSampleIdx, trainSampleCount);

	vector<int> bounds(cvFolds+1);
	int blockSize = trainSampleCount/cvFolds;
	int incCount = trainSampleCount%cvFolds;
	bounds[0] = 0;
	for(int i = 1;i<=incCount;i++)
	{
		bounds[i] = bounds[i-1]+blockSize+1;
	}
	for(int i = incCount+1;i<=cvFolds;i++)
	{
		bounds[i] += bounds[i-1]+blockSize;
	}

	for(int i = 0;i<cvFolds;i++)
	{
		validMasks[i] = cv::Mat(1, sampleCount, CV_8UC1);
		validMasks[i] = 0;
		
		trainMasks[i] = cv::Mat(1, sampleCount, CV_8UC1);
		memcpy(trainMasks[i].ptr(), testMask.ptr(), sampleCount);
		for(int j = bounds[i];j<bounds[i+1];j++)
		{
			int idx = mixedTrainSampleIdx[j];
			validMasks[i].at<uchar>(idx) = 1;
			trainMasks[i].at<uchar>(idx) = 1;
		}
	}
	delete[] mixedTrainSampleIdx;
}

