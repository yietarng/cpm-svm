#include "MLExperiment.h"
#include "MLParameters.h"
#include "StatModel.h"
#include "MLData.h"
#include "ErrorHandling.h"
#include "ToString.h"
#include "Measures.h"
#include "MLBenchmark.h"


#include <ml.h>
#include <vector>


using namespace std;
using namespace cv;
using namespace mle;

bool patternsLoaded = false;

void SubVector(vector<float>& vec, const CvMat* column, const CvMat* idxRow);
void PrintTime(ostream& log);

void LoadPatterns(const std::string& directory)
{
	StatModelParams::LoadPatterns(directory);
	MLData::LoadPatterns(directory); 
	MLBenchmark::LoadPatterns(directory);
	patternsLoaded = true;
}

void Experiment(const std::string& benchmarkFilename, std::ostream& log, bool printCVErrors)
{
	PrintTime(log);
	Print("�������� ", benchmarkFilename, log);
	if(patternsLoaded==false)
	{
		log << "������� �� ���������" << endl;
		return;
	}
	MLBenchmark benchmark;
	if(0==benchmark.Load(benchmarkFilename))
	{
		log << "���� " << benchmarkFilename << " �� ������" << endl;
		return;
	}
	Print("���������� ������� ������", benchmark.data.directory, log);
	Print("���������� ����������", benchmark.algorithms.directory, log);

	int dataCounter = 0;
	for(list<string>::const_iterator dataIter  = benchmark.data.name.begin();
		dataIter!=benchmark.data.name.end();dataIter++, dataCounter++)
	{
		MLData data;
		string dataPath = benchmark.data.directory+"\\"+*dataIter;
		try
		{
			data.Load(dataPath);

			int algCounter = 0;
			for(list<string>::const_iterator algIter = benchmark.algorithms.name.begin();
			algIter!=benchmark.algorithms.name.end();algIter++,algCounter++)
			{
				try
				{
					log << endl;
					log << "====================================================";
					log << endl << endl;
					Print("����� ������", *dataIter, log);
					Print("��������", *algIter, log);

					string algPath = benchmark.algorithms.directory+"\\"+*algIter;
					StatModelParams params(algPath);
					switch(data.get_var_type(data.get_response_idx()))
					{
					case Classification : Print("��� ������", "�������������", log);break;
					case Regression		: Print("��� ������", "���������", log);break;
					}

					mle::StatModel model;
					Time time;string timeStr;
					if(params.isCVParams())
					{
						timeStr = "����� �������� ������";
						log << endl;
						ostream* cvErrorsStream = 0;
						if(printCVErrors)
						{
							cvErrorsStream = &log;
						}

						time.Start();
						model.Fit(&data, params, cvErrorsStream);
						time.Finish();

						log << endl << "����������� ���������:" << endl;
						params.PrintGridValues(log);
						log << endl;
					}
					else
					{
						timeStr = "����� ��������";
						time.Start();
						model.Train(&data, params);
						time.Finish();
					}
					Print(timeStr, log);log << time << endl;
					
					Measures* mes;
					log << endl <<  "���������� �� ��������� �������:" << endl;
					mes = model.CalcMeasures(&data, CV_TRAIN_ERROR, &time);
					log << *mes;delete mes;
					Print("����� ���������� �������", log);log << time << endl;

					log << endl <<  "���������� �� �������� �������:" << endl;
					mes = model.CalcMeasures(&data, CV_TEST_ERROR, &time);
					log << *mes;delete mes;
					Print("����� ���������� �������", log);log << time << endl;
				}
				catch(cv::Exception& error)
				{
					log << error.what();
				}
			}
		}
		catch(cv::Exception& error)
		{
			log << error.what();
		}
		log << endl;
	}
	log << "====================================================" << endl << endl;
	log << "����������� ��������" << endl;
	PrintTime(log);
}

void CreateMLDataHeaders(const std::string& directory, std::ostream& stream, 
			const std::vector<std::string>& extentions)
{
	mle::MLData::CreateMLDataHeaders(directory, stream, extentions);
}

void PrintTime(ostream& log)
{
	SYSTEMTIME sTime;
	GetLocalTime(&sTime);
	log << sTime.wDay << "." << sTime.wMonth << "." << sTime.wYear << " ";
	log << sTime.wHour << ":" << sTime.wMinute;
	log << endl;
}

void SubVector(vector<float>& vec, const CvMat* column, const CvMat* idxRow)
{
	const int* idx = idxRow->data.i;
	const float* ptr = column->data.fl;
	for(int i = 0;i<idxRow->cols;i++)
	{
		vec.push_back(ptr[idx[i]]);
	}
}


