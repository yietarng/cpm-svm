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
	Print("бенчмарк ", benchmarkFilename, log);
	if(patternsLoaded==false)
	{
		log << "Ўаблоны не загружены" << endl;
		return;
	}
	MLBenchmark benchmark;
	if(0==benchmark.Load(benchmarkFilename))
	{
		log << "файл " << benchmarkFilename << " не найден" << endl;
		return;
	}
	Print("директори€ наборов данных", benchmark.data.directory, log);
	Print("директори€ алгоритмов", benchmark.algorithms.directory, log);

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
					Print("набор данных", *dataIter, log);
					Print("алгоритм", *algIter, log);

					string algPath = benchmark.algorithms.directory+"\\"+*algIter;
					StatModelParams params(algPath);
					switch(data.get_var_type(data.get_response_idx()))
					{
					case Classification : Print("тип задачи", "классификаци€", log);break;
					case Regression		: Print("тип задачи", "регресси€", log);break;
					}

					mle::StatModel model;
					Time time;string timeStr;
					if(params.isCVParams())
					{
						timeStr = "врем€ подгонки модели";
						log << endl;
						ostream* cvErrorsStream = 0;
						if(printCVErrors)
						{
							cvErrorsStream = &log;
						}

						time.Start();
						model.Fit(&data, params, cvErrorsStream);
						time.Finish();

						log << endl << "оптимальные параметры:" << endl;
						params.PrintGridValues(log);
						log << endl;
					}
					else
					{
						timeStr = "врем€ обучени€";
						time.Start();
						model.Train(&data, params);
						time.Finish();
					}
					Print(timeStr, log);log << time << endl;
					
					Measures* mes;
					log << endl <<  "показатели на обучающей выборке:" << endl;
					mes = model.CalcMeasures(&data, CV_TRAIN_ERROR, &time);
					log << *mes;delete mes;
					Print("врем€ вычислени€ ответов", log);log << time << endl;

					log << endl <<  "показатели на тестовой выборке:" << endl;
					mes = model.CalcMeasures(&data, CV_TEST_ERROR, &time);
					log << *mes;delete mes;
					Print("врем€ вычислени€ ответов", log);log << time << endl;
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
	log << "Ёксперимент завершен" << endl;
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


