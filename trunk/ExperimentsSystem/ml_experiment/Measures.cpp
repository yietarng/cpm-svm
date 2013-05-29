#include "Measures.h"
#include "ErrorHandling.h"
#include "ToString.h"


#include "math.h"


using namespace std;
using namespace cv;


/*
void CalcAccuracy(const std::vector<float>& actual, const std::vector<float>& predicted)
{
	int n = actual.size();
	accuracy = 0;
	for(int i = 0;i<n;i++)
	{
		accuracy += AreEqual(actual[i], predicted[i]);
	}
	accuracy = accuracy/n;
}
*/


int Measures::AreEqual(float a, float b)
{
	return (fabs(double(a)-b) <= FLT_EPSILON);
}

std::ostream& operator<<(std::ostream& stream, const Measures& mes)
{
	mes.PrintObject(stream);
	return stream;
}



//============================================
ClMeasures::ClMeasures() 
{
	Clear();
}

ClMeasures::ClMeasures(const std::vector<float>& actual, const std::vector<float>& predicted) 
{
	Calculate(actual, predicted);
}

void ClMeasures::Calculate(const std::vector<float>& actual, const std::vector<float>& predicted)
{
	ES_Assert(actual.size()==predicted.size());
	Clear();
	GetConfusionMatrix(actual, predicted);
	CalcMeasures();
}


void ClMeasures::CalcMeasures()
{
	accuracy = 0;
	precision = 0;
	fScore = 0;

	unsigned classCount = confusion.rows;
	unsigned sampleCount = 0;
	for(int i = 0;i<classCount;i++)
	{
		for(int j = 0;j<classCount;j++)
		{
			sampleCount += confusion.at<unsigned short>(i, j);
		}
	}
	for(int i = 0;i<classCount;i++)
	{
		//число объектов i-ого класса, которые классификатор отнес к i-ому классу
		unsigned truePredictedCount = confusion.at<unsigned short>(i, i);

		//число объектов, которые классификатор отнес к i-ому классу
		unsigned predictedCount = 0;

		//число объектов i-ого класса
		unsigned actualCount = 0;

		for(int k = 0;k<classCount;k++)
		{
			predictedCount += confusion.at<unsigned short>(k, i);
			actualCount += confusion.at<unsigned short>(i, k);
		}
		float classRation = float(actualCount)/sampleCount;
		if(predictedCount+actualCount!=0)
		{
			fScore += float(2*truePredictedCount)/(predictedCount+actualCount)*classRation;
			if(predictedCount!=0)
			{
				precision += float(truePredictedCount)/predictedCount*classRation;
			}
		}
		accuracy += truePredictedCount;
	}
	accuracy /= sampleCount;
}



void ClMeasures::Clear() 
{
	confusion.release();
	accuracy = -1;
	precision = -1;
	fScore = -1;
}


void ClMeasures::GetConfusionMatrix(const std::vector<float>& actual, 
									const std::vector<float>& predicted)
{
	map<float, int> classes;
	AddClasses(actual, classes);
	AddClasses(predicted, classes);

	confusion.create(classes.size(), classes.size(), CV_16UC1);
	confusion = 0;
	for(int i = 0;i<actual.size();i++)
	{
		map<float, int>::iterator iter;
		iter = classes.find(actual[i]);		int actualClass = iter->second;
		iter = classes.find(predicted[i]);	int predictedClass = iter->second;
		confusion.at<unsigned short>(actualClass, predictedClass) += 1;
	}
}


void AddClasses(const std::vector<float>& v, map<float, int>& classes)
{
	for(int i = 0;i<v.size();i++)
	{
		const float label = v[i];
		map<float, int>::iterator iter = classes.find(label);
		if(iter==classes.end())
		{
			pair<float,int> p(label,-1);
			classes.insert(p);
		}
	}
	int i = 0;
	for(map<float, int>::iterator iter = classes.begin();iter!=classes.end();iter++)
	{
		iter->second = i;
		i++;
	}
}


void ClMeasures::PrintObject(std::ostream& stream) const
{
	Print("дол€ правильных ответов", accuracy, stream);
	Print("точность", precision, stream);
	Print("f-мера", fScore, stream);
}

float ClMeasures::Error() const
{
	return 1-accuracy;
}

//====================================================
ReMeasures::ReMeasures() 
{
	Clear();
}

ReMeasures::ReMeasures(const std::vector<float>& actual, const std::vector<float>& predicted)
{
	Calculate(actual, predicted);
}

void ReMeasures::Calculate(const std::vector<float>& actual, const std::vector<float>& predicted)
{
	if(actual.size()!=predicted.size())
	{
		ES_Error("");
	}
	int n = actual.size();
	meanSqErr = 0;
	meanAbsErr = 0;
	for(int i = 0;i<n;i++)
	{
		float deviance = predicted[i]-actual[i];
		meanSqErr += pow(deviance, 2);
		meanAbsErr += fabs(deviance);
	}
	meanSqErr /= n;
	meanAbsErr /= n;
}

void ReMeasures::Clear()
{
	meanSqErr = -1;
	meanAbsErr = -1;
}

void ReMeasures::PrintObject(std::ostream& stream) const
{
	Print("средн€€ квадратична€ ошибка", meanSqErr, stream);
	Print("средн€€ абсолютна€ ошибка", meanAbsErr, stream);
}

float ReMeasures::Error() const
{
	return meanSqErr;
}
//====================================================

Time::Time()
{
	mSeconds = -1;
	seconds = -1;
	minutes = -1;
	hours = -1;
	ms = -1;
}

void Time::Start() 
{
	ms = GetTickCount();
}

void Time::Finish() 
{
	ms = GetTickCount()-ms;
	mSeconds = ms%1000;
	ms /= 1000;
	seconds = ms%60;
	ms /= 60;
	minutes = ms%60;
	ms = ms/60;
	hours = ms;
	ms = -1;
}

std::ostream& operator << (std::ostream& stream, const Time& time)
{
	stream << time.hours << " ч " << time.minutes << " мин " << time.seconds << " сек ";
	stream << time.mSeconds << " мс";
	return stream;
}





