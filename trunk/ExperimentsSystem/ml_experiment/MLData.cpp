#include "MLData.h"
#include <list>
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include "ErrorHandling.h"

#include "ToString.h"
#include "Scheme.h"


using namespace cv;
using namespace std;
using namespace mle;

Scheme scheme;

bool ReadScv(CvMLData* data, const string& header_filename, const string& csv_filename);
int FindAttribute(const FileNode& attributes, const FileNodeIterator& end, const string& name);


void mle::MLData::LoadPatterns(const std::string& patternsPath)
{
	if(!scheme.Load(patternsPath+"\\ml_data_header.yml"))
	{
		ES_Error("не удалось загрузить шаблон");
	}
}


bool mle::MLData::SaveHeader(const std::string& header_filename, const std::string& csv_filename)
{
	FileStorage storage(header_filename, FileStorage::WRITE);
	if(!storage.isOpened())
	{
		return false;
	}
	
	storage << "ml_data_set_header" << "{";

	storage << "attributes" << "[";
	string varType[2] = {"ordered", "categorical"};
	for(int i = 0;i<this->get_values()->cols;i++)
	{
		storage << "{";
		storage << "name" << "X"+ToString(i+1);
		storage << "type" << varType[this->get_var_type(i)];
		storage << "}";
	}
	storage << "]";

	storage << "responses" << "X"+ToString(this->get_response_idx()+1);
	storage << "number_of_samples" << this->get_values()->rows;
	storage << "test_samples_percentage" << 25;

	string missingStr = "No";
	if(this->get_missing())
	{
		missingStr = "Yes";
	}
	storage << "missed_values" << missingStr;
	
	storage << "csv_file" << "{";
	storage << "filename" << csv_filename;
	string delimiter;
	char ch = this->get_delimiter();
	switch(ch)
	{
		case ' '	: delimiter = "space";	break;
		case '\t'	: delimiter = "tab";	break;
		default		: delimiter = ToString(ch);
	}
	storage << "delimiter" << delimiter;
	string miss_ch; 
	if(missing)
	{
		miss_ch = ToString(this->get_miss_ch());
	}
	storage << "miss_character" << miss_ch;
	storage << "}";

	storage << "}";

	storage.release();
	return true;
}



void mle::MLData::Load(const std::string& header_filename)
{
	FileStorage storage;
	if(!storage.open(header_filename, FileStorage::READ))
	{
		ES_Error("ошибка открытия yml-файла");
	}
	scheme.Check(&storage);
	FileNode topNode = storage.getFirstTopLevelNode();

	string s = topNode["missed_values"];
	bool missed_values;
	if(s=="Yes")
	{
		missed_values = true;
	}
	else
	{
		//s=="No"
		missed_values = false;
	}
	
	FileNode csv_file = topNode["csv_file"];
	string delimiter = csv_file["delimiter"];
	char ch = 0;
	if(delimiter.length()==1)
	{
		ch = delimiter[0];
	}
	if(delimiter=="space")
	{
		ch = ' ';
	}	
	if(delimiter=="tab")
	{
		ch = '\t';
	}
	if(ch==0)
	{
		ES_Error("Некорректное значение для поля delimiter");
	}
	this->set_delimiter(ch);	
	
	
	string miss_ch = csv_file["miss_character"];
	if(miss_ch.length()==1)
	{
		if(missed_values)
		{
			this->set_miss_ch(miss_ch[0]);
		}
		else
		{
			ES_Error("поле miss_ch в yml-файле должно быть пустым");
		}
	}
	
	string csv_filename = csv_file["filename"];

	if(!ReadScv(this, header_filename, csv_filename))
	{
		ES_Error("ошибка при загрузке scv-файла");
	}

	if(int(topNode["number_of_samples"])!=this->get_values()->rows)
	{
		ES_Error("не совпадает число прецедентов в csv- и yml- файлах");
	}

	float testPercentage = int(topNode["test_samples_percentage"])/float(100);
	CvTrainTestSplit spl(1-testPercentage, false);
	set_train_test_split(&spl);

	FileNode attributes = topNode["attributes"];
	if(attributes.size()!=this->get_values()->cols)
	{
		ES_Error("не совпадает число атрибутов в csv- и yml- файлах");
	}
	FileNodeIterator iter; int count = 0;
	for(iter = attributes.begin();iter!=attributes.end();iter++)
	{
		if(FindAttribute(attributes, iter, (*iter)["name"])>=0)
		{
			ES_Error("два атрибута с одинаковыми названиями");
		}
		if((string)(*iter)["type"]=="ordered")
		{
			int var_type = this->get_var_type(count);
			if(var_type==CV_VAR_CATEGORICAL)
			{
				ES_Error("типы переменной номер "+ToString(count+1)+"(нумерация с единицы) в yml-файле"+ 
					" и data-файле не совпадают");
			}
		}
		else
		{
			this->change_var_type(count, CV_VAR_CATEGORICAL);
		}
		count++;
	}

	int responseIdx = FindAttribute(attributes, attributes.end(), topNode["responses"]);
	if(responseIdx>=0)
	{
		this->set_response_idx(responseIdx);
	}
	else
	{
		ES_Error("поле responses должно совпадать с названием одного из атрибутов");
	}
	storage.release();
}



bool ReadScv(CvMLData* data, const string& header_filename, const string& csv_filename)
{
	if(data->read_csv(csv_filename.c_str()))
	{
		string path = header_filename;
		char slash = 92;
		int j = path.find_last_of(slash);
		path = path.substr(0,j+1)+csv_filename;
		if(data->read_csv(path.c_str()))
		{
			int j = header_filename.find_last_of(".");
			string path = header_filename.substr(0,j+1)+"data";
			if(data->read_csv(path.c_str()))
			{
				path = header_filename.substr(0,j+1)+"csv";
				if(data->read_csv(path.c_str()))
				{
					return false;
				}
			}
		}
	}
	return true;
}

int FindAttribute(const FileNode& attributes, const FileNodeIterator& end, const string& name)
{
	FileNodeIterator iter;
	int count = 0;
	for(iter = attributes.begin();iter!=end;iter++)
	{
		string atrbName = (*iter)["name"];

		if(atrbName==name)
		{
			return count;
		}
		count++;
	}
	return -1;
}

void mle::MLData::CreateMLDataHeaders(const std::string& _directory, std::ostream& stream, 
									  const std::vector<std::string>& extentions)
{
	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind;

	const string directory = _directory+"\\";

	hFind = FindFirstFileA((directory+"*").c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) 
	{
		stream << _directory << " - папка не найдена" << endl;
		return;
	}
	else 
	{
		stream << _directory << endl << endl; 
		MLData* data = new MLData();
		while(FindNextFileA(hFind, &FindFileData))
		{
			string csv_name = FindFileData.cFileName;
			
			bool flag = extentions.size()==0;
			for(int k = 0;k<extentions.size();k++)
			{
				int i = csv_name.find(extentions[k]);
				if(i!=-1 && i+extentions[k].length() == csv_name.length())
				{
					flag = true;
					break;
				}
			}

			if(flag)
			{
				stream << csv_name;
				string message = "ok";
				try
				{
					string csv_filename = directory+csv_name;
					const int dCount = 4;
					char delimiter[] = {',', ';', ' ', '\t'};
					int s = 0;
					for(;s<dCount;s++)
					{
						data->set_delimiter(delimiter[s]);
						if(!data->read_csv(csv_filename.c_str()))
						{
							break;
						}
					}
					if(s==dCount)
					{
						throw 0;
					}
					int i = csv_name.find(".");
					string header_filename = directory+csv_name.substr(0, i)+".yml";
					data->set_response_idx(data->get_values()->cols-1);
					if(!data->SaveHeader(header_filename, csv_name))
					{
						throw 0;
					}
				}
				catch(...)
				{
					message = "failed";
				}
				stream.width(50-csv_name.length());
				stream << "\t" << message << endl;
			}
		}
		delete data;
		FindClose(hFind);
	}
}


void MLData::set_train_test_split(const cv::Mat& mask)
{
	const int sample_count = values->rows;
	if(mask.cols!=sample_count || mask.rows!=1 || mask.type()!=CV_8UC1)
	{
		ES_Error("");
	}
	int k = 0;int s = sample_count-1;
	for(int i = 0;i<sample_count;i++)
	{
		if(!mask.at<uchar>(i))
		{
			sample_idx[k] = i;
			k++;
		}
		else
		{
			sample_idx[s] = i;
			s--;
		}
	}
	train_sample_count = k;
	*train_sample_idx = cvMat(1, train_sample_count, CV_32SC1, sample_idx);
	*test_sample_idx  =	cvMat(1, sample_count-train_sample_count, CV_32SC1, 
						&sample_idx[train_sample_count]);
}

void MLData::set_train_test_split(const CvTrainTestSplit* spl)
{
	CvMLData::set_train_test_split(spl);
}

cv::Mat MLData::SubData(int type) const
{
	//!!!
	//Здесь предполагается, что используются все признаки, 
	//а ответы находятся лежат в последнем столбце
	int m = 0==get_missing();
	int rIdx = get_response_idx();

	CV_Assert(get_response_idx()==values->cols-1 && get_missing()==0);

	const CvMat* sampleIdx;
	if(type==CV_TRAIN_ERROR)
	{
		sampleIdx = this->get_train_sample_idx();
	}
	else //type==CV_TEST_ERROR
	{
		sampleIdx = this->get_test_sample_idx();
	}
	const int sampleCount = sampleIdx->cols;

	cv::Mat samples(sampleCount, values->cols-1, CV_32FC1);
	for(int i = 0;i<sampleCount;i++)
	{
		memcpy(samples.ptr<uchar>()+size_t(samples.step)*i, 
			values->data.ptr+values->step*(sampleIdx->data.i[i]),
				sizeof(float)*(values->cols-1));
	}
	return samples;
}



