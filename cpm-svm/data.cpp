#include "data.h"

#include <fstream>
#include <vector>
#include <sstream>


//================================
//#include <iostream>
//================================


using namespace std;



bool GetStringVector(string filename, vector<string>& vec);



Data::Data()
{
    isLoaded = false;
    trainCount = -1;
}


Data::~Data()
{
    Clear();
}


void Data::Clear()
{
    isLoaded = false;
    samples.clear();
    responses.clear();
    sampleIdx.clear();

    trainCount = -1;
}


bool Data::ReadFile(std::string filename)
{
    Clear();

    vector<string> vec;
    bool retVal = GetStringVector(filename, vec);
    if(retVal == false)
    {
        return false;
    }

    int varCount = 0;
    for(int k = 0;k<vec.size();k++)
    {
        string line = vec[k];

        int colonPos = line.rfind(':');

        if(colonPos!=-1)
        {
            int length = 0;
            while(line[colonPos-length]!=' ')
            {
                length++;
            }
            string str = line.substr(colonPos-length, length);
            int i = atoi(str.c_str());
            if(i>varCount)
            {
                varCount = i;
            }
        }
    }


    int dim = varCount;
    int n = vec.size();
    responses = Vec(n);
    samples = ZeroMat(n, dim);

    for(int i = 0;i<n;i++)
    {
        stringstream ss;
        ss << vec[i];

        ss >> responses[i];

        string str;
        while(ss >> str)
        {
            stringstream ss2;
            ss2 << str;

            int idx = -1;
            ss2 >> idx;

            char ch;
            ss2 >> ch;

            Real value;
            ss2 >> value;
            samples(i, idx-1) = value;
        }
    }

    sampleIdx.resize(n);
    for(int i = 0;i<n;i++)
    {
        sampleIdx[i] = i;
    }
    trainCount = n;

    isLoaded = true;
    return true;
}


void Data::SetTrainTestSplit(float trainPortion)
{
    if(!isLoaded)
    {
        throw Exception();
    }
    trainCount = floor(trainPortion*samples.size1());
}


//const Mat& Data::Samples() const
//{
//    return samples;
//}

//const Vec& Data::Responses() const
//{
//    return responses;
//}

//std::vector<int> Data::TrainSampleIdx() const
//{
//    vector<int> trainSampleIdx;
//    for(int i = 0;i<trainCount;i++)
//    {
//        trainSampleIdx.push_back(sampleIdx[i]);
//    }
//    return trainSampleIdx;
//}

//std::vector<int> Data::TestSampleIdx() const
//{
//    vector<int> testSampleIdx;
//    for(int i = trainCount;i<sampleIdx.size();i++)
//    {
//        testSampleIdx.push_back(sampleIdx[i]);
//    }
//    return testSampleIdx;
//}

bool Data::IsLoaded() const
{
    return isLoaded;
}

// Mix перемешивает выборку, используя rand для генерации случайных чисел
// srand не вызывается
void Data::Mix()
{
    int n = sampleIdx.size();
    for(int i = 0;i<n-1;i++)
    {
        int j = (rand() % (n-i)) + i;
        swap(sampleIdx[i], sampleIdx[j]);
    }
}



//---------------------------------------------------------------

bool GetStringVector(string filename, vector<string>& vec)
{
    ifstream f;
    f.open(filename.c_str());
    if(!f.is_open())
    {
        return false;
    }

    while(!f.eof())
    {
        string line;
        std::getline(f, line);
        vec.push_back(line);
    }
    vec.pop_back();

    f.close();
    return true;
}