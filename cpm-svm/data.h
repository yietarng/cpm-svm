#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>

#include "linear_algebra.h"



class Data
{
public:
    class Exception {};

    Data();
    ~Data();

    bool ReadFile(std::string filename);
    void SetTrainTestSplit(float trainPortion);
    bool IsLoaded() const;
    void Mix();

    const Mat& Samples() const;
    const Vec& Responses() const;
    std::vector<int> TrainSampleIdx() const;
    std::vector<int> TestSampleIdx() const;


private:
    void Clear();

    bool isLoaded;
    std::vector<int> sampleIdx;
    int trainCount;
    Mat samples;
    Vec responses;
};

#endif // DATA_H
