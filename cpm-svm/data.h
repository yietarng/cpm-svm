#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>
#include <list>

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

    const SparseMat& Samples() const;
    const Vec& Responses() const;
    std::vector<int> TrainSampleIdx() const;
    std::vector<int> TestSampleIdx() const;
    unsigned VarNumber() const;


private:
    void Clear();

    bool isLoaded;
    std::vector<int> sampleIdx;
    int trainCount;
    SparseMat samples;
    Vec responses;
    unsigned varNumber;
};


std::ostream& operator << (std::ostream& stream, const Data& data);

#endif // DATA_H
