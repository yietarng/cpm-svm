#include "svm.h"
#include "data.h"

#include <iostream>


using namespace std;




int main(int argc, char* argv[])
{
    if(argc!=7)
    {
        cout << "Usage: lsvm <data_filename> <train_portion> <mix_seed> <C_value> <epsilon> <max_iter>" << endl;
        return 1;
    }
    const char* dataPathStr = argv[1];
    const char* trainPortionStr = argv[2];
    const char* seedStr = argv[3];
    const char* cValueStr = argv[4];
    const char* epsilonStr = argv[5];
    const char* maxIterStr = argv[6];



    cout << "Data file: " << dataPathStr << endl;
    Data data;
    data.ReadFile(dataPathStr);
    if(!data.IsLoaded())
    {
        cout << "Error while loading data" << endl;
        return 1;
    }
    cout << "Number of variables: " << data.Samples().size2() << endl;


    float trainPortion = atof(trainPortionStr);
    data.SetTrainTestSplit(trainPortion);
    cout << trainPortion*100 << "% of data is used for training" << endl;
    cout << "Train sample count: " << data.TrainSampleIdx().size() << endl;
    cout << "Test sample count: " << data.TestSampleIdx().size() << endl;


    int seed = atoi(seedStr);
    if(seed>=0)
    {
        srand(unsigned(seed));
        data.Mix();
        cout << "Data mixed with seed=" << seed << endl;
    }
    else
    {
        cout << "Data not mixed" << endl;
    }



    Real cValue = Real(atof(cValueStr));
    Real epsilon = Real(atof(epsilonStr));
    int maxIter = atoi(maxIterStr);

    cout << endl;
    cout << "Cvalue = " << cValue << endl;
    cout << "Epsilon = " << epsilon << endl;
    cout << "Max number of iterations = " << maxIter << endl;


    cout << "Svm training..." << endl;
    SVM svm;
    svm.Train(data, cValue, epsilon, maxIter);
    cout << "Svm training completed." << endl;

    cout << "Train error: " << svm.CalcError(data, SVM::TRAIN) << endl;
    cout << "Test error: " << svm.CalcError(data, SVM::TEST) << endl;

    return 0;
}

