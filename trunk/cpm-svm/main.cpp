#include "svm.h"
#include "data.h"

#include <iostream>


//
#include "solve_qp.h"


using namespace std;


int Main(int argc, char* argv[]);


int main(int argc, char* argv[])
{
    return Main(argc, argv);
}



int Main(int argc, char* argv[])
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



    float trainPortion = atof(trainPortionStr);
    data.SetTrainTestSplit(trainPortion);



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

    long long time_svm = -gettimeus();
    svm.Train(data, cValue, epsilon, maxIter);
    time_svm += gettimeus();

    cout << endl << "Svm training completed." << endl;
    cout << "svm training time: " << double(time_svm)/1000000 << " seconds" << endl;

    cout << "Number of variables: " << data.VarNumber() << endl;
    cout << "Number of samples: " <<
            data.TestSampleIdx().size() + data.TrainSampleIdx().size() << endl;
    cout << trainPortion*100 << "% of data is used for training" << endl;
    cout << "Train sample count: " << data.TrainSampleIdx().size() << endl;
    cout << "Test sample count: " << data.TestSampleIdx().size() << endl;

    cout << "Train accuracy: " << (1.0-svm.CalcError(data, SVM::TRAIN))*100 << "%" << endl;
    cout << "Test  accuracy: " << (1.0-svm.CalcError(data, SVM::TEST))*100 << "%" << endl;

    return 0;
}

