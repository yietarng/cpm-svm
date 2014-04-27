#include "svm.h"
#include "data.h"

#include <iostream>


using namespace std;



int main(int argc, char* argv[])
{
    if(argc!=5)
    {
        cout << "Usage: lsvm <data_filename> <C_value> <epsilon> <max_iter>" << endl;
        return 1;
    }


    // "/home/sergei/libsvm_data/test2.data"

    Data data;
    data.ReadFile(argv[1]);
    if(!data.IsLoaded())
    {
        cout << "Error while loading data" << endl;
        return 1;
    }

    Real cValue = Real(atof(argv[2]));
    Real epsilon = Real(atof(argv[3]));
    int maxIter = atoi(argv[4]);

    //data.SetTrainTestSplit(0.8);
//    data.Mix();

    cout << "Data file: " << argv[1] << endl;
    cout << "Cvalue = " << cValue << endl;
    cout << "Epsilon = " << epsilon << endl;
    cout << "Max number of iterations = " << maxIter << endl;
    cout << "Starting svm..." << endl;
    SVM svm;
    svm.Train(data, 1, 0.01, 100);

    return 0;
}

