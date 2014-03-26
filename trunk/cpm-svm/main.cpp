#include "svm.h"
#include "data.h"

#include <iostream>


using namespace std;


int main()
{
    SVM svm;

    Data data;
    data.ReadFile("/home/sergei/libsvm_data/test2.data");
    cout << "Data loaded: " << data.IsLoaded() << endl;
    //data.SetTrainTestSplit(0.8);
//    data.Mix();

    svm.Train(data);

    Vec x(2);
    x[0] = 3;
    x[1] = -10;
//    cout << "Prediction = " << svm.Predict(x) << endl;



    return 0;
}

