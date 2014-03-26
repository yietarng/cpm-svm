#include "svm.h"
#include "data.h"

#include <iostream>


using namespace std;


int main()
{
//    SVM svm;
//    Mat X;
//    Vec Y;
//    X.resize(5, 2);
//    Y.resize(5);

//    svm.Train(X, Y);

//    Vec x(2);
//    x[0] = 3;
//    x[1] = -10;
//    cout << svm.Predict(x) << endl;

    Data data;
    data.ReadFile("/home/sergei/libsvm_data/test.data");
    cout << data.IsLoaded() << endl;
    //data.SetTrainTestSplit(0.8);
    data.Mix();
    data.Mix();

    return 0;
}

