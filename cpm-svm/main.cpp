#include "svm.h"
#include "data.h"

#include <iostream>


using namespace std;



#define BMRM_INFO


////==============================
//#include <CGAL/basic.h>
//#include <CGAL/MP_Float.h>
//#include <CGAL/Gmpz.h>
//#include <CGAL/to_rational.h>
//typedef CGAL::Gmpz ET;
//typedef CGAL::Quotient<int> Rational;
////==============================


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

    data.SetTrainTestSplit(0.6);
    cout << "Train sample count: " << data.TrainSampleIdx().size() << endl;
    cout << "Test sample count: " << data.TestSampleIdx().size() << endl;
    cout << "Number of variables: " << data.Samples().size2() << endl;
//    cout << "Train sample indexes: ";
//    for(int i = 0;i<data.TrainSampleIdx().size();i++)
//    {
//        cout << data.TrainSampleIdx()[i] << ' ';
//    }
//    cout << endl;

    data.Mix();
    cout << "Data mixed" << endl;

    cout << "Data file: " << argv[1] << endl;
    cout << "Cvalue = " << cValue << endl;
    cout << "Epsilon = " << epsilon << endl;
    cout << "Max number of iterations = " << maxIter << endl;
    cout << "Starting svm..." << endl;
    SVM svm;
    svm.Train(data, cValue, epsilon, maxIter);
    cout << "Train error: " << svm.CalcError(data, SVM::TRAIN) << endl;
    cout << "Test error: " << svm.CalcError(data, SVM::TEST) << endl;

//    CGAL::MP_Float fl(334.125);
//    ET et(fl.);
//    cout << fl << endl;
//    cout << et << endl;
    //cout << CGAL::to_rational<Rational>(334.125) << endl;


    return 0;
}

