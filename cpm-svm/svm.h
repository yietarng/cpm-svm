#ifndef SVM_H
#define SVM_H


#include "linear_algebra.h"
#include "data.h"




class SVM
{
public:
    class Exception {};
    SVM();
    void Train(const Data& data, const Real cValue, const Real epsilon, const int tMax);
    Real Predict(Vec sample) const;


private:

    Vec betta;
};

#endif // SVM_H
