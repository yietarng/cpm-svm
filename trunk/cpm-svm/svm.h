#ifndef SVM_H
#define SVM_H


#include "linear_algebra.h"


class SVM
{
public:
    class Exception {};
    SVM();
    void Train(Mat samples, Vec responses);
    Real Predict(Vec sample) const;


private:

    Vec betta;
};

#endif // SVM_H
