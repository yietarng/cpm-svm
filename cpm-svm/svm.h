#ifndef SVM_H
#define SVM_H


#include "linear_algebra.h"
#include "data.h"


#define BMRM_INFO


class SVM
{
public:
    enum {TRAIN, TEST};
    class Exception {};
    SVM();
    void Train(const Data& data, const Real lambda,  const Real epsilon_abs,
               const Real epsilon_tol, const int tMax);
    Real Predict(Vec sample) const;
    Real CalcError(const Data& data, int type) const;


private:

    Vec betta;
};

#endif // SVM_H
