#ifndef SOLVE_QP_H
#define SOLVE_QP_H

#include "linear_algebra.h"


void SolveQP(std::vector<Vec>& a, std::vector<Real>& b, Real lambda, Real epsilon_tol,
             Vec& alpha);


#endif // SOLVE_QP_H
