#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H


#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>


typedef double Real;
typedef boost::numeric::ublas::matrix<Real> Mat;
typedef boost::numeric::ublas::vector<Real> Vec;

typedef boost::numeric::ublas::zero_matrix<Real> ZeroMat;


#endif // LINEAR_ALGEBRA_H
