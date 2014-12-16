#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <list>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/rational.hpp>


typedef double Real;
typedef boost::numeric::ublas::matrix<Real> Mat;
typedef boost::numeric::ublas::vector<Real> Vec;

typedef boost::numeric::ublas::zero_matrix<Real> ZeroMat;
typedef boost::numeric::ublas::zero_vector<Real> ZeroVector;


struct Pair
{
    Pair(Real _value, int _idx) : value(_value), idx(_idx) {}

    Real value;
    int idx;
};

typedef std::vector< std::list<Pair> > SparseMat;




#include <sys/time.h>

inline long long gettimeus()
{
    struct timeval tv;

    gettimeofday( &tv, NULL );
    return (long long) tv.tv_sec * 1000000LL + (long long) tv.tv_usec;
}


#endif // LINEAR_ALGEBRA_H
