#ifndef BMRM_MAC
#define BMRM_MAC


#include <vector>
//#include "stdint.h"

//EmpRisk

class BMRMSolver
{
public:

	BMRMSolver(const double** samples, const double* responses, int dimention, int sampleCount, 
		double _lambda);
	void Solve(double epsilon, int maxIter, double* _betta);

private:

	double EmpRisk(const double* w) const;
	double Jcp(const double* w, const std::vector<double*>& a, const std::vector<double>& b) const;
	double Regularizer(const double* w) const;
	void CalcEmpRiskSubnt(const double* w, double* subnt) const;
	double J(double* w) const;

	const double** x;
	const double* y;
	int m, n;
	double lambda;
};




#endif BMRM_MAC