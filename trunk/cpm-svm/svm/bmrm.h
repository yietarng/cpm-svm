#ifndef BMRM_MAC
#define BMRM_MAC


#include <vector>
#include <stdint.h>

//EmpRisk

class BMRMSolver
{
public:

	BMRMSolver(const float** samples, const float* responses, int dimention, int sampleCount, 
		float _lambda);
	void Solve(float epsilon, int maxIter, float* _betta);

private:

	float EmpRisk(const float* w) const;
	float Jcp(const float* w, const std::vector<float*>& a, const std::vector<float>& b) const;
	float Regularizer(const float* w) const;
	void CalcEmpRiskSubnt(const float* w, float* subnt) const;
	float J(float* w) const;

	const float** x;
	const float* y;
	int m, n;
	float lambda;
};




#endif BMRM_MAC