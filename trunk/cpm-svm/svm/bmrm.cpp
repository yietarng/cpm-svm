#include "bmrm.h"

#include <math.h>
#include <list>
#include <float.h>

using namespace std;


float DotProduct(const float* x, const float* y, int n);
float Max(float a, float b);
void SetNull(float* x, int n);
void Multiply(const float* x, float alpha, float* result, int n);



BMRMSolver::BMRMSolver(const float** samples, const float* responses, 
					   int dimention, int sampleCount, float _lambda)
{
	x = samples;
	y = responses;
	n = dimention;
	m = sampleCount;
	lambda = _lambda;
}

void BMRMSolver::Solve(float epsilon, int maxIter, float *_betta) const
{
	list<float*> a;
	list<float> b;
	float* w = _betta;
	SetNull(w, n);

	int t = 0;
	while(t<maxIter)
	{
		t++;

		float* subnt = new float[n];
		CalcEmpRiskSubnt(w, subnt);
		a.push_back(subnt);
		b.push_back( EmpRisk(w)-DotProduct(w, subnt, n) );

		//Пересчет w...

		float gap = J(w)-Jcp(w, a, b);
		if(gap<=epsilon)
		{
			break;
		}
	}

	while(!a.empty())
	{
		delete[] a.front();
		a.pop_front();
	}
}


float BMRMSolver::Jcp(const float* w, const std::list<float*>& a, const std::list<float>& b) const
{
	float max = FLT_MIN;
	list<float*>::const_iterator aIter = a.begin();
	list<float>::const_iterator bIter = b.begin();
	while(aIter!=a.end())
	{
		float value = DotProduct(w, *aIter, n);
		if(value>max)
		{
			max = value;
		}
		aIter++;
		bIter++;
	}
	return lambda*Regularizer(w)+max;
}


float BMRMSolver::Regularizer(const float* w) const
{
	return DotProduct(w, w, n)/2;
}

float BMRMSolver::EmpRisk(const float* w) const
{
	float sum = 0;
	for(int i = 0;i<m;i++)
	{
		sum += Max( 0, 1-y[i]*DotProduct(x[i], w, n) );
	}
	return sum;
}

float BMRMSolver::J(float* w) const
{
	return lambda*Regularizer(w)+EmpRisk(w);
}

void BMRMSolver::CalcEmpRiskSubnt(const float* w, float* subnt) const
{
	SetNull(subnt, n);
	for(int i = 0;i<m;i++)
	{
		const float* sample = x[i];
		float value = Max( 0, 1-y[i]*DotProduct(sample, w, n) );
		if(value!=0)
		{
			for(int j = 0;j<n;j++)
			{
				subnt[j] += -y[i]*sample[j];
			}
		}
	}
}

//====================================================================================

float DotProduct(const float* x, const float* y, int n)
{
	float res = 0;
	for(int i = 0;i<n;i++)
	{
		res += x[i]*y[i];
	}
	return res;
}

float Max(float a, float b)
{
	if(a>b)
	{
		return a;
	}
	else
	{
		return b;
	}
}

void SetNull(float* x, int n)
{
	for(int i = 0;i<n;i++)
	{
		x[i] = 0;
	}
}

void Multiply(const float* x, float alpha, float* result, int n)
{
	for(int i = 0;i<n;i++)
	{
		result[i] = alpha*x[i];
	}
}

