#include "bmrm.h"

#include <math.h>
#include <float.h>
#include "libqp.h"
#include <limits.h>
#include <Windows.h>


#define SHOW_CALCULATIONS


//============================
#ifdef SHOW_CALCULATIONS
#include <iostream>
#endif SHOW_CALCULATIONS
//============================

using namespace std;


float DotProduct(const float* x, const float* y, int n);
float Max(float a, float b);
void SetNull(float* x, int n);
void Multiply(const float* x, float alpha, float* result, int n);

const double* GramCol(uint32_t j);

#ifdef SHOW_CALCULATIONS
void Print(const float* x, int n); 
#endif SHOW_CALCULATIONS



double** gram;



BMRMSolver::BMRMSolver(const float** samples, const float* responses, 
					   int dimention, int sampleCount, float _lambda)
{
	x = samples;
	y = responses;
	n = dimention;
	m = sampleCount;
	lambda = _lambda;
}

void BMRMSolver::Solve(float epsilon, int maxIter, float *_betta) 
{
	vector<float*> a;
	vector<float> b;
	float* w = _betta;
	SetNull(w, n);

	//======================================
	#ifdef SHOW_CALCULATIONS
	cout << "w[0] = ";
	Print(w, n);
	cout << endl;
	#endif SHOW_CALCULATIONS
	//======================================

	int t = 0;
	while(t<maxIter)
	{
		t++;

		float* subnt = new float[n];
		CalcEmpRiskSubnt(w, subnt);
		a.push_back(subnt);
		b.push_back( EmpRisk(w)-DotProduct(w, subnt, n) );

		//==============================================
		//Пересчет w...
		gram = new double*[t];
		for(int i = 0;i<t;i++)
		{
			gram[i] = new double[t];
		}
		for(int i = 0;i<t;i++)
		{
			for(int j = 0;j<t;j++)
			{
				gram[i][j] = DotProduct(a[i], a[j], n);
			}
		}

		double* f = new double[t];
		for(int i = 0;i<t;i++)
		{
			f[i] = b[i];
		}
		
		double bArr[1] = {1};
		uint8_t S[1] = {0};

		uint32_t* I = new uint32_t[t];
		for(int i = 0;i<t;i++)
		{
			I[i] = 1;
		}

		double* diag = new double[t];
		for(int i = 0;i<t;i++)
		{
			diag[i] = gram[i][i];
		}

		double* alpha = new double[t];
		for(int i = 0;i<t;i++)
		{
			alpha[i] = 1./t;
		}

		int start = GetTickCount();
		libqp_state_T state = libqp_splx_solver(GramCol, diag, f, bArr, I, S, alpha, t, INT_MAX, epsilon*10, epsilon/10, DBL_MIN, 0); 
		int finish = GetTickCount();
		//===============================================
		#ifdef SHOW_CALCULATIONS
		cout << "время решения задачи кв. прогр. " << float(finish-start)/1000 << "секунд" << endl;
		cout << "сделано " << state.nIter << "итераций" << endl;
		cout << "eps для задачи квадр. программирования " << state.QP-state.QD << endl;
		#endif SHOW_CALCULATIONS
		//===============================================
		for(int i = 0;i<n;i++)
		{
			float sum = 0;
			for(int j = 0;j<t;j++)
			{
				sum += a[j][i]*float(alpha[j]);
			}
			w[i] = (-1/lambda)*sum;
		}

		for(int i = 0;i<t;i++)
		{
			delete[] gram[i];
		}
		delete[] gram;
		delete[] diag;
		delete[] I;
		delete[] f;
		delete[] alpha;
		//==============================================


		float aa = J(w);
		float bb = Jcp(w, a, b);
		float gap = J(w)-Jcp(w, a, b);

		//===============================================
		#ifdef SHOW_CALCULATIONS

		//вывод результатов
		cout << "субградиент a[" << t << "] = ";
		Print(a.back(), n);

		cout << "b[" << t << "] = " << b.back() << endl;

		cout << "w[" << t << "] = ";
		Print(w, n);

		cout << "eps[" << t << "] = " << gap << endl;
		cout << endl << endl;

		#endif SHOW_CALCULATIONS
		//===============================================

		if(gap<=epsilon)
		{
			#ifdef SHOW_CALCULATIONS
			cout << "выход" << endl;
			#endif SHOW_CALCULATIONS

			break;
		}
	}

	for(int i = 0;i<a.size();i++)
	{
		delete[] a[i];
	}
}


float BMRMSolver::Jcp(const float* w, const std::vector<float*>& a, const std::vector<float>& b) const
{
	float max = -FLT_MAX;
	for(int k = 0;k<a.size();k++)
	{
		float value = b[k]+DotProduct(w, a[k], n);
		if(value>max)
		{
			max = value;
		}
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



#ifdef SHOW_CALCULATIONS
void Print(const float* x, int n)
{
	cout << "[";
	for(int i = 0;i<n;i++)
	{
		cout << x[i];
		if(i!=n-1)
		{
			cout << ", ";
		}
	}
	cout << "]" << endl;
}
#endif SHOW_CALCULATIONS

const double* GramCol(uint32_t j)
{
	return gram[j];
}

