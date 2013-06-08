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


double DotProduct(const double* x, const double* y, int n);
double Max(double a, double b);
void SetNull(double* x, int n);
void Multiply(const double* x, double alpha, double* result, int n);

const double* GramCol(uint32_t j);


#ifdef SHOW_CALCULATIONS
template<class T>
void Print(const T* x, int n); 
#endif SHOW_CALCULATIONS



double** gram;



BMRMSolver::BMRMSolver(const double** samples, const double* responses, 
					   int dimention, int sampleCount, double _lambda)
{
	x = samples;
	y = responses;
	n = dimention;
	m = sampleCount;
	lambda = _lambda;
}

void BMRMSolver::Solve(double epsilon, int maxIter, double *_betta) 
{
	vector<double*> a;
	vector<double> b;
	double* w = _betta;
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
		#ifdef SHOW_CALCULATIONS
		int iterTime = -GetTickCount();
		#endif SHOW_CALCULATIONS

		t++;

		double* subnt = new double[n];
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
			f[i] = -b[i];
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
		libqp_state_T state = libqp_splx_solver(GramCol, diag, f, bArr, I, S, alpha, t, INT_MAX, epsilon/2, 0, -DBL_MAX, 0); 
		int finish = GetTickCount();
		//===============================================
		#ifdef SHOW_CALCULATIONS
		cout << "alpha[" << t << "] = ";
		Print(alpha, t);
		cout << "время решения задачи кв. прогр. " << double(finish-start)/1000 << "секунд" << endl;
		cout << "сделано " << state.nIter << " итераций" << endl;
		cout << "eps для задачи квадр. программирования " << state.QP-state.QD << endl;
		#endif SHOW_CALCULATIONS
		//===============================================
		for(int i = 0;i<n;i++)
		{
			double sum = 0;
			for(int j = 0;j<t;j++)
			{
				sum += a[j][i]*double(alpha[j]);
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

		double aa = J(w);
		double bb = Jcp(w, a, b);
		double gap = J(w)-Jcp(w, a, b);

		//===============================================
		#ifdef SHOW_CALCULATIONS

		//вывод результатов
		cout << "субградиент a[" << t << "] = ";
		Print(a.back(), n);

		cout << "b[" << t << "] = " << b.back() << endl;

		cout << "w[" << t << "] = ";
		Print(w, n);

		cout << "eps[" << t << "] = " << gap << endl;

		iterTime += GetTickCount();
		cout << "время выполнения итерации: " << float(iterTime)/1000 << " секунд" << endl; 
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


double BMRMSolver::Jcp(const double* w, const std::vector<double*>& a, const std::vector<double>& b) const
{
	double max = -FLT_MAX;
	for(int k = 0;k<a.size();k++)
	{
		double value = b[k]+DotProduct(w, a[k], n);
		if(value>max)
		{
			max = value;
		}
	}
	return lambda*Regularizer(w)+max;
}


double BMRMSolver::Regularizer(const double* w) const
{
	return DotProduct(w, w, n)/2;
}

double BMRMSolver::EmpRisk(const double* w) const
{
	double sum = 0;
	for(int i = 0;i<m;i++)
	{
		sum += Max( 0, 1-y[i]*DotProduct(x[i], w, n) );
	}
	return sum;
}

double BMRMSolver::J(double* w) const
{
	return lambda*Regularizer(w)+EmpRisk(w);
}

void BMRMSolver::CalcEmpRiskSubnt(const double* w, double* subnt) const
{
	SetNull(subnt, n);
	for(int i = 0;i<m;i++)
	{
		const double* sample = x[i];
		double value = Max( 0, 1-y[i]*DotProduct(sample, w, n) );
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

double DotProduct(const double* x, const double* y, int n)
{
	double res = 0;
	for(int i = 0;i<n;i++)
	{
		res += x[i]*y[i];
	}
	return res;
}

double Max(double a, double b)
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

void SetNull(double* x, int n)
{
	for(int i = 0;i<n;i++)
	{
		x[i] = 0;
	}
}

void Multiply(const double* x, double alpha, double* result, int n)
{
	for(int i = 0;i<n;i++)
	{
		result[i] = alpha*x[i];
	}
}



#ifdef SHOW_CALCULATIONS
template<class T>
void Print(const T* x, int n)
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

