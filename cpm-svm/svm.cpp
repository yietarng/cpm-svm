#include "svm.h"

#include <vector>
#include <assert.h>

using namespace std;
//using namespace boost::numeric::ublas;



//=============================================
// CGAL
#include <iostream>
#include <cassert>
#include <CGAL/basic.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
//// choose exact integral type
//#ifdef CGAL_USE_GMP
//#include <CGAL/Gmpz.h>
//typedef CGAL::Gmpz ET;
//#else
//#include <CGAL/MP_Float.h>
//typedef CGAL::MP_Float ET;
//#endif

typedef Real ET;
// program and solution types
typedef CGAL::Quadratic_program<Real> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;
//=============================================





Real empRisk(const Data& data, const Vec& w);
Real empRiskCP(const vector<Vec>& a, const vector<Real>& b, const Vec& w);
Vec empRiskSubgradient(const Data& data, const Vec& w);
Real Omega(const Vec& w);


Real Product(int row, const Mat& mat, const Vec& vec);


//--------------------------------------------------------------------------------

SVM::SVM()
{
}


void SVM::Train(const Data& data, const Real cValue, const Real epsilon, const int tMax)
{
    // Очистка модели
    betta.clear();


    Mat samples = data.Samples();
    Vec responses = data.Responses();
    for(int i = 0;i<responses.size();i++)
    {
        assert(responses[i]==-1 || responses[i]==1);
    }
//    vector<int> testSampleIdx = data.TestSampleIdx();
//    vector<int> trainSampleIdx = data.TrainSampleIdx();

    int n = samples.size1(); // число прецедентов в обучающей выборке
    int d = samples.size2(); // размерность пространства признаков

    // проверка входных данных
    if(n<=0 || d<=0 || n!=responses.size())
    {
        throw SVM::Exception();
    }


    // Обучение
    //-----------------------------------
    Vec w(d);
    std::fill(w.begin(), w.end(), 0);
    int t = 0;
    int currentEps = 1;
    vector<Vec> a;
    vector<Real> b;

    cout << "Training..." << endl;
    do
    {
//        if(t%10==0)
//        {
//            cout << ".";
//        }


        t++;
        a.push_back( empRiskSubgradient(data, w) );
        b.push_back( empRisk(data, w) - inner_prod(w, a.back()) );


        cout << endl << "Iteration " << t << endl;
        cout << "empRisk(w) = " << empRisk(data, w) << endl;
        cout << "w[" << t-1 << "] = " << w << endl;
        cout << "a[" << t << "] = " << a.back() << endl;
        cout << "b[" << t << "] = " << b.back() << endl;


        // argmin begin
        Program qp(CGAL::EQUAL, true, 0, false, 0);

        for(int i = 0;i<t;i++)
        {
            qp.set_a(i, 0, 1);
        }

        qp.set_b(0, 1);

        for(int i = 0;i<t;i++)
        {
            for(int j = 0;j<=i;j++)
            {
                qp.set_d(i, j, inner_prod(a[i], a[j]));
            }
        }

        for(int i = 0;i<t;i++)
        {
            qp.set_c(i, -cValue*b[i]);
        }


        // solve the program, using ET as the exact type
        Solution s = CGAL::solve_quadratic_program(qp, ET());
        assert (s.solves_quadratic_program(qp));

//        for (Solution::Index_iterator it = s.basic_variable_indices_begin();
//                it != s.basic_variable_indices_end(); ++it)
//        {
//            std::cout << *it << " ";
//        }
//        cout << endl;

        Vec alpha(t);
        Solution::Variable_numerator_iterator it = s.variable_numerators_begin();
        int k = 0;
        for(;it!=s.variable_numerators_end();it++)
        {
            alpha[k] = *it/s.variables_common_denominator();
            k++;
        }

        Vec temp(d);
        std::fill(temp.begin(), temp.end(), 0);
        for(int i = 0;i<t;i++)
        {
            temp = temp + alpha[i]*a[i];
        }
        temp = -temp/cValue;
        w = temp;
        cout << "w[" << t << "] = " << w << endl;




        //Solution::Variable_value_iterator iter;
        //s.variable_values_begin();
        //int i = 1;
//        for(iter = s.variable_values_begin();iter!=s.variable_values_end();iter++)
//        {
////            cout << "alpha[" << i << "] = " << *iter << endl;
//            i++;
//        }


        // output solution
        //std::cout << endl << s;

        // argmin end


        currentEps = empRisk(data, w) - empRiskCP(a, b, w);
        cout << "Current epsilon = " << currentEps << endl;
    }
    while(currentEps>epsilon && t<tMax);

    cout << endl;
    cout << "Training completed." << endl;
    cout << "Achieved epsilon: " << currentEps << " (required - " << epsilon << ")" << endl;
    cout << "Number of iterations: " << t << " (max - " << tMax << ")" << endl;





//    Vec w;
//    w.resize(2);
//    w[0] = 0.8; w[1] = -0.8;

//    cout << "Omega = " << Omega(w) << endl;
//    cout << "empRisk = " << empRisk(data, w) << endl;
//    cout << "subGradient = " << empRiskSubgradient(data, w) << endl;
    //-----------------------------------


    // вектор betta должен быть решением задачи оптимизации,
    // но здесь он назначается равным [1;1;1...1], в целях тестирования
    betta.resize(d);
    for(int i = 0;i<d;i++)
    {
        betta[i] = w[i];
    }


}


Real SVM::Predict(Vec sample) const
{
    // проверка входных данных
    if(betta.size()==0 || betta.size()!=sample.size())
    {
        throw SVM::Exception();
    }

    return inner_prod(betta, sample);
}


//------------------------------------------------------------------------------------------

Real Omega(const Vec& w)
{
    return Real(0.5)*inner_prod(w, w);
}

Real empRisk(const Data& data, const Vec& w)
{
    Mat samples = data.Samples();
    Vec responses = data.Responses();
    vector<int> trainSampleIdx = data.TrainSampleIdx();

    Real sum = 0;
    for(int i = 0;i<trainSampleIdx.size();i++)
    {
        int idx = trainSampleIdx[i];
        sum += max(  Real(0), 1-responses[idx]*Product(idx, samples, w)  );
    }
    sum /= trainSampleIdx.size();
    return sum;
}



Real Product(int row, const Mat& mat, const Vec& vec)
{
    assert(mat.size2()==vec.size());
    assert(vec.size()!=0);

    Real sum = 0;
    for(int i = 0;i<vec.size();i++)
    {
        sum += mat(row, i)*vec[i];
    }
    return sum;
}



Vec empRiskSubgradient(const Data& data, const Vec& w)
{
    Mat samples = data.Samples();
    Vec responses = data.Responses();
    vector<int> trainSampleIdx = data.TrainSampleIdx();

    Vec subgr;
    subgr.resize(w.size());
    for(int i = 0;i<subgr.size();i++)
    {
        subgr[i] = 0;
    }

    for(int i = 0;i<trainSampleIdx.size();i++)
    {
        int idx = trainSampleIdx[i];
        Real maxVal = max(  Real(0), 1-responses[idx]*Product(idx, samples, w)  );
        if(maxVal > 0)
        {
            subgr = subgr + (-responses[idx])*boost::numeric::ublas::matrix_row<Mat>(samples, idx);
        }
    }
    subgr /= trainSampleIdx.size();
    return subgr;
}


Real empRiskCP(const vector<Vec>& a, const vector<Real>& b, const Vec& w)
{
    Real val = inner_prod(w, a[0]) + b[0];
    for(int i = 1;i<a.size();i++)
    {
        val = std::max(val, inner_prod(w, a[i]) + b[i]);
    }
    return val;
}
