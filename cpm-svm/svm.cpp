#include "svm.h"

#include <vector>
#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "solve_qp.h"







using namespace std;


Real empRisk(const Data& data, const Vec& w);
Real empRiskCP(const vector<Vec>& a, const vector<Real>& b, const Vec& w);
Vec empRiskSubgradient(const Data& data, const Vec& w);
Real Omega(const Vec& w);


//Real Product(int row, const Mat& mat, const Vec& vec);
Real SparseProduct(int rowIdx, const SparseMat& mat, const Vec& vec);

//--------------------------------------------------------------------------------

SVM::SVM()
{
}


void SVM::Train(const Data& data, const Real lambda, const Real epsilon_abs,
                const Real epsilon_tol, const int tMax)
{
    // Очистка модели
    betta.clear();



    if(!data.IsLoaded())
    {
        throw SVM::Exception();
    }


    Vec responses = data.Responses();
    for(unsigned i = 0;i<responses.size();i++)
    {
        assert(responses[i]==-1 || responses[i]==1);
    }

    int d = data.VarNumber(); // размерность пространства признаков



    // Обучение
    //-----------------------------------
    Vec w(d);
    std::fill(w.begin(), w.end(), 0);
    int t = 0;
    Real currentEps = -1;
    vector<Vec> a;
    vector<Real> b;



    do
    {
        t++;

        long long time_a = -gettimeus();
        a.push_back( empRiskSubgradient(data, w) );
        time_a += gettimeus();


        long long time_b = -gettimeus();
        b.push_back( empRisk(data, w) - inner_prod(w, a.back()) );
        time_b += gettimeus();

#ifdef BMRM_INFO
        cout << endl << "Iteration " << t << endl;
//        cout << "empRisk(w) = " << empRisk(data, w) << endl;


        cout << "Subgradient calculating time: " << double(time_a)/1000000 << " seconds" << endl;
        cout << "Coef calculating time: " << double(time_b)/1000000 << " seconds" << endl;

//        cout << "w[" << t-1 << "] = " << w << endl;
//        cout << "a[" << t << "] = " << a.back() << endl;
//        cout << "b[" << t << "] = " << b.back() << endl;
#endif



        //==========================================
        // argmin begin
        //==========================================
        Vec alpha(t);

        long long time_qp = -gettimeus();
        SolveQP(a, b, lambda, epsilon_tol*0.5, alpha);
        time_qp += gettimeus();




        // Получение w из alpha
        Vec temp(d);
        std::fill(temp.begin(), temp.end(), 0);
        for(int i = 0;i<t;i++)
        {
            temp = temp + alpha[i]*a[i];
        }
        w = -temp/lambda;

//#ifdef BMRM_INFO
        cout << "J(w) = " << lambda*Omega(w)+empRisk(data, w) << endl;
//        cout << "EmpRisk(w) = " << empRisk(data, w) << endl;
//        cout << "EmpRiskCP(w) = " << empRiskCP(a, b, w) << endl;
//#endif
        //==========================================
        // argmin end
        //==========================================


        currentEps = empRisk(data, w) - empRiskCP(a, b, w);

#ifdef BMRM_INFO
        cout << "QP solving time: " << double(time_qp)/1000000 << " seconds" << endl;

        cout << "Current epsilon = " << currentEps << endl;
#endif

    }
    while(
          currentEps>epsilon_abs
          && currentEps>epsilon_tol*(lambda*Omega(w)+empRisk(data, w))
          && t<tMax
         );

    cout << endl << endl;
    cout << "BMRM => J(w) = " << lambda*Omega(w)+empRisk(data, w) << endl;
    printf("BMRM => Achieved epsilon: %e \n", currentEps);
    printf("BMRM => Required abs epsilon: %e \n", epsilon_abs);
    printf("BMRM => Required tol epsilon: %e \n", epsilon_tol);
    cout << "BMRM => Number of iterations: " << t << " (max - " << tMax << ")" << endl;

    cout << "w = " << w << endl;


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

    if(inner_prod(betta, sample)<0)
    {
        return -1;
    }
    else
    {
        return 1;
    }
}


Real SVM::CalcError(const Data& data, int type) const
{
    const vector<int>& trainSampleIdx = data.TrainSampleIdx();
    const vector<int>& testSampleIdx = data.TestSampleIdx();
    const vector<int>* sampleIdxPtr;
    if(type==TRAIN)
    {
        sampleIdxPtr = &trainSampleIdx;
    }
    else // type==TEST
    {
        sampleIdxPtr = &testSampleIdx;
    }
    const vector<int>& sampleIdx = *sampleIdxPtr;


    assert( data.VarNumber() == betta.size() );
    assert( sampleIdx.size() != 0 );


    int errorCount = 0;
    for(unsigned i = 0;i<sampleIdx.size();i++)
    {
        int row = sampleIdx[i];
        assert(data.Responses()[row]==-1.0 || data.Responses()[row]==1.0);

        Real pred = SparseProduct(row, data.Samples(), betta);
        if(pred<0)
        {
            pred = -1;
        }
        else
        {
            pred = 1;
        }

        if(pred*data.Responses()[row]<0)
        {
            errorCount++;
        }

    }
    return Real(errorCount)/sampleIdx.size();
}


//------------------------------------------------------------------------------------------

Real Omega(const Vec& w)
{
    return Real(0.5)*inner_prod(w, w);
}

Real empRisk(const Data& data, const Vec& w)
{
//    Mat samples = data.Samples();
    Vec responses = data.Responses();
    vector<int> trainSampleIdx = data.TrainSampleIdx();

    Real sum = 0;
    for(unsigned i = 0;i<trainSampleIdx.size();i++)
    {
        int idx = trainSampleIdx[i];
//        sum += max(  Real(0), 1-responses[idx]*Product(idx, samples, w)  );
        sum += max(  Real(0), 1-responses[idx]*SparseProduct(idx, data.Samples(), w)  );
    }

    //=================================
    sum /= trainSampleIdx.size();
    //=================================

    return sum;
}



//Real Product(int row, const Mat& mat, const Vec& vec)
//{
//    assert(mat.size2()==vec.size());
//    assert(vec.size()!=0);

//    Real sum = 0;
//    for(int i = 0;i<vec.size();i++)
//    {
//        sum += mat(row, i)*vec[i];
//    }
//    return sum;
//}


Real SparseProduct(int rowIdx, const SparseMat& mat, const Vec& vec)
{
    Real result = 0;

    const list<Pair>& row =  mat[rowIdx];
    list<Pair>::const_iterator iter = row.begin();
    while(iter!=row.end())
    {
        result += iter->value*vec[iter->idx-1];
        iter++;
    }

    return result;
}






Vec empRiskSubgradient(const Data& data, const Vec& w)
{
//    Mat samples = data.Samples();
    Vec responses = data.Responses();
    vector<int> trainSampleIdx = data.TrainSampleIdx();

    Vec subgr;
    subgr.resize(w.size());
    for(unsigned i = 0;i<subgr.size();i++)
    {
        subgr[i] = 0;
    }

    for(unsigned i = 0;i<trainSampleIdx.size();i++)
    {
        int idx = trainSampleIdx[i];
//        Real maxVal = max(  Real(0), 1-responses[idx]*Product(idx, samples, w)  );
        Real maxVal = max(  Real(0), 1-responses[idx]*SparseProduct(idx, data.Samples(), w)  );
        if(maxVal > 0)
        {
//            subgr = subgr + (-responses[idx])*boost::numeric::ublas::matrix_row<Mat>(samples, idx);
//            subgr = subgr + (-responses[idx])*boost::numeric::ublas::matrix_row<Mat>(data.Samples(), idx);
            const list<Pair>& row = data.Samples()[idx];
            list<Pair>::const_iterator iter = row.begin();
            while(iter!=row.end())
            {
                subgr[iter->idx-1] += -responses[idx]*iter->value;
                iter++;
            }
        }
    }

    //=================================
    subgr /= trainSampleIdx.size();
    //=================================

    return subgr;
}


Real empRiskCP(const vector<Vec>& a, const vector<Real>& b, const Vec& w)
{
    Real val = inner_prod(w, a[0]) + b[0];
    for(unsigned i = 1;i<a.size();i++)
    {
        val = std::max(val, inner_prod(w, a[i]) + b[i]);
    }
    return val;
}
