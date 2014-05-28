#include "svm.h"

#include <vector>
#include <assert.h>
#include <iostream>


using namespace std;




//=======================================
// mosek declarations begin
//=======================================

#include "mosek.h"
#include <stdio.h>
static void MSKAPI printstr(void *handle,
                            MSKCONST char str[])
{
  printf("%s",str);
}

//=======================================
// mosek declarations end
//=======================================



//using namespace boost::numeric::ublas;



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


    Vec responses = data.Responses();
    for(int i = 0;i<responses.size();i++)
    {
        assert(responses[i]==-1 || responses[i]==1);
    }

    int n = data.TrainSampleIdx().size(); // число прецедентов в обучающей выборке
    int d = data.Samples().size2(); // размерность пространства признаков

    // проверка входных данных
    if(n<=0 || d<=0)
    {
        throw SVM::Exception();
    }


    // Обучение
    //-----------------------------------
    const Real lambda = 1/cValue;
    Vec w(d);
    std::fill(w.begin(), w.end(), 0);
    int t = 0;
    Real currentEps = -1;
    vector<Vec> a;
    vector<Real> b;



    do
    {
        t++;
        a.push_back( empRiskSubgradient(data, w) );
        b.push_back( empRisk(data, w) - inner_prod(w, a.back()) );

#ifdef BMRM_INFO
        cout << endl << "Iteration " << t << endl;
//        cout << "empRisk(w) = " << empRisk(data, w) << endl;
        cout << "J(w) = " << Omega(w)+empRisk(data, w) << endl;
        cout << "w[" << t-1 << "] = " << w << endl;
        cout << "a[" << t << "] = " << a.back() << endl;
        cout << "b[" << t << "] = " << b.back() << endl;
#endif



        //==========================================
        // argmin begin
        //==========================================
        Vec alpha(t);


        //==========================================
        // mosek begin
        //==========================================
        MSKenv_t      env = NULL;
        MSKtask_t     task = NULL;
        MSKrescodee   resultCode;

        const int constraintNumber = 1;
        int varNumber = t;

        resultCode = MSK_makeenv(&env,NULL);
        assert(resultCode==MSK_RES_OK);

        resultCode = MSK_maketask(env, constraintNumber,varNumber,&task);
        assert(resultCode==MSK_RES_OK);

        resultCode = MSK_linkfunctotaskstream(task,MSK_STREAM_LOG,NULL,NULL);
        assert(resultCode==MSK_RES_OK);


        // Добавляем данные
        resultCode = MSK_appendcons(task, constraintNumber);
        assert(resultCode==MSK_RES_OK);

        resultCode = MSK_appendvars(task, varNumber);
        assert(resultCode==MSK_RES_OK);

        // Вектор c
        for(int j = 0;j<varNumber;j++)
        {
            resultCode = MSK_putcj(task,j,-b[j]);
            assert(resultCode==MSK_RES_OK);
        }

        // Ограничения на переменные: alpha[i]>=0
        for(int j = 0;j<varNumber;j++)
        {
            resultCode = MSK_putvarbound(task,
                                j,
                                MSK_BK_LO,
                                0,
                                +MSK_INFINITY);
            assert(resultCode==MSK_RES_OK);
        }


        // Матрица A = [1 1 1 ... 1 1]
        for(int j = 0;j<t;j++)
        {
            resultCode = MSK_putaij(task, 0, j, 1.0);
            assert(resultCode==MSK_RES_OK);
        }

        resultCode = MSK_putconbound(task,
                            0,
                            MSK_BK_FX,
                            1.0,
                            1.0);
        assert(resultCode==MSK_RES_OK);


        // матрица Q = 1/lambda*transp(A)*A
        for(int i = 0;i<t;i++)
        {
            for(int j = 0;j<=i;j++)
            {
                resultCode = MSK_putqobjij( task, i, j, 1.0/lambda*inner_prod(a[i], a[j]) );
                assert(resultCode==MSK_RES_OK);
            }
        }


        // Решение задачи
        MSKrescodee trmcode;

        /* Run optimizer */
        resultCode = MSK_optimizetrm(task,&trmcode);

        /* Print a summary containing information
           about the solution for debugging purposes*/
        MSK_solutionsummary (task,MSK_STREAM_MSG);



        MSKsolstae solsta;
        double        xx[varNumber];

        MSK_getsolsta (task,MSK_SOL_ITR,&solsta);

        switch(solsta)
        {
        case MSK_SOL_STA_OPTIMAL:
        case MSK_SOL_STA_NEAR_OPTIMAL:
            MSK_getxx(task,
                   MSK_SOL_ITR,    /* Request the interior solution. */
                   xx);
#ifdef BMRM_INFO
            printf("Optimal primal solution\n");
#endif
            for(int j=0; j<varNumber; ++j)
            {
#ifdef BMRM_INFO
                printf("x[%d]: %e\n",j,xx[j]);
#endif

                alpha[j] = xx[j];
            }


            break;
        case MSK_SOL_STA_DUAL_INFEAS_CER:
        case MSK_SOL_STA_PRIM_INFEAS_CER:
        case MSK_SOL_STA_NEAR_DUAL_INFEAS_CER:
        case MSK_SOL_STA_NEAR_PRIM_INFEAS_CER:
          printf("Primal or dual infeasibility certificate found.\n");
          break;

        case MSK_SOL_STA_UNKNOWN:
          printf("The status of the solution could not be determined.\n");
          break;
        default:
          printf("Other solution status.");
          break;
        }

        assert(solsta==MSK_SOL_STA_OPTIMAL);


        // delete
        MSK_deletetask(&task);
        MSK_deleteenv(&env);

        //==========================================
        // mosek end
        //==========================================


        // Получение w из alpha
        Vec temp(d);
        std::fill(temp.begin(), temp.end(), 0);
        for(int i = 0;i<t;i++)
        {
            temp = temp + alpha[i]*a[i];
        }
        temp = -temp/lambda;
        w = temp;

#ifdef BMRM_INFO
        cout << "w[" << t << "] = " << w << endl;
#endif
        //==========================================
        // argmin end
        //==========================================


        currentEps = empRisk(data, w) - empRiskCP(a, b, w);

#ifdef BMRM_INFO
        cout << "Current epsilon = " << currentEps << endl;
#endif

    }
    while(currentEps>epsilon && t<tMax);

    cout << "BMRM => J(w) = " << Omega(w)+empRisk(data, w) << endl;
    printf("BMRM => Achieved epsilon: %e (required - %e)\n", currentEps, epsilon);
    cout << "BMRM => Number of iterations: " << t << " (max - " << tMax << ")" << endl;




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
    Mat samples = data.Samples();

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


    assert( samples.size2() == betta.size() );
    assert( sampleIdx.size() != 0 );


    int errorCount = 0;
    for(int i = 0;i<sampleIdx.size();i++)
    {
        int row = sampleIdx[i];
        assert(data.Responses()[row]==-1.0 || data.Responses()[row]==1.0);

        Real pred = Product(row, samples, betta);
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
