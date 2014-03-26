#include "svm.h"

#include <vector>
#include <assert.h>

using namespace std;


Real empRisk(const Data& data, const Vec& w);
Vec empRiskSubgradient(const Data& data, const Vec& w);
Real Omega(const Vec& w);


Real Product(int row, const Mat& mat, const Vec& vec);


//--------------------------------------------------------------------------------

SVM::SVM()
{
}


void SVM::Train(const Data& data)
{
    // Очистка модели
    betta.clear();


    Mat samples = data.Samples();
    Vec responses = data.Responses();
    vector<int> testSampleIdx = data.TestSampleIdx();
    vector<int> trainSampleIdx = data.TrainSampleIdx();

    int n = samples.size1(); // число прецедентов в обучающей выборке
    int d = samples.size2(); // размерность пространства признаков

    // проверка входных данных
    if(n<=0 || d<=0 || n!=responses.size())
    {
        throw SVM::Exception();
    }


    // Обучение
    //-----------------------------------
//    Vec w;
//    w.resize(2);
//    w[0] = 1; w[1] = -2;

//    cout << "empRisk = " << empRisk(data, w) << endl;
    //-----------------------------------


    // вектор betta должен быть решением задачи оптимизации,
    // но здесь он назначается равным [1;1;1...1], в целях тестирования
    betta.resize(d);
    for(int i = 0;i<d;i++)
    {
        betta[i] = 1;
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
    int d = samples.size2();

    Real sum = 0;
    for(int i = 0;i<trainSampleIdx.size();i++)
    {
        int idx = trainSampleIdx[i];
        cout << "idx = " << idx << ' ';
        sum += max(  Real(0), 1-responses[idx]*Product(idx, samples, w)  );
        cout << "sum = " << sum << endl;
    }
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
