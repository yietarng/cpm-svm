#include "svm.h"

SVM::SVM()
{
}


void SVM::Train(Mat samples, Vec responses)
{
    betta.clear();


    int n = samples.size1(); // число прецедентов в обучающей выборке
    int d = samples.size2(); // размерность пространства признаков

    // проверка входных данных
    if(n<=0 || d<=0 || n!=responses.size())
    {
        throw SVM::Exception();
    }


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
