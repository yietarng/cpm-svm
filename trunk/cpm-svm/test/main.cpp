#include <iostream>
#include <fstream>
#include <clocale>
#include <assert.h>
#include <float.h>

#include "svm.h"
#include "libqp.h"

using namespace std;
using namespace cv;



const double* HCol(uint32_t j);

double H[2][2];

int main()
{
	setlocale(LC_ALL, "Russian");

	//CvMLData data;
	//data.set_delimiter(' ');
	//data.read_csv("test.txt");
	//data.set_response_idx(2);
	//CvTrainTestSplit spl(6, false);
	//data.set_train_test_split(&spl);
	///*Mat resp = data.get_responses();
	//Mat varIdx = data.get_var_idx();
	//cout << resp;*/
	//BSVM svm;
	//svm.train(&data);


	H[0][0] = 1;
	H[0][1] = 0;
	H[1][0] = 0;
	H[1][1] = 1;

	double diagH[2] = {1, 1};
	double f[2] = {0, 0};
	double b[1] = {1};
	uint32_t I[2] = {1, 1};
	uint8_t S[1] = {0};

	double x[2] = {0, 0};

	libqp_state_T state = libqp_splx_solver(HCol, diagH, f, b, I, S, x, 2, 100, 0.01, 0, DBL_MIN, 0);


}



const double* HCol(uint32_t j)
{
	assert(j==0 || j==1);
	return H[j];
}

