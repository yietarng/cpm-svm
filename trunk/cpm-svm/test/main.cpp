#include <iostream>
#include <fstream>
#include <clocale>
#include <assert.h>
#include <float.h>

#include "svm.h"
#include "libqp.h"

using namespace std;
using namespace cv;


int main()
{
	setlocale(LC_ALL, "Russian");

	CvMLData data;
	data.set_delimiter(' ');
	data.read_csv("test.txt");
	data.set_response_idx(2);
	CvTrainTestSplit spl(6, false);
	data.set_train_test_split(&spl);
	/*Mat resp = data.get_responses();
	Mat varIdx = data.get_var_idx();
	cout << resp;*/
	BSVM svm;
	svm.train(&data, BSVMParams(BSVMParams::BMRM_SOLVER, 1, 0.01, 3));

}



