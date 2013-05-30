#include <iostream>
#include <fstream>
#include <clocale>
#include <assert.h>
#include <float.h>

#include "bsvm.h"
#include "libqp.h"

using namespace std;
using namespace cv;


int main()
{
	setlocale(LC_ALL, "Russian");

	CvMLData data;
	data.set_delimiter(',');
	data.read_csv("D:\\Code\\Data\\Classification\\tic-tac-toe.data");
	data.set_response_idx(9);
	CvTrainTestSplit spl(0.75f, true);
	data.set_train_test_split(&spl);
	/*Mat resp = data.get_responses();
	Mat varIdx = data.get_var_idx();
	cout << resp;*/
	BSVM svm;
	svm.train(&data, BSVMParams(BSVMParams::BMRM_SOLVER, 1, 0.01, 1000));
	cout << "train error:" << svm.calc_error(&data, CV_TRAIN_ERROR) << endl;
	cout << "test error:" << svm.calc_error(&data, CV_TEST_ERROR) << endl;

}



