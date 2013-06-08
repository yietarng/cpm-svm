#ifdef _DEBUG
	#pragma comment(lib, "..\\Debug\\bsvm.lib")
#endif

#ifndef _DEBUG
	#pragma comment(lib, "..\\Release\\bsvm.lib")
#endif


#include <iostream>
#include <fstream>
#include <clocale>
#include <assert.h>
#include <float.h>

#include "bsvm.h"
//#include "libqp.h"
#include <windows.h>



using namespace std;
using namespace cv;



int main()
{
	setlocale(LC_ALL, "Russian");

	CvMLData data;
	data.set_delimiter(' ');
	data.read_csv("..\\..\\Data\\Classification\\adult.data");
	data.set_response_idx(14);
	CvTrainTestSplit spl(0.75f, false);
	data.set_train_test_split(&spl);
	/*Mat resp = data.get_responses();
	Mat varIdx = data.get_var_idx();
	cout << resp;*/
	BSVM svm;

	int start = GetTickCount();
	svm.train(&data, BSVMParams(BSVMParams::BMRM_SOLVER, 0.1, 0.01, 1000));
	int finish = GetTickCount();

	cout << "время обучения " << float(finish-start)/1000 << " секунд" << endl;
	cout << "train error:" << svm.calc_error(&data, CV_TRAIN_ERROR) << endl;
	cout << "test error:" << svm.calc_error(&data, CV_TEST_ERROR) << endl;


}





