#include <iostream>
#include <fstream>

#include "svm.h"

using namespace std;
using namespace cv;


int main()
{
	CvMLData data;
	data.read_csv("tic-tac-toe.data");
	data.set_response_idx(9);
	CvTrainTestSplit spl(10, true);
	data.set_train_test_split(&spl);
	/*Mat resp = data.get_responses();
	Mat varIdx = data.get_var_idx();
	cout << resp;*/
	BSVM svm;
	svm.train(&data);
}

