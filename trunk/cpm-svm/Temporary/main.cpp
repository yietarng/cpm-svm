#include "SVM.h"
#include "functions.h"

#include <string>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	CvMLData data;
	string dataPath = "C:\\Users\\Сергей\\Documents\\Studies\\MachineLearning\\code\\ExperimentsSystem\\Data\\Classification\\";
	data.read_csv("SPECT_Heart.data");
	data.set_response_idx(0);
	CvTrainTestSplit split = CvTrainTestSplit(0.23f, false);
	data.set_train_test_split(&split);

	LSVM svm;
	svm.train(&data);
	cout << svm.calc_error(&data, CV_TRAIN_ERROR) << endl;


	return 0;
}