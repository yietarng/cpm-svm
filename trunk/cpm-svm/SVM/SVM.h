#pragma once


#include <ml.h>

struct LSVMParams
{
	LSVMParams()
	{
		C = 1;
	}
	LSVMParams(float _C) : C(_C) {}

	float C;
};

class LSVM : public CvStatModel
{
public:

	LSVM();
	~LSVM();
	void clear();

	bool train(const cv::Mat& trainData, const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(), 
		const cv::Mat& sampleIdx=cv::Mat(), LSVMParams params=LSVMParams());
	float predict(const cv::Mat& sample) const;

	void save(const char* filename, const char* name=0) const;
	void load(const char* filename, const char* name=0);
	void write(CvFileStorage* storage, const char* name) const;
	void read(CvFileStorage* storage, CvFileNode* node);

private:

	cv::Mat w;
};