#pragma once


#include <ml.h>

struct LSVMParams
{
	LSVMParams()
	{
		C = 1;
		epsilon = 0.01f;
	}
	LSVMParams(float _C, float _epsilon) : C(_C), epsilon(_epsilon) {}

	float C;
	float epsilon;
};


class LSVM : public CvStatModel
{
public:

	LSVM();
	~LSVM();
	void clear();

	bool train(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx=0, 
		const CvMat* sampleIdx=0, LSVMParams params=LSVMParams());
	bool train(const cv::Mat& trainData, const cv::Mat& responses, const cv::Mat& _varIdx=cv::Mat(), 
		const cv::Mat& _sampleIdx=cv::Mat(), LSVMParams params=LSVMParams());
	bool train(CvMLData* trainData, LSVMParams params = LSVMParams());

	float predict(const cv::Mat& sample) const;
	float predict(const CvMat* sample) const;

	float calc_error(CvMLData* data, int type, std::vector<float>* resp = 0) const;

	void save(const char* filename, const char* name=0) const;
	void load(const char* filename, const char* name=0);
	void write(CvFileStorage* storage, const char* name) const;
	void read(CvFileStorage* storage, CvFileNode* node);

private:
	enum {BINARY, MULTICLASS};
	int type;
	cv::Mat normal;
	std::map<float, int> classLabels;

	float GetUnitLabel(float label) const;
};