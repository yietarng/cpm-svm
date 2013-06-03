#ifndef SVM_MAC
#define SVM_MAC

#include <vector>
#include <ml.h>

struct BSVMParams
{
	enum {BMRM_SOLVER, LSBMRM_SOLVER};

	BSVMParams(int _solver, float _lambda, float _epsilon, int _maxIter)
		: solver(_solver), lambda(_lambda), epsilon(_epsilon), maxIter(_maxIter) {} 
	BSVMParams() 
	{
		*this = BSVMParams(BMRM_SOLVER, 1.f, 0.01f, 100);
	}

	int solver;
	float lambda;

	float epsilon;
	int maxIter;
};

class BSVM : public CvStatModel
{
public:

	BSVM();
	~BSVM();
	bool train(const cv::Mat& trainData, const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(), 
		const cv::Mat& sampleIdx=cv::Mat(), BSVMParams params = BSVMParams());
	bool train(CvMLData* trainData, BSVMParams params=BSVMParams() );
	float BSVM::calc_error(CvMLData* trainData, int type, std::vector<float>* resp=0);
	float predict(const cv::Mat& sample) const;
	void clear();

private:

	std::vector<float> betta;
	std::vector<int> var_idx;
	int sampleDim;
	std::vector<float> labels;
};


#endif SVM_MAC