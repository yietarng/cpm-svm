#pragma once


#include "MLParameters.h"
#include "Measures.h"
#include "ParamGrid.h"
#include "MLData.h"



namespace mle
{

enum MLTask {Regression, Classification};



class StatModel
{
public:
	
	StatModel();
	~StatModel();
	void Clear();
	const CvStatModel* CvPointer() const;
	void Train(MLData* data, const StatModelParams& params);
	Measures* CalcMeasures(	MLData* data, int type, Time* predictingTime = 0, 
							std::vector<float>* resp = 0) const;
	float CalcError(MLData* data, int type) const;
	int Task() const;
	void Fit(MLData* data, StatModelParams& params, std::ostream* stream = 0);
	
private:
	
	int algorithm;
	CvStatModel* model;
	int task;
	std::map<float, int> classes;
};

}


