#pragma once

#include <string>
#include <ml.h>
#include "ParamGrid.h"
#include <vector>
#include "bsvm.h"



namespace mle
{
	enum MLAlgorithm
	{
		DECISION_TREE,
		GRADIENT_BOOSTED_TREES, 
		SUPPORT_VECTOR_MACHINE, 
		RANDOM_TREES,
		EXTREMELY_RANDOMIZED_TREES,
		MULTI_LAYER_PERCEPTRON,
		B_SVM
	};
	const int algorithmCount = 7;
	const std::string algorithmName[algorithmCount] =
	{
		"decision_tree",
		"gradient_boosted_trees", 
		"support_vector_machine", 
		"random_trees",
		"extremely_randomized_trees",
		"multi_layer_perceptron",
		"b_svm"
	};

	struct MLP_Params
	{
		int hiddenLayersCount;
		int hiddenLayerSize;
		int activateFunc;
		double fParam1, fParam2;

		CvANN_MLP_TrainParams trainParams;
	};

	class StatModelParams
	{
	public:
		StatModelParams(const std::string& filename);
		~StatModelParams();
		static void LoadPatterns(const std::string& path);
		void Load(const std::string& filename);

		operator CvDTreeParams() const;
		operator CvGBTreesParams() const;
		operator CvRTParams() const;
		operator CvSVMParams() const;
		operator MLP_Params() const;
		operator BSVMParams() const;

		void InitVaryingByGrid();
		bool VaryByGrid();
		void SaveGridValues();
		void SetSavedGridValues();
		void PrintGridValues(std::ostream& stream);
		
		bool isCVParams() const;
		int Algorithm() const;
		int CVFolds() const;

	private:

		void ClearCV();

		int algorithm;
		CvDTreeParams DTree;
		CvGBTreesParams GBTrees;	
		CvRTParams RTrees;
		CvSVMParams SVM;
		MLP_Params MLP;
		BSVMParams bsvm;

		bool CVflag;
		int cvFolds;
		std::vector<ParamGrid*> grids;
	};

}
