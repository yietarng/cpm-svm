#include "MLParameters.h"
#include "Scheme.h"
#include "ErrorHandling.h"


using namespace std;
using namespace mle;
using namespace cv;


Scheme paramsPattern[algorithmCount];
Scheme fitParamsPattern[algorithmCount];

bool StringToBool(const string& str);
int GetTermCritType(int a);

void SetTopNodeAndGrids(const cv::FileNode* _top, std::vector<mle::ParamGrid*>* _grids);
template <typename Type> void AddGrid(Type* ptr, const std::string& name);


void StatModelParams::LoadPatterns(const std::string& path)
{
	string extention = ".yml";
	for(int i = 0;i<algorithmCount;i++)
	{
		if(	!paramsPattern[i].Load(path+"\\"+algorithmName[i]+extention) ||
			!fitParamsPattern[i].Load(path+"\\"+algorithmName[i]+"_fit"+extention))
		{
			ES_Error("ошибка загрузки шаблона для алгоритма "+algorithmName[i]);
		}
	}
}



StatModelParams::~StatModelParams()
{
	ClearCV();
}

void StatModelParams::Load(const std::string &filename)
{
	ClearCV();

	FileStorage storage(filename, FileStorage::READ);
	FileNode top = storage.getFirstTopLevelNode();
	algorithm = -1;
	for(int i = 0;i<algorithmCount;i++)
	{
		if(top.name()==algorithmName[i])
		{
			algorithm = i;
			break;
		}
	}
	if(algorithm==-1)
	{
		ES_Error("Ошибка в файле с параметрами: неправильное название алгоритма");
	}

	FileNode cv_folds = storage["cv_folds"];
	vector<int> defList;
	if(cv_folds.empty())
	{	
		paramsPattern[algorithm].Check(&storage, &defList);
	}
	else
	{
		fitParamsPattern[algorithm].Check(&storage, &defList);
		cvFolds = (int)cv_folds;
		if(cvFolds<1)
		{
			ES_Error("Ошибка в файле с параметрами: cv_folds<1");
		}
	}

	SetTopNodeAndGrids(&top, &grids);
	switch(algorithm)
	{
	case DECISION_TREE: 
		{
			DTree =  CvDTreeParams(	-1, 
									-1, 
									-1, 
									StringToBool(top["use_surrogates"]),      
									-1, 
									-1,
									StringToBool(top["use_1se_rule"]), 
									StringToBool(top["truncate_pruned_tree"]), 
									0);

			AddGrid(&DTree.max_depth, "max_depth");
			AddGrid(&DTree.min_sample_count, "min_sample_count");
			AddGrid(&DTree.regression_accuracy, "regression_accuracy");	
			AddGrid(&DTree.max_categories, "max_categories");
			AddGrid(&DTree.cv_folds, "cv_folds");
		}
		break;
	case GRADIENT_BOOSTED_TREES: 
		{
			this->GBTrees = CvGBTreesParams(defList.front(), 
											-1, -1, -1, -1,
											StringToBool(top["use_surrogates"]));

			AddGrid(&GBTrees.weak_count, "weak_count");
			AddGrid(&GBTrees.shrinkage, "shrinkage");
			AddGrid(&GBTrees.subsample_portion, "subsample_portion");
			AddGrid(&GBTrees.max_depth, "max_depth");
		}
		break;
	case SUPPORT_VECTOR_MACHINE:
		{
			int svm_type	= 100+defList[0];
			int kernel_type = defList[1];

			FileNode tcNode = top["term_crit"];
			CvTermCriteria termCrit;
			termCrit.epsilon = tcNode["epsilon"];
			termCrit.max_iter = tcNode["max_iter"];
			termCrit.type = GetTermCritType(defList.back());

			this->SVM = CvSVMParams(svm_type, kernel_type, -1, -1, -1, -1, -1, -1, 0, 
								    termCrit);

			AddGrid(&SVM.degree, "degree");
			AddGrid(&SVM.gamma, "gamma");
			AddGrid(&SVM.coef0, "coef0");
			AddGrid(&SVM.C, "Cvalue");
			AddGrid(&SVM.nu, "nu");
			AddGrid(&SVM.p, "p");
		}
		break;
	case B_SVM:
		{
			bsvm.epsilon = top["epsilon"];
			bsvm.solver = defList[0];
			bsvm.maxIter = top["max_iter"];
			bsvm.lambda = top["lambda"];
			break;
		}
	case RANDOM_TREES:
	case EXTREMELY_RANDOMIZED_TREES:
		{
			this->RTrees = CvRTParams(
								-1, 
								-1, 
								-1, 
								StringToBool(top["use_surrogates"]), 
								-1, 
								0, 
								StringToBool(top["calc_var_importance"]), 
								-1,
								-1, 
								-1, 
								GetTermCritType(defList.back()));

			AddGrid(&RTrees.max_depth, "max_depth");
			AddGrid(&RTrees.min_sample_count, "min_sample_count");
			AddGrid(&RTrees.regression_accuracy, "regression_accuracy");
			AddGrid(&RTrees.max_categories, "max_categories");
			AddGrid(&RTrees.nactive_vars, "nactive_vars");
			AddGrid(&RTrees.term_crit.max_iter, "max_num_of_trees_in_the_forest");
			AddGrid(&RTrees.term_crit.epsilon, "forest_accuracy");
		}
		break;
	case MULTI_LAYER_PERCEPTRON:
		{
			CvANN_MLP_TrainParams* ptr = &MLP.trainParams;
			ptr->train_method = defList[0];
			MLP.activateFunc = defList[1];
			ptr->term_crit.type = defList[2];

			FileNode node = top["bp"];
			SetTopNodeAndGrids(&node, &grids);
			AddGrid(&ptr->bp_dw_scale, "dw_scale");
			AddGrid(&ptr->bp_moment_scale, "moment_scale");

			node = top["rp"];
			SetTopNodeAndGrids(&node, &grids);
			AddGrid(&ptr->rp_dw0, "dw0");
			AddGrid(&ptr->rp_dw_max, "dw_max");
			AddGrid(&ptr->rp_dw_min, "dw_min");
			AddGrid(&ptr->rp_dw_minus, "dw_minus");
			AddGrid(&ptr->rp_dw_plus, "dw_plus");

			node = top["term_crit"];
			SetTopNodeAndGrids(&node, &grids);
			AddGrid(&ptr->term_crit.epsilon, "epsilon");
			AddGrid(&ptr->term_crit.max_iter, "max_iter");

			SetTopNodeAndGrids(&top, &grids);
			AddGrid(&MLP.hiddenLayersCount, "hidden_layers_count");
			AddGrid(&MLP.hiddenLayerSize, "hidden_layer_size");
			AddGrid(&MLP.fParam1, "fparam1");
			AddGrid(&MLP.fParam2, "fparam2");
		}
		break;
	}
	if(cvFolds && grids.size()==0)
	{
		ES_Error("Ошибка в файле с параметрами: не указано ни одного параметра для подбора кросс-валидацией");
	}
}

StatModelParams::operator BSVMParams() const
{
	ES_Assert(algorithm==B_SVM);
	return this->bsvm;
}

StatModelParams::operator CvDTreeParams() const
{
	ES_Assert(algorithm==DECISION_TREE);
	return this->DTree;
}

StatModelParams::operator CvRTParams() const
{
	ES_Assert(algorithm==RANDOM_TREES || algorithm==EXTREMELY_RANDOMIZED_TREES);
	return this->RTrees;
}

StatModelParams::operator CvSVMParams() const
{
	ES_Assert(algorithm==SUPPORT_VECTOR_MACHINE);
	return this->SVM;
}

StatModelParams::operator CvGBTreesParams() const
{
	ES_Assert(algorithm==GRADIENT_BOOSTED_TREES);
	return this->GBTrees;
}

StatModelParams::operator MLP_Params() const
{
	ES_Assert(algorithm==MULTI_LAYER_PERCEPTRON);
	return this->MLP;
}

StatModelParams::StatModelParams(const std::string& filename)
{
	Load(filename);
}

void StatModelParams::ClearCV()
{
	cvFolds = 0;
	for(unsigned i = 0;i<grids.size();i++)
	{
		delete grids[i];
	}
	grids.clear();
}

bool StatModelParams::isCVParams() const
{
	return cvFolds!=0;
}

void StatModelParams::InitVaryingByGrid()
{
	ES_Assert(isCVParams());

	for(unsigned i = 0;i<grids.size();i++)
	{
		grids[i]->SetMinValue();
	}
}

bool StatModelParams::VaryByGrid()
{
	ES_Assert(isCVParams());

	grids[0]->NextValue();
	int s = 0;
	while(grids[s]->isExceeded())
	{
		s++;
		if(s==grids.size())
		{
			return false;
		}
		grids[s]->NextValue();
	}
	if(s!=0)
	{
		for(int i = 0;i<s;i++)
		{
			grids[i]->SetMinValue();
		}
	}
	return true;
}

void StatModelParams::SaveGridValues()
{
	ES_Assert(isCVParams());

	for(unsigned i = 0;i<grids.size();i++)
	{
		grids[i]->FixValue();
	}
}

void StatModelParams::SetSavedGridValues()
{
	ES_Assert(isCVParams());

	for(unsigned i = 0;i<grids.size();i++)
	{
		grids[i]->SetFixedValue();
	}
}

void StatModelParams::PrintGridValues(std::ostream& stream)
{
	ES_Assert(isCVParams());

	for(unsigned i = 0;i<grids.size();i++)
	{
		grids[i]->PrintParameter(stream);
	}
}

int StatModelParams::Algorithm() const
{
	return algorithm;
}

int StatModelParams::CVFolds() const
{
	ES_Assert(isCVParams());
	return cvFolds;
}

//===========================================

bool StringToBool(const string& str)
{
	if(str=="true")
	{
		return true;
	}
	else
	{
		if(str=="false")
		{
			return false;
		}
		else
		{
			ES_Error("");
		}
	}
}


int GetTermCritType(int a)
{
	int type = 1+a;
	if(type==2) //оба критерия
	{
		type = 3;
	}
	return type;
}

//=================================================

const cv::FileNode* top = 0;
std::vector<mle::ParamGrid*>* grids = 0;

void SetTopNodeAndGrids(const cv::FileNode* _top, std::vector<mle::ParamGrid*>* _grids)
{
	top = _top;
	grids = _grids;
}

template <typename Type> void AddGrid(Type* ptr, const std::string& name)
{
	FileNode node = (*top)[name];
	if(node.empty()) 
	{
		node = (*top)[name+"_grid"];
		int scale;
		if((string)node["scale"]=="LOGARITHMIC")
		{
			scale = 0;
		}
		else //	  LINEAR
		{
			scale = 1;
		}
		PGrid<Type>* grid = new PGrid<Type>(name, node["min_value"], node["max_value"], 
			ptr, node["step"], scale);
		grids->push_back(grid);
	}
	else
	{
		*ptr = (Type)node;
	}
}