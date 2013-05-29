#include "Scheme.h"

#include <stack>
#include <string>
#include <iostream>
#include <sstream>
#include "ErrorHandling.h"


using namespace std;
using namespace cv;

const std::string alternativesStr = "ALTERNATIVES";

Scheme::Scheme()
{
	pattern = new FileStorage();
	isLoaded = false;
}

Scheme::Scheme(const std::string& filename)
{
	pattern = new FileStorage();
	Load(filename);
}

Scheme::~Scheme()
{
	delete pattern; 
}

bool Scheme::Load(const std::string& filename)
{
	isLoaded = pattern->open(filename, FileStorage::READ, "1.0");
	return isLoaded;	
}

bool Scheme::IsLoaded()
{
	return isLoaded;
}

void Scheme::Check(const cv::FileStorage* storage, std::vector<int>* defList)
{
	if(!isLoaded)
	{
		ES_Error("шаблон не загружен");
	}
	else
	{
		CheckBranch(storage->root().begin(), pattern->root().begin(), defList);
	}
}


void Scheme::CheckTerminalNode(const FileNodeIterator& st_pNode, const FileNodeIterator& pt_pNode, 
					   std::vector<int>* defList)
{
	FileNode st_node = *st_pNode, pt_node = *pt_pNode;

	string param =  pt_node;
	int i = param.find_first_of("_");
	if(i>=0)
	{
		char slash = 92;
		string values = param.substr(i+1, param.length())+slash;
		int k = 0;
		string valuesList;
		while(values.length()!=0)
		{
			int i = values.find_first_of(slash);
			string value = values.substr(0, i);
			valuesList += " "+value;
			if(value==string(st_node))
			{
				if(defList!=0)
				{
					defList->push_back(k);
				}
				break;
			}
			values = values.substr(i+1, values.length());
			k++;
		}
		if(values.length()==0)
		{
			ES_Error("поле "+pt_node.name()+" должно принимать одно из следующих значений:"
						+valuesList);
		}
	}
	else
	{
		int isQ = st_node.isInt()+st_node.isReal();
		string type = param;
		if(
			(type=="string"		&& !st_node.isString())
		||	(type=="int"		&& !st_node.isInt())
		||	(type=="unsigned"	&& (!st_node.isInt() || int(st_node)<0))
		||	(type=="double"		&& !isQ)
		||	(type=="float"		&& !isQ)
		)
		{
			ES_Error("поле "+pt_node.name()+" должно принимать значения типа "+type);
		}
		if(type=="bool")
		{
			string s;
			st_node >> s;
			if(s!="true" && s!="false")
			{
				ES_Error("поле "+st_node.name()+" должно принимать значения <true> или <false>");
			}
		}
		if(type=="char")
		{
			int l = string(st_node).length();
			if(!st_node.isString() || l>1)
			{
				ES_Error("поле "+st_node.name()+" должно принимать значения типа char");;
			}
		}
	}		
}

void Scheme::CheckNonTerminalNode(const cv::FileNodeIterator& st_iter, const cv::FileNodeIterator& pt_iter,
								  std::vector<int>* defList)
{
	FileNode st_node = *st_iter, pt_node = *pt_iter;
	string st_name = st_node.name(), pt_name = pt_node.name();
	if(st_name!=pt_name)
	{
		ES_Error("вместо "+st_name+" должно быть "+pt_name);
	}
	if(pt_node.size()!=st_node.size())
	{
		if(pt_node.isSeq())
		{
			FileNodeIterator pt_iter = pt_node.begin(), st_iter = st_node.begin();
			while(st_iter!=st_node.end())
			{
				CheckBranch(st_iter, pt_iter, defList);
				st_iter++;
			}
		}
		else
		{
			ES_Error("вершина "+st_node.name()+" - не совпадает число потомков");
		}
	}
	else
	{

		FileNodeIterator pt_iter = pt_node.begin(), st_iter = st_node.begin();
		while(st_iter!=st_node.end())
		{
			CheckBranch(st_iter, pt_iter, defList);
			pt_iter++;
			st_iter++;
		}
	}
}

void Scheme::CheckBranch(const cv::FileNodeIterator& st_iter, const cv::FileNodeIterator& pt_iter, 
						 std::vector<int>* defList)
{
	FileNode pt_node = *pt_iter;
	FileNode st_node = *st_iter;
	string pt_name = pt_node.name();
	string st_name = st_node.name();

	if(pt_name.substr(0, alternativesStr.length())==alternativesStr)
	{
		bool flag = false;
		string namesList;
		for(FileNodeIterator iter = pt_node.begin();iter!=pt_node.end();iter++)
		{
			FileNode node = *iter;
			string name = node.name();
			namesList += " "+name;
			if(name==st_name)
			{
				CheckBranch(st_iter, iter, defList);
				flag = true;
				break;
			}
		}
		if(flag==false)
		{
			ES_Error("вместо "+st_node.name()+" требуется один из вариантов:"+namesList);
		}
		return;
	}

	//cout << pt_node.name() << " " << st_node.name() << endl;
	bool pt_IsTerm = pt_iter==pt_node.begin();
	bool st_IsTerm = st_iter==st_node.begin();
	//cout << pt_IsTerm << " " << st_IsTerm << endl;
	if(pt_IsTerm && st_IsTerm)
	{
		CheckTerminalNode(st_iter, pt_iter, defList);
	}
	else
	{
		if(!pt_IsTerm && !st_IsTerm)
		{
			
			CheckNonTerminalNode(st_iter, pt_iter, defList);
		}
		else
		{
			ES_Error(pt_node.name()+" и "+st_node.name()+
				" - не совпадает терминальность");
		}
	}
	/*
	FileNodeIterator pt_current = pt_node.begin();
	FileNodeIterator st_current = st_node.begin();
	stack<FileNodeIterator> pt_stack, st_stack;
	stack<bool> isSeq_stack;
	pt_stack.push(pt_current);
	st_stack.push(st_current);
	isSeq_stack.push(false);
	bool isParentSeq = false;
	while(!pt_stack.empty())
	{
		pt_current = pt_stack.top();pt_stack.pop();
		isParentSeq = isSeq_stack.top();isSeq_stack.pop();
		st_current = st_stack.top();st_stack.pop();
		CheckNode(st_current, pt_current, defList);
		
		FileNodeIterator st_left = st_current;
		st_left++;
		if(pt_current.remaining!=1)
		{
			if(st_left.remaining==0)
			{
				ES_Error("отсутствует элемент "+(*pt_current).name());
			}
			FileNodeIterator pt_left = pt_current;
			pt_left++;
			DoNodesMismatch(st_left, pt_left);
			st_stack.push(st_left);
			pt_stack.push(pt_left);
			isSeq_stack.push((*pt_current).isSeq());
		}
		else
		{
			if(isParentSeq && (st_left.remaining!=0))
			{
				DoNodesMismatch(st_left, pt_current);
				st_stack.push(st_left);
				pt_stack.push(pt_current);
				isSeq_stack.push(true);
			}
		}
		FileNodeIterator pt_down = (*pt_current).begin();
		if((*pt_down).begin()!=pt_current)
		{
			FileNodeIterator st_down = (*st_current).begin();
			string st_name = (*st_down).name();
			string pt_name = (*pt_down).name();
			if(st_down==st_current)
			{
				ES_Error("у "+st_name+" отсутствует элемент нижнего уровня "+pt_name);
			}
			DoNodesMismatch(st_down, pt_down);
			st_stack.push(st_down);
			pt_stack.push(pt_down);
			isSeq_stack.push((*pt_current).isSeq());
		}
	}*/
}

