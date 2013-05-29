#pragma once

#include <list>
#include <string>
#include <ml.h>


class Scheme
{
public:
	Scheme();
	Scheme(const std::string& filename);
	bool Load(const std::string& filename);
	void Check(const cv::FileStorage* storage, std::vector<int>* defList = 0);
	bool IsLoaded();
	~Scheme();

private:
	bool isLoaded;
	cv::FileStorage* pattern;

	void CheckBranch(const cv::FileNodeIterator& st_iter, const cv::FileNodeIterator& pt_iter,
		std::vector<int>* defList = 0);
	void CheckTerminalNode(const cv::FileNodeIterator& st_pNode, const cv::FileNodeIterator& pt_pNode, 
		std::vector<int>* defList = 0);
	void CheckNonTerminalNode(const cv::FileNodeIterator& st_iter, const cv::FileNodeIterator& pt_iter,
		std::vector<int>* defList = 0);
};
