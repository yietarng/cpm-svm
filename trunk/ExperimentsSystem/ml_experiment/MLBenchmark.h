#pragma once


#include <string>
#include <list>
#include "Scheme.h"


struct FileList
{
	std::string directory;
	std::list<std::string> name;	
};

class MLBenchmark
{
public:
	void static LoadPatterns(const std::string& path);
	bool Load(const std::string& filename);

	std::string filename;
	FileList data;
	FileList algorithms;
	static Scheme scheme;
};