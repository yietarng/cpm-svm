#include "MLBenchmark.h"


#include <ml.h>
#include "ErrorHandling.h"


using namespace std;
using namespace cv;


Scheme MLBenchmark::scheme;


void ReadFileList(const FileNode& node, FileList& fl);


void MLBenchmark::LoadPatterns(const std::string& path)
{
	if(!scheme.Load(path+"\\"+"benchmark.yml"))
	{
		ES_Error("Не удалось загрузить бенчмарк");
	}
}

bool MLBenchmark::Load(const std::string& filename)
{
	FileStorage storage(filename, FileStorage::READ);
	if(!storage.isOpened() )
	{
		return false;
	}
	this->filename = filename;
	scheme.Check(&storage);
	FileNode top = storage.getFirstTopLevelNode();
	ReadFileList(top["data_sets"], data); 
	ReadFileList(top["algorithms"], algorithms);

	storage.release();
	return true;
}

void ReadFileList(const FileNode& node, FileList& fl)
{
	node["directory"] >> fl.directory;
	fl.name.clear();
	FileNode config_files = node["config_files"];
	for(FileNodeIterator iter = config_files.begin();iter!=config_files.end();iter++)
	{
		fl.name.push_back(*iter);
	}
}