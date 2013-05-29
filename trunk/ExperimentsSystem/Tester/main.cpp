#include "MLExperiment.h"

#include <fstream>
#include <ml.h>

using namespace std;


int main(int argc, char* argv[])
{
	//setlocale(LC_ALL, "Russian");

	//первый аргумент - путь к папке с шаблонами
	//второй - путь к папке с результатами
	//остальные для бенчмарков

	LoadPatterns(argv[1]);
	string logPath = string(argv[2])+"\\log.txt";
	//ostream& f = cout;
	fstream f(logPath.c_str(), ios_base::out);
	
	for(int i = 3;i<argc;i++)
	{
		f << endl << "####################################################" << endl;
		try
		{
			Experiment(argv[i], f, false);
		}
		catch(cv::Exception& error)
		{
			f << error.what();
		}
	}
	f << endl;
	f.close();
	return 0;
}