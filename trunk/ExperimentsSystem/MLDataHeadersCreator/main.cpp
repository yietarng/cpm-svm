#include "MLExperiment.h"

#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
	for(int i = 1;i<argc;i++)
	{
		string dir = argv[i];
		CreateMLDataHeaders(dir, cout);   
	}
	return 0;
}
