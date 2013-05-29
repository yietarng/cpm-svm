#include "ToString.h"
#include "ErrorHandling.h"

#include <sstream>
#include <iostream>
#include <iomanip>


using namespace std;


string ToString(const char* value)
{
	if(!value)
	{
		ES_Error("");
	}
	string str;
	stringstream sStream (stringstream::in|stringstream::out);
	sStream<<value;
	sStream>>str;
	return str;
}

std::string Name(const std::string& filename)
{
	//טלÿ פאיכא בוח נאסרטנוםטÿ
	int i = filename.find_last_of("\\");
	int j = filename.find_last_of(".");
	string name = filename.substr(i+1, j-i-1);
	return name;
}

//===========================================
void Print(const std::string& str, std::ostream& stream)
{
	stream << left << setw(nameLength) << str;
}