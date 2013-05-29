#pragma once

#include <string>
#include <iostream>
#include <iomanip>


std::string ToString(const char* value);
std::string Name(const std::string& filename);


template<typename Type> std::string ToString(const Type value)
{
	std::stringstream sStream (std::stringstream::in|std::stringstream::out);
	sStream << value;
	return sStream.str();
}

//==========================================

const int nameLength = 30;

template<typename Type> 
void Print(const std::string& name, const Type& value, std::ostream& stream)
{
	stream << std::left << std::setw(nameLength) << name << value << std::endl;
}

void Print(const std::string& str, std::ostream& stream);








