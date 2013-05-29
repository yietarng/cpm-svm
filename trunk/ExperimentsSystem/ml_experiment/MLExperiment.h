#pragma once


#include <string>
#include <iostream>
#include <vector>


void LoadPatterns(const std::string& directory);
void Experiment(const std::string& benchmarkFilename, std::ostream& log, 
				bool printCVErrors = false);
void CreateMLDataHeaders(const std::string& directory, std::ostream& stream, 
			const std::vector<std::string>& extentions = std::vector<std::string>::vector());



