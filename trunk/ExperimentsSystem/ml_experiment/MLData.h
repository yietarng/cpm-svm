#pragma once


#include "Scheme.h"


#include <string>
#include <iostream>
#include <ml.h>



namespace mle
{
	class MLData : public CvMLData
	{
	public:
		static void LoadPatterns(const std::string& patternsPath);
		static void CreateMLDataHeaders(const std::string& _directory, std::ostream& stream, 
			const std::vector<std::string>& extentions = std::vector<std::string>::vector());
		
		void set_train_test_split(const CvTrainTestSplit* spl);
		void set_train_test_split(const cv::Mat& mask);
		void Load(const std::string& header_filename);
		bool SaveHeader(const std::string& header_filename, const std::string& csv_filename = "");
		cv::Mat SubData(int type) const;
	};
}