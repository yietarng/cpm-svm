#pragma once


#include <ml.h>
#include <vector>
#include <iostream>
#include <windows.h>
#include <iostream>


class Time
{
public:
	Time();
	void Start();
	void Finish();
	friend std::ostream& operator << (std::ostream& stream, const Time& time);

	int seconds;
	int minutes;
	int hours;
	int mSeconds;
private:
	unsigned long ms;
};

class Measures abstract
{
public:
	virtual void Clear() = 0;
	virtual void Calculate(const std::vector<float>& actual, const std::vector<float>& predicted) = 0;
	virtual float Error() const = 0;
	friend std::ostream& operator<<(std::ostream& stream, const Measures& mes);

protected:
	int AreEqual(float a, float b);
	virtual void PrintObject(std::ostream& stream) const = 0;
};

class ClMeasures : public Measures
{
public:
	ClMeasures();
	ClMeasures(const std::vector<float>& actual, const std::vector<float>& predicted);
	virtual void Calculate(const std::vector<float>& actual, const std::vector<float>& predicted);
	virtual void Clear();
	virtual float Error() const;

	float accuracy;
	float precision;
	float fScore;
	cv::Mat confusion;
private:
	virtual void PrintObject(std::ostream& stream) const;
	void GetConfusionMatrix(const std::vector<float>& actual, const std::vector<float>& predicted);
	void CalcMeasures();
};

class ReMeasures : public Measures
{
public:
	ReMeasures();
	ReMeasures(const std::vector<float>& actual, const std::vector<float>& predicted);
	virtual void Calculate(const std::vector<float>& actual, const std::vector<float>& predicted);
	virtual void Clear();
	virtual float Error() const;

	float meanSqErr;
	float meanAbsErr;
private:
	virtual void PrintObject(std::ostream& stream) const;
};

void AddClasses(const std::vector<float>& v, std::map<float, int>& classes);





