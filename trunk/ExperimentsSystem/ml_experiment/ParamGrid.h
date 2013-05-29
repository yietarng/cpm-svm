#pragma once


#include <string>


#include "ErrorHandling.h"
#include "ToString.h"


namespace mle
{
	class ParamGrid abstract
	{
	public:
		enum {Logarithmic, Linear};

		virtual void FixValue() = 0;
		virtual void SetFixedValue() = 0;
		virtual void NextValue() = 0;
		virtual void SetMinValue() = 0;
		virtual bool isExceeded() = 0;
		virtual void PrintParameter(std::ostream& stream) = 0;

	protected:
		std::string name;
	};

	template <class Type>
	class PGrid : public ParamGrid
	{
	public:
		PGrid() {}
		PGrid(	const std::string& _name, Type _minVal, Type _maxVal, Type* _value, 
				float _step, int _scale);

		virtual void FixValue();
		virtual void SetFixedValue();
		virtual void NextValue();
		virtual void SetMinValue();
		virtual bool isExceeded();
		virtual void PrintParameter(std::ostream& stream);

	private:
		Type minVal;
		Type maxVal;
		Type* value;
		float step;
		int scale;
		Type fixedValue;
		double t;

		void Check() const;
	};

	template <class Type>
	PGrid<Type>::PGrid(	const std::string& _name, Type _minVal, Type _maxVal, Type* _value, 
						float _step, int _scale) :
		minVal(_minVal), maxVal(_maxVal), value(_value), 
		step(_step), scale(_scale) 
	{
		name = _name;
		*value = _minVal;
		fixedValue = _minVal;
		t = _minVal;

		Check();
	}

	template <class Type>
	void PGrid<Type>::FixValue()
	{
		fixedValue = *value;
	}

	template <class Type>
	void PGrid<Type>::SetFixedValue()
	{
		*value = fixedValue;
	}

	template <class Type>
	void PGrid<Type>::Check() const
	{
		if(maxVal<minVal || step<=0)
		{
			ES_Error("");
		}
		switch(scale)
		{
		case ParamGrid::Linear		: 
			break;
		case ParamGrid::Logarithmic : 
			if(step<=1 || minVal<=0)
			{
				ES_Error("");
			}
			break;
		default : ES_Error("");
		}
	}

	template <class Type>
	void PGrid<Type>::NextValue()
	{
		do
		{
			switch(scale)
			{
			case ParamGrid::Linear		: t += step;break;
			case ParamGrid::Logarithmic : t *= step;break;
			}
		}
		while(*value==Type(t));
		*value = Type(t);
	}

	template <class Type>
	void PGrid<Type>::SetMinValue()
	{
		t = minVal;
		*value = minVal;
	}

	template <class Type>
	bool PGrid<Type>::isExceeded()
	{
		return *value>maxVal;
	}

	
	template <class Type>
	void PGrid<Type>::PrintParameter(std::ostream& stream)
	{
		Print(name, *value, stream);
	}
}

