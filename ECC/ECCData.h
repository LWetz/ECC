#pragma once
#include "ARFFparser/arff_parser.h"
#include "ARFFparser/arff_data.h"
#include "Util.h"

class MultilabelInstance
{
private:
	int numLabels;
	int numAttribs;
	std::vector<double> data;

public:
	MultilabelInstance(const ArffInstance* inst, int _numLabels) : numLabels(_numLabels), numAttribs(inst->size() - _numLabels)
	{
		data.reserve(inst->size());
		
		for (int i = 0; i < inst->size(); ++i)
		{
			ArffValue* arffval = inst->get(i);
			double val;
			if (arffval->type() == STRING) //Labels are defined as nominals, which are stored as strings, try parsing
			{
				try
				{

					std::string str = std::string(*(inst->get(i)));

					if (i < (inst->size() - _numLabels))
					{
						val = std::stod(str);
					}
					else
					{
						if (str.compare("0") == 0)
							val = 0.0;
						else if (str.compare("1") == 0)
							val = 1.0;
						else
							throw;
					}
				}
				catch (...)
				{
					THROW("String parameter not of {0, 1}");
				}
			}
			else
			{
				val = *(inst->get(i));
			}
			data.push_back(val);
		}

	}

	std::vector<double>& getData()
	{
		return data;
	}

	int getNumLabels()
	{
		return numLabels;
	}

	int getNumAttribs()
	{
		return numAttribs;
	}

	int getValueCount()
	{
		return numLabels + numAttribs;
	}
};

class ECCData
{
private:
	int numAttributes;
	int numLabels;

	std::vector<MultilabelInstance> instances;

public:
	ECCData(int labelCount, std::string arrfFile) : numLabels(labelCount)
	{
		ArffParser parser(arrfFile);
		ArffData *data(parser.parse());

		numAttributes = data->num_attributes() - labelCount;
		instances.reserve(data->num_instances());
		for (int i = 0; i < data->num_instances(); i++)
		{
			instances.push_back(MultilabelInstance(data->get_instance(i), labelCount));
		}
	}

	ECCData(const std::vector<MultilabelInstance>& _instances, int _numAttributes, int _numLabels) : instances(_instances), numAttributes(_numAttributes), numLabels(_numLabels)
	{
	}

	~ECCData()
	{}

	std::vector<MultilabelInstance>& getInstances()
	{
		return instances;
	}

	int getAttribCount() const
	{
		return numAttributes;
	}

	int getLabelCount() const
	{
		return numLabels;
	}

	int getValueCount() const
	{
		return numLabels + numAttributes;
	}

	int getSize() const
	{
		return instances.size();
	}
};

