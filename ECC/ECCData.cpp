#include "ECCData.hpp"

MultilabelInstance::MultilabelInstance(const ArffInstance* inst, int _numLabels) : numLabels(_numLabels), numAttribs(inst->size() - _numLabels)
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

std::vector<double>& MultilabelInstance::getData()
{
	return data;
}

bool MultilabelInstance::getLabel(int labelIndex)
{
	return data[numAttribs + labelIndex] > 0.0;
}

int MultilabelInstance::getNumLabels()
{
	return numLabels;
}

int MultilabelInstance::getNumAttribs()
{
	return numAttribs;
}

int MultilabelInstance::getValueCount()
{
	return numLabels + numAttribs;
}

MultilabelPrediction::MultilabelPrediction(double* begin, double* end)
{
	for (double* v = begin; v != end; ++v)
	{
		confidence.push_back(*v);
	}
}

int MultilabelPrediction::getNumLabels()
{
	return confidence.size();
}

double MultilabelPrediction::getConfidence(int labelIndex)
{
	return confidence[labelIndex];
}

bool MultilabelPrediction::getPrediction(int labelIndex, double threshold)
{
	return getConfidence(labelIndex) > threshold;
}

ECCData::ECCData(int labelCount, std::string arrfFile) : numLabels(labelCount)
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

ECCData::ECCData(const std::vector<MultilabelInstance>& _instances, int _numAttributes, int _numLabels) : instances(_instances), numAttributes(_numAttributes), numLabels(_numLabels)
{
}

ECCData::~ECCData()
{}

std::vector<MultilabelInstance>& ECCData::getInstances()
{
	return instances;
}

int ECCData::getAttribCount() const
{
	return numAttributes;
}

int ECCData::getLabelCount() const
{
	return numLabels;
}

int ECCData::getValueCount() const
{
	return numLabels + numAttributes;
}

int ECCData::getSize() const
{
	return instances.size();
}

