#include "ECCData.hpp"

MultilabelInstance::MultilabelInstance(const ArffInstance* inst, size_t _numLabels) : numLabels(_numLabels), numAttribs(inst->size() - _numLabels)
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

bool MultilabelInstance::getLabel(size_t labelIndex)
{
	return data[numAttribs + labelIndex] > 0.0;
}

size_t MultilabelInstance::getNumLabels()
{
	return numLabels;
}

size_t MultilabelInstance::getNumAttribs()
{
	return numAttribs;
}

size_t MultilabelInstance::getValueCount()
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

size_t MultilabelPrediction::getNumLabels()
{
	return confidence.size();
}

double MultilabelPrediction::getConfidence(size_t labelIndex)
{
	return confidence[labelIndex];
}

bool MultilabelPrediction::getPrediction(size_t labelIndex, double threshold)
{
	return getConfidence(labelIndex) > threshold;
}

ECCData::ECCData(size_t labelCount, std::string arrfFile) : numLabels(labelCount)
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

ECCData::ECCData(const std::vector<MultilabelInstance>& _instances, size_t _numAttributes, size_t _numLabels) : instances(_instances), numAttributes(_numAttributes), numLabels(_numLabels)
{
}

ECCData::~ECCData()
{}

std::vector<MultilabelInstance>& ECCData::getInstances()
{
	return instances;
}

size_t ECCData::getAttribCount() const
{
	return numAttributes;
}

size_t ECCData::getLabelCount() const
{
	return numLabels;
}

size_t ECCData::getValueCount() const
{
	return numLabels + numAttributes;
}

size_t ECCData::getSize() const
{
	return instances.size();
}

