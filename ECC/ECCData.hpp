#pragma once
#include "ARFFparser/arff_parser.h"
#include "ARFFparser/arff_data.h"
#include "Util.hpp"

class MultilabelInstance
{
private:
	size_t numLabels;
	size_t numAttribs;
	std::vector<double> data;

public:
	MultilabelInstance(const ArffInstance* inst, size_t _numLabels);

	std::vector<double>& getData();
	bool getLabel(size_t labelIndex);
	size_t getNumLabels();
	size_t getNumAttribs();
	size_t getValueCount();
};

class MultilabelPrediction
{
private: 
	std::vector<double> confidence;

public:
	MultilabelPrediction(double* begin, double* end);

	size_t getNumLabels();
	double getConfidence(size_t labelIndex);
	bool getPrediction(size_t labelIndex, double threshold);
};

class ECCData
{
private:
	size_t numAttributes;
	size_t numLabels;

	std::vector<MultilabelInstance> instances;

public:
	ECCData(size_t labelCount, std::string arrfFile);
	ECCData(const std::vector<MultilabelInstance>& _instances, size_t _numAttributes, size_t _numLabels);
	~ECCData();

	std::vector<MultilabelInstance>& getInstances();

	size_t getAttribCount() const;
	size_t getLabelCount() const;
	size_t getValueCount() const;
	size_t getSize() const;
};

