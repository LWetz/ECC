#pragma once
#include "ARFFparser/arff_parser.h"
#include "ARFFparser/arff_data.h"
#include "Util.hpp"

class MultilabelInstance
{
private:
	int numLabels;
	int numAttribs;
	std::vector<double> data;

public:
	MultilabelInstance(const ArffInstance* inst, int _numLabels);

	std::vector<double>& getData();
	bool getLabel(int labelIndex);
	int getNumLabels();
	int getNumAttribs();
	int getValueCount();
};

class MultilabelPrediction
{
private: 
	std::vector<double> confidence;

public:
	MultilabelPrediction(double* begin, double* end);

	int getNumLabels();
	double getConfidence(int labelIndex);
	bool getPrediction(int labelIndex, double threshold);
};

class ECCData
{
private:
	int numAttributes;
	int numLabels;

	std::vector<MultilabelInstance> instances;

public:
	ECCData(int labelCount, std::string arrfFile);
	ECCData(const std::vector<MultilabelInstance>& _instances, int _numAttributes, int _numLabels);
	~ECCData();

	std::vector<MultilabelInstance>& getInstances();

	int getAttribCount() const;
	int getLabelCount() const;
	int getValueCount() const;
	size_t getSize() const;
};

