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
	int getNumLabels();
	int getNumAttribs();
	int getValueCount();
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
	int getSize() const;
};

