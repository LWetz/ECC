#pragma once

#include<CL/cl.h>

#include"EnsembleOfClassifierChains.hpp"
#include"PlatformUtil.hpp"
#include"ECCData.hpp"
#include "Kernel.hpp"
#include <chrono>
#include <climits>
#include "Util.hpp"

class ECCExecutorOld
{
	EnsembleOfClassifierChains *ecc;

	double* nodeValues;
	int* nodeIndices;
	Buffer labelOrderBuffer;
	int maxLevel;
	int numTrees;
	int numLabels;
	int numChains;
	int maxAttributes;
	int numAttributes;

	std::string buildSource;
	std::string classifySource;
	std::string classifyFixSource;

	std::vector<int> partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc);

	Measurement measurement;
public:
	ECCExecutorOld(int _maxLevel, int _maxAttributes, int _numAttributes, int _numTrees, int _numLabels, int _numChains, int _ensembleSubSetSize, int _forestSubSetSize);

	void runBuild(ECCData& data, int treeLimit);

public:
	std::vector<MultilabelPrediction> runClassify(ECCData& data, bool fix = true);

	Measurement getMeasurement();

	~ECCExecutorOld();
};

