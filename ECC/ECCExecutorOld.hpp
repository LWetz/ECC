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
	size_t maxLevel;
	size_t numTrees;
	size_t numLabels;
	size_t numChains;
	size_t maxAttributes;
	size_t numAttributes;

	std::string buildSource;
	std::string classifySource;
	std::string classifyFixSource;

	std::vector<int> partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc);

	Measurement measurement;
public:
	ECCExecutorOld(size_t _maxLevel, size_t _maxAttributes, size_t _numAttributes, size_t _numTrees, size_t _numLabels, size_t _numChains, size_t _ensembleSubSetSize, size_t _forestSubSetSize);

	void runBuild(ECCData& data, size_t treeLimit);

public:
	std::vector<MultilabelPrediction> runClassify(ECCData& data, bool fix = true);

	Measurement getMeasurement();

	~ECCExecutorOld();
};

