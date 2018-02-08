#pragma once

#include<CL/cl.h>

#include "EnsembleOfClassifierChains.hpp"
#include "PlatformUtil.hpp"
#include "ECCData.hpp"
#include "Kernel.hpp"
#include <chrono>
#include <climits>
#include "Util.hpp"

class ECCExecutorNew
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
	std::string stepCalcSource;
	std::string stepReduceSource;
	std::string finalCalcSource;
	std::string finalReduceSource;

	std::vector<int> partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc);

	struct BuildData
	{
		Buffer tmpNodeIndexBuffer;
		Buffer tmpNodeValueBuffer;
		Buffer dataBuffer;
		Buffer instancesBuffer;
		Buffer instancesLengthBuffer;
		Buffer instancesNextBuffer;
		Buffer instancesNextLengthBuffer;
		Buffer seedsBuffer;

		size_t numTrees;
		size_t numInstances;
	};

	BuildData *buildData;

	struct ClassifyData
	{
		Buffer dataBuffer;
		Buffer resultBuffer;
		Buffer labelBuffer;
		Buffer stepNodeValueBuffer;
		Buffer stepNodeIndexBuffer;

		size_t numInstances;
	};

	ClassifyData *classifyData;

	Measurement measurement;

public:
	ECCExecutorNew(size_t _maxLevel, size_t _maxAttributes, size_t _numAttributes, size_t _numTrees, size_t _numLabels, size_t _numChains, size_t _ensembleSubSetSize, size_t _forestSubSetSize);

	void prepareBuild(ECCData& data, size_t treesPerRun);
	double tuneBuild(size_t workitems, size_t workgroups);
	void finishBuild();

	void runBuild(ECCData& data, size_t treesPerRun, size_t workitems, size_t workgroups);

private:
	typedef struct TreeVote
	{
		double result;
		int vote;
	}TreeVote;

public:
	void prepareClassify(ECCData& data);
	double tuneClassifyStep(Configuration config, bool oneStep = true);
	double tuneClassifyFinal(Configuration config);
	void finishClassify();

	std::vector<MultilabelPrediction> runClassify(ECCData& data, Configuration config);

	Measurement getMeasurement();

	~ECCExecutorNew();
};

