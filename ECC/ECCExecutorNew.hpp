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
	int maxLevel;
	int numTrees;
	int numLabels;
	int numChains;
	int maxAttributes;
	int numAttributes;

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

		int numTrees;
		int numInstances;
	};

	BuildData *buildData;

	struct ClassifyData
	{
		Buffer dataBuffer;
		Buffer resultBuffer;
		Buffer labelBuffer;
		Buffer stepNodeValueBuffer;
		Buffer stepNodeIndexBuffer;

		int numInstances;
	};

	ClassifyData *classifyData;

	Measurement measurement;

public:
	ECCExecutorNew(int _maxLevel, int _maxAttributes, int _numAttributes, int _numTrees, int _numLabels, int _numChains, int _ensembleSubSetSize, int _forestSubSetSize);

	void prepareBuild(ECCData& data, int treesPerRun);
	double tuneBuild(size_t workitems, size_t workgroups);
	void finishBuild();

	void runBuild(ECCData& data, int treesPerRun, size_t workitems, size_t workgroups);

private:
	typedef struct TreeVote
	{
		double result;
		int vote;
	}TreeVote;

public:
	void prepareClassify(ECCData& data);
	double tuneClassifyStep(Configuration config, int oneStep = true);
	double tuneClassifyFinal(Configuration config);
	void finishClassify();

	std::vector<MultilabelPrediction> runClassify(ECCData& data, Configuration config);

	Measurement getMeasurement();

	~ECCExecutorNew();
};

