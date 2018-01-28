#pragma once

#include<CL/cl.h>

#include"EnsembleOfClassifierChains.hpp"
#include"PlatformUtil.hpp"
#include"ECCData.hpp"
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

public:
	ECCExecutorNew(int _maxLevel, int _maxAttributes, int _numAttributes, int _numTrees, int _numLabels, int _numChains, int _ensembleSubSetSize, int _forestSubSetSize);

	void prepareBuild(ECCData& data, int treesPerRun);
	double tuneBuild(int workitems, int workgroups);
	void finishBuild();

	void runBuild(ECCData& data, int treesPerRun, int workitems, int workgroups);

private:
	typedef struct OutputAtom
	{
		double result;
		int vote;
	}OutputAtom;

public:
	void prepareClassify(ECCData& data);
	double tuneClassifyStep(std::map<std::string, int> config, int oneStep = true);
	double tuneClassifyFinal(std::map<std::string, int> config);
	void finishClassify();

	void runClassifyNew(ECCData& data, std::vector<double>& values, std::vector<int>& votes, std::map<std::string, int> config);

	~ECCExecutorNew();
};

