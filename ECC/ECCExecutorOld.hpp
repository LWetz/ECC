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

	bool partitionInstance;
	bool oldKernelMode;

	double* nodeValues;
	int* nodeIndices;
	Buffer labelOrderBuffer;
	int maxLevel;
	int forestSize;
	int chainSize;
	int ensembleSize;
	int maxAttributes;

	size_t oldTime;

	std::vector<int> partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc);
public:
	ECCExecutorOld(int _maxLevel, int _maxAttributes, int _forestSize);

	void runBuild(ECCData& data, int treesPerRun, int ensembleSize, int chainsPerRun, int ensembleSubSetSize, int forestSubSetSize);

public:
	void runClassifyOld(ECCData& data, std::vector<double>& values, std::vector<int>& votes, bool fix = true);

	size_t getTime();
	~ECCExecutorOld();
};

