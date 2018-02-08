#pragma once

#ifndef _WIN32
#include "ECCExecutorNew.hpp"
#include "atf_library/include/tp_value.hpp"

class ECCTuner
{
private:
	ECCExecutorNew eccEx;
	int numChains;
	int numTrees;
	int numLabels;

	double tuneClassifyStepFunc(atf::configuration config);
	double tuneClassifyFinalFunc(atf::configuration config);
	double tuneBuildFunc(atf::configuration config);

	Configuration runBuildTuner(int treesPerRun);
	Configuration runClassifyStepTuner(int numInstances);
	Configuration runClassifyFinalTuner(int numInstances);
public:
	ECCTuner(int _maxLevel, int _maxAttributes, int _numAttributes, int _numTrees, int _numLabels, int _numChains, int _ensembleSubSetSize, int _forestSubSetSize);

	Configuration tuneBuild(ECCData& buildData, int treesPerRun);
	Configuration tuneClassifyStep(ECCData& buildData, int treesPerRun, ECCData& classifyData, Configuration config);
	Configuration tuneClassifyFinal(ECCData& buildData, int treesPerRun, ECCData& classifyData, Configuration config);
};

#endif