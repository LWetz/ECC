#pragma once

#ifndef _WIN32
#include "ECCExecutorNew.hpp"
#include "atf_library/include/tp_value.hpp"

class ECCTuner
{
private:
	ECCExecutorNew eccEx;
	size_t numChains;
	size_t numTrees;
	size_t numLabels;

	double tuneClassifyStepFunc(atf::configuration config);
	double tuneClassifyFinalFunc(atf::configuration config);
	double tuneBuildFunc(atf::configuration config);

	Configuration runBuildTuner(size_t treesPerRun);
	Configuration runClassifyStepTuner(size_t numInstances);
	Configuration runClassifyFinalTuner(size_t numInstances);
public:
	ECCTuner(size_t _maxLevel, size_t _maxAttributes, size_t _numAttributes, size_t _numTrees, size_t _numLabels, size_t _numChains, size_t _ensembleSubSetSize, size_t _forestSubSetSize);

	Configuration tuneBuild(ECCData& buildData, size_t treesPerRun);
	Configuration tuneClassifyStep(ECCData& buildData, size_t treesPerRun, ECCData& classifyData, Configuration config);
	Configuration tuneClassifyFinal(ECCData& buildData, size_t treesPerRun, ECCData& classifyData, Configuration config);
};

#endif