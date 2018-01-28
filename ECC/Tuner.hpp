#pragma once

#ifndef _WIN32

#include "atf_library/atf.h"
#include "ECCExecutorNew.hpp"

class ECCTuner
{
private:
	ECCExecutorNew eccEx;
	int numChains;
	int numTrees;
	int numLabels;

	std::map<std::string, int> bestBuildConfig;
	std::map<std::string, int> bestStepConfig;
	std::map<std::string, int> bestFinalConfig;

	double tuneClassifyStepFunc(atf::configuration config);
	double tuneClassifyFinalFunc(atf::configuration config);
	double tuneBuildFunc(atf::configuration config);

	void tuneBuild(int treesPerRun);
	void tuneClassifyStep(int numInstances);
	void tuneClassifyFinal(int numInstances);
public:
	ECCTuner(int _maxLevel, int _maxAttributes, int _numAttributes, int _numTrees, int _numLabels, int _numChains, int _ensembleSubSetSize, int _forestSubSetSize);

	void tune(ECCData& buildData, int treesPerRun, ECCData& classifyData);

	std::map<std::string, int> getBestBuildConfig();
	std::map<std::string, int> getBestStepConfig();
	std::map<std::string, int> getBestFinalConfig();
};

#endif