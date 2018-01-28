#pragma once
#include "atf_library/atf.h";
#include "ECCExecutorNew.h";

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

	double tuneClassifyStepFunc(atf::configuration config)
	{
		std::map<std::string, int> cfgmap;
		for (auto it = config.begin(); it != config.end(); ++it)
		{
			cfgmap[it->first] = it->second;
		}

		return eccEx.tuneClassifyStep(cfgmap, true);
	}

	double tuneClassifyFinalFunc(atf::configuration config)
	{
		std::map<std::string, int> cfgmap;
		for (auto it = config.begin(); it != config.end(); ++it)
		{
			cfgmap[it->first] = it->second;
		}

		return eccEx.tuneClassifyFinal(cfgmap);
	}

	double tuneBuildFunc(atf::configuration config)
	{
		return eccEx.tuneBuild(config["NUM_WI"], config["NUM_WG"]);
	}

	void tuneBuild(int treesPerRun)
	{
		auto tp_NUM_WG = atf::tp("NUM_WG", atf::interval(1, treesPerRun),
			[&](auto tp_NUM_WG) { return (treesPerRun % tp_NUM_WG) == 0; });
		auto tp_NUM_WI = atf::tp("NUM_WI", atf::interval(1, treesPerRun),
			[&](auto tp_NUM_WI) { return ((treesPerRun / tp_NUM_WG) % tp_NUM_WI) == 0; });

		auto tuner = atf::exhaustive();//atf::open_tuner(atf::cond::evaluations(1000));
		auto best_config = tuner(G(tp_NUM_WG, tp_NUM_WI))(std::bind(&ECCTuner::tuneBuildFunc, this, std::placeholders::_1));

		for (auto it = best_config.begin(); it != best_config.end(); ++it)
		{
			bestBuildConfig[it->first] = it->second;
		}
	}

	void tuneClassifyStep(int numInstances)
	{
		auto tp_NUM_WG_CHAINS_SC = atf::tp("NUM_WG_CHAINS_SC", atf::interval(1, numChains),
			[&](auto tp_NUM_WG_CHAINS_SC) { return (numChains % tp_NUM_WG_CHAINS_SC) == 0; });
		auto tp_NUM_WG_INSTANCES_SC = atf::tp("NUM_WG_INSTANCES_SC", atf::interval(1, (int)numInstances),
			[&](auto tp_NUM_WG_INSTANCES_SC) { return (numInstances % tp_NUM_WG_INSTANCES_SC) == 0; });
		auto tp_NUM_WG_TREES_SC = atf::tp("NUM_WG_TREES_SC", atf::interval(1, numTrees),
			[&](auto tp_NUM_WG_TREES_SC) { return (numTrees % tp_NUM_WG_TREES_SC) == 0; });
		auto tp_NUM_WI_CHAINS_SC = atf::tp("NUM_WI_CHAINS_SC", atf::interval(1, numChains),
			[&](auto tp_NUM_WI_CHAINS_SC) { return ((numChains / tp_NUM_WG_CHAINS_SC) % tp_NUM_WI_CHAINS_SC) == 0; });
		auto tp_NUM_WI_INSTANCES_SC = atf::tp("NUM_WI_INSTANCES_SC", atf::interval(1, (int)numInstances),
			[&](auto tp_NUM_WI_INSTANCES_SC) { return ((numInstances / tp_NUM_WG_INSTANCES_SC) % tp_NUM_WI_INSTANCES_SC) == 0; });
		auto tp_NUM_WI_TREES_SC = atf::tp("NUM_WI_TREES_SC", atf::interval(1, numTrees),
			[&](auto tp_NUM_WI_TREES_SC) { return ((numTrees / tp_NUM_WG_TREES_SC) % tp_NUM_WI_TREES_SC) == 0; });
		auto tp_NUM_WI_TREES_SR = atf::tp("NUM_WI_TREES_SR", atf::interval(1, numTrees),
			[&](auto tp_NUM_WI_TREES_SR) { return (tp_NUM_WG_TREES_SC % tp_NUM_WI_TREES_SR) == 0; });

		auto tuner = atf::exhaustive();//atf::open_tuner(atf::cond::evaluations(1000));
		auto best_config = tuner(
			G(tp_NUM_WG_CHAINS_SC, tp_NUM_WI_CHAINS_SC),
			G(tp_NUM_WG_INSTANCES_SC, tp_NUM_WI_INSTANCES_SC),
			G(tp_NUM_WG_TREES_SC, tp_NUM_WI_TREES_SC, tp_NUM_WI_TREES_SR)
		)(std::bind(&ECCTuner::tuneClassifyStepFunc, this, std::placeholders::_1));

		for (auto it = best_config.begin(); it != best_config.end(); ++it)
		{
			bestStepConfig[it->first] = it->second;
		}
	}

	void tuneClassifyFinal(int numInstances)
	{
		auto tp_NUM_WG_CHAINS_FC = atf::tp("NUM_WG_CHAINS_FC", atf::interval(1, numChains),
			[&](auto tp_NUM_WG_CHAINS_FC) { return (numChains % tp_NUM_WG_CHAINS_FC) == 0; });
		auto tp_NUM_WG_INSTANCES_FC = atf::tp("NUM_WG_INSTANCES_FC", atf::interval(1, (int)numInstances),
			[&](auto tp_NUM_WG_INSTANCES_FC) { return (numInstances % tp_NUM_WG_INSTANCES_FC) == 0; });
		auto tp_NUM_WG_LABELS_FC = atf::tp("NUM_WG_LABELS_FC", atf::interval(1, numLabels),
			[&](auto tp_NUM_WG_LABELS_FC) { return (numLabels % tp_NUM_WG_LABELS_FC) == 0; });
		auto tp_NUM_WI_CHAINS_FC = atf::tp("NUM_WI_CHAINS_FC", atf::interval(1, numChains),
			[&](auto tp_NUM_WI_CHAINS_FC) { return ((numChains / tp_NUM_WG_CHAINS_FC) % tp_NUM_WI_CHAINS_FC) == 0; });
		auto tp_NUM_WI_INSTANCES_FC = atf::tp("NUM_WI_INSTANCES_FC", atf::interval(1, (int)numInstances),
			[&](auto tp_NUM_WI_INSTANCES_FC) { return ((numInstances / tp_NUM_WG_INSTANCES_FC) % tp_NUM_WI_INSTANCES_FC) == 0; });
		auto tp_NUM_WI_LABELS_FC = atf::tp("NUM_WI_LABELS_FC", atf::interval(1, numLabels),
			[&](auto tp_NUM_WI_LABELS_FC) { return ((numLabels / tp_NUM_WG_LABELS_FC) % tp_NUM_WI_LABELS_FC) == 0; });
		auto tp_NUM_WI_CHAINS_FR = atf::tp("NUM_WI_CHAINS_FR", atf::interval(1, numChains),
			[&](auto tp_NUM_WI_CHAINS_FR) { return (tp_NUM_WG_CHAINS_FC % tp_NUM_WI_CHAINS_FR) == 0; });

		auto tuner = atf::exhaustive();
		auto best_config = tuner(
			G(tp_NUM_WG_CHAINS_FC, tp_NUM_WI_CHAINS_FC, tp_NUM_WI_CHAINS_FR),
			G(tp_NUM_WG_INSTANCES_FC, tp_NUM_WI_INSTANCES_FC),
			G(tp_NUM_WG_LABELS_FC, tp_NUM_WI_LABELS_FC)
		)(std::bind(&ECCTuner::tuneClassifyFinalFunc, this, std::placeholders::_1));

		for (auto it = best_config.begin(); it != best_config.end(); ++it)
		{
			bestFinalConfig[it->first] = it->second;
		}
	}

public:
	ECCTuner(int _maxLevel, int _maxAttributes, int _numAttributes, int _numTrees, int _numLabels, int _numChains, int _ensembleSubSetSize, int _forestSubSetSize)
		: eccEx(_maxLevel, _maxAttributes, _numAttributes, _numTrees, _numLabels, _numChains, _ensembleSubSetSize, _forestSubSetSize),
		numChains(_numChains), numTrees(_numTrees), numLabels(_numLabels)
	{
	}

	void tune(ECCData& buildData, int treesPerRun, ECCData& classifyData)
	{
		eccEx.prepareBuild(buildData, treesPerRun);
		tuneBuild(treesPerRun);
		eccEx.finishBuild();
		eccEx.runBuild(buildData, treesPerRun, bestBuildConfig["NUM_WI"], bestBuildConfig["NUM_WG"]);
		eccEx.prepareClassify(classifyData);
		tuneClassifyStep(classifyData.getSize());
		eccEx.tuneClassifyStep(bestStepConfig, false);
		tuneClassifyFinal(classifyData.getSize());
		eccEx.finishClassify();
	}

	auto getBestBuildConfig()
	{
		return bestBuildConfig;
	}

	auto getBestStepConfig()
	{
		return bestStepConfig;
	}

	auto getBestFinalConfig()
	{
		return bestFinalConfig;
	}
};