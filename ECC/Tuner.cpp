#include "Tuner.hpp"

#ifndef _WIN32
#include "atf_library/atf.h"

#define EXHAUSTIVE_THRESHOLD ((size_t)25000)
#define ONE ((size_t)1)

template<typename... G_CLASSES >
std::unique_ptr<atf::tuner_with_constraints> make_tuner(G_CLASSES... G_classes)
{
	size_t search_space_size = 0;
	{
		auto tuner = atf::exhaustive(atf::cond::evaluations(0));
		tuner(G_classes...);
		search_space_size = tuner.search_space_size();
	}

	atf::tuner_with_constraints *tuner;
	if (search_space_size <= EXHAUSTIVE_THRESHOLD)
	{
		std::cout << "Exhaustive, " << search_space_size << " Evaluations" << std::endl;
		tuner = new atf::exhaustive_class<>(atf::cond::evaluations(search_space_size));
	}
	else
	{
		size_t evaluation = std::min(EXHAUSTIVE_THRESHOLD, (size_t)(0.1f * search_space_size));

		std::cout << "Open Tuner, " << evaluation << " Evaluations" << std::endl;
		tuner = new atf::open_tuner_class<>(atf::cond::evaluations(evaluation));
	}

	tuner = &(tuner->operator()(G_classes...));

	return std::unique_ptr<atf::tuner_with_constraints>(tuner);
}

double ECCTuner::tuneClassifyStepFunc(atf::configuration config)
{
	Configuration cfgmap;
	for (auto it = config.begin(); it != config.end(); ++it)
	{
		cfgmap[it->first] = it->second;
	}

	return eccEx.tuneClassifyStep(cfgmap, true);
}

double ECCTuner::tuneClassifyFinalFunc(atf::configuration config)
{
	Configuration cfgmap;
	for (auto it = config.begin(); it != config.end(); ++it)
	{
		cfgmap[it->first] = it->second;
	}

	return eccEx.tuneClassifyFinal(cfgmap);
}

double ECCTuner::tuneBuildFunc(atf::configuration config)
{
	return eccEx.tuneBuild(config["NUM_WI"], config["NUM_WG"]);
}

Configuration ECCTuner::runBuildTuner(size_t treesPerRun)
{
	auto tp_NUM_WG = atf::tp("NUM_WG", atf::interval(ONE, treesPerRun),
		[&](auto tp_NUM_WG) { return (treesPerRun % tp_NUM_WG) == 0; });
	auto tp_NUM_WI = atf::tp("NUM_WI", atf::interval(ONE, treesPerRun),
		[&](auto tp_NUM_WI) { return ((treesPerRun / tp_NUM_WG) % tp_NUM_WI) == 0; });

	auto tuner = make_tuner(G(tp_NUM_WG, tp_NUM_WI));
	auto best_config = tuner->operator()(std::bind(&ECCTuner::tuneBuildFunc, this, std::placeholders::_1));

	Configuration bestBuildConfig;
	for (auto it = best_config.begin(); it != best_config.end(); ++it)
	{
		bestBuildConfig[it->first] = it->second;
	}
	return bestBuildConfig;
}

Configuration ECCTuner::runClassifyStepTuner(size_t numInstances)
{
	auto tp_NUM_WG_CHAINS_SC = atf::tp("NUM_WG_CHAINS_SC", atf::interval(ONE, numChains),
		[&](auto tp_NUM_WG_CHAINS_SC) { return (numChains % tp_NUM_WG_CHAINS_SC) == 0; });
	auto tp_NUM_WG_INSTANCES_SC = atf::tp("NUM_WG_INSTANCES_SC", atf::interval(ONE, numInstances),
		[&](auto tp_NUM_WG_INSTANCES_SC) { return (numInstances % tp_NUM_WG_INSTANCES_SC) == 0; });
	auto tp_NUM_WG_TREES_SC = atf::tp("NUM_WG_TREES_SC", atf::interval(ONE, numTrees),
		[&](auto tp_NUM_WG_TREES_SC) { return (numTrees % tp_NUM_WG_TREES_SC) == 0; });
	auto tp_NUM_WI_CHAINS_SC = atf::tp("NUM_WI_CHAINS_SC", atf::interval(ONE, numChains),
		[&](auto tp_NUM_WI_CHAINS_SC) { return ((numChains / tp_NUM_WG_CHAINS_SC) % tp_NUM_WI_CHAINS_SC) == 0; });
	auto tp_NUM_WI_INSTANCES_SC = atf::tp("NUM_WI_INSTANCES_SC", atf::interval(ONE, numInstances),
		[&](auto tp_NUM_WI_INSTANCES_SC) { return ((numInstances / tp_NUM_WG_INSTANCES_SC) % tp_NUM_WI_INSTANCES_SC) == 0; });
	auto tp_NUM_WI_TREES_SC = atf::tp("NUM_WI_TREES_SC", atf::interval(ONE, numTrees),
		[&](auto tp_NUM_WI_TREES_SC) { return ((numTrees / tp_NUM_WG_TREES_SC) % tp_NUM_WI_TREES_SC) == 0; });
	auto tp_NUM_WI_TREES_SR = atf::tp("NUM_WI_TREES_SR", atf::interval(1ul, numTrees),
		[&](auto tp_NUM_WI_TREES_SR) { return (tp_NUM_WG_TREES_SC % tp_NUM_WI_TREES_SR) == 0; });

	auto tuner = make_tuner(
		G(tp_NUM_WG_CHAINS_SC, tp_NUM_WI_CHAINS_SC),
		G(tp_NUM_WG_INSTANCES_SC, tp_NUM_WI_INSTANCES_SC),
		G(tp_NUM_WG_TREES_SC, tp_NUM_WI_TREES_SC, tp_NUM_WI_TREES_SR));
	auto best_config = tuner->operator()(std::bind(&ECCTuner::tuneClassifyStepFunc, this, std::placeholders::_1));

	Configuration bestStepConfig;
	for (auto it = best_config.begin(); it != best_config.end(); ++it)
	{
		bestStepConfig[it->first] = it->second;
	}
	return bestStepConfig;
}

Configuration ECCTuner::runClassifyFinalTuner(size_t numInstances)
{
	auto tp_NUM_WG_CHAINS_FC = atf::tp("NUM_WG_CHAINS_FC", atf::interval(ONE, numChains),
		[&](auto tp_NUM_WG_CHAINS_FC) { return (numChains % tp_NUM_WG_CHAINS_FC) == 0; });
	auto tp_NUM_WG_INSTANCES_FC = atf::tp("NUM_WG_INSTANCES_FC", atf::interval(ONE, numInstances),
		[&](auto tp_NUM_WG_INSTANCES_FC) { return (numInstances % tp_NUM_WG_INSTANCES_FC) == 0; });
	auto tp_NUM_WG_LABELS_FC = atf::tp("NUM_WG_LABELS_FC", atf::interval(ONE, numLabels),
		[&](auto tp_NUM_WG_LABELS_FC) { return (numLabels % tp_NUM_WG_LABELS_FC) == 0; });
	auto tp_NUM_WI_CHAINS_FC = atf::tp("NUM_WI_CHAINS_FC", atf::interval(ONE, numChains),
		[&](auto tp_NUM_WI_CHAINS_FC) { return ((numChains / tp_NUM_WG_CHAINS_FC) % tp_NUM_WI_CHAINS_FC) == 0; });
	auto tp_NUM_WI_INSTANCES_FC = atf::tp("NUM_WI_INSTANCES_FC", atf::interval(ONE, numInstances),
		[&](auto tp_NUM_WI_INSTANCES_FC) { return ((numInstances / tp_NUM_WG_INSTANCES_FC) % tp_NUM_WI_INSTANCES_FC) == 0; });
	auto tp_NUM_WI_LABELS_FC = atf::tp("NUM_WI_LABELS_FC", atf::interval(ONE, numLabels),
		[&](auto tp_NUM_WI_LABELS_FC) { return ((numLabels / tp_NUM_WG_LABELS_FC) % tp_NUM_WI_LABELS_FC) == 0; });
	auto tp_NUM_WI_CHAINS_FR = atf::tp("NUM_WI_CHAINS_FR", atf::interval(ONE, numChains),
		[&](auto tp_NUM_WI_CHAINS_FR) { return (tp_NUM_WG_CHAINS_FC % tp_NUM_WI_CHAINS_FR) == 0; });

	auto tuner = make_tuner(
		G(tp_NUM_WG_CHAINS_FC, tp_NUM_WI_CHAINS_FC, tp_NUM_WI_CHAINS_FR),
		G(tp_NUM_WG_INSTANCES_FC, tp_NUM_WI_INSTANCES_FC),
		G(tp_NUM_WG_LABELS_FC, tp_NUM_WI_LABELS_FC));
	auto best_config = tuner->operator()(std::bind(&ECCTuner::tuneClassifyFinalFunc, this, std::placeholders::_1));

	Configuration bestFinalConfig;
	for (auto it = best_config.begin(); it != best_config.end(); ++it)
	{
		bestFinalConfig[it->first] = it->second;
	}
	return bestFinalConfig;
}

ECCTuner::ECCTuner(size_t _maxLevel, size_t _maxAttributes, size_t _numAttributes, size_t _numTrees, size_t _numLabels, size_t _numChains, size_t _ensembleSubSetSize, size_t _forestSubSetSize)
	: eccEx(_maxLevel, _maxAttributes, _numAttributes, _numTrees, _numLabels, _numChains, _ensembleSubSetSize, _forestSubSetSize),
	numChains(_numChains), numTrees(_numTrees), numLabels(_numLabels)
{
}

Configuration ECCTuner::tuneBuild(ECCData& buildData, size_t treesPerRun)
{
	eccEx.prepareBuild(buildData, treesPerRun);
	auto best_config = runBuildTuner(treesPerRun);
	eccEx.finishBuild();
	return best_config;
}

Configuration ECCTuner::tuneClassifyStep(ECCData& buildData, size_t treesPerRun, ECCData& classifyData, Configuration config)
{
	eccEx.runBuild(buildData, treesPerRun, config["NUM_WI"], config["NUM_WG"]);
	eccEx.prepareClassify(classifyData);
	auto best_config = runClassifyStepTuner(classifyData.getSize());
	eccEx.finishClassify();
	return best_config;
}

Configuration ECCTuner::tuneClassifyFinal(ECCData& buildData, size_t treesPerRun, ECCData& classifyData, Configuration config)
{
	eccEx.runBuild(buildData, treesPerRun, config["NUM_WI"], config["NUM_WG"]);
	eccEx.prepareClassify(classifyData);
	eccEx.tuneClassifyStep(config, false);
	auto best_config = runClassifyFinalTuner(classifyData.getSize());
	eccEx.finishClassify();
	return best_config;
}

#endif
