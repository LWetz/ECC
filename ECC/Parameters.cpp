#include "ECCExecutor.h"

#define NUM_CHAINS 8
#define NUM_TREES 8
#define MAX_LEVEL 10

ECCExecutor *ecc;
ECCData *evalData;
std::vector<MultilabelInstance> evalCopy;

std::vector<double> valOld, valNew;
std::vector<int> voteOld, voteNew;

int numAttributes = 0;
int numLabels = 0;
int numInstances = 0;

std::map<std::string, int> extraParams;

bool measureStep;

namespace atf
{
	typedef std::map<std::string, size_t> configuration;
}

size_t tune(atf::configuration config){
	std::map<std::string, int> params = extraParams;
	for (auto it = config.begin(); it!=config.end(); ++it)
		params[it->first] = it->second;

	params["NUM_WG_CHAINS_SR"] = params["NUM_WG_CHAINS_SC"];
	params["NUM_WG_INSTANCES_SR"] = params["NUM_WG_INSTANCES_SC"];
	params["NUM_WI_CHAINS_SR"] = params["NUM_WI_CHAINS_SC"];
	params["NUM_WI_INSTANCES_SR"] = params["NUM_WI_INSTANCES_SC"];
	params["NUM_WI_TREES_SR"] = params["NUM_WI_TREES_SC"];

    params["NUM_WG_LABELS_FR"] = params["NUM_WG_LABELS_FC"];
    params["NUM_WG_INSTANCES_FR"] = params["NUM_WG_INSTANCES_FC"];
    params["NUM_WI_LABELS_FR"] = params["NUM_WI_LABELS_FC"];
    params["NUM_WI_INSTANCES_FR"] = params["NUM_WI_INSTANCES_FC"];
    params["NUM_WI_CHAINS_FR"] = params["NUM_WI_CHAINS_FC"]; 

	ecc->runClassifyNew(*evalData, valNew, voteNew, params, measureStep);
	bool sameResult = true;
	size_t hitsOld = 0, hitsNew = 0;
        for (int i = 0; i < evalCopy.size(); ++i)
        {
        	MultilabelInstance iOrig = evalCopy[i];
        	int numL = iOrig.getNumLabels();
        	for (int l = 0; l < iOrig.getNumLabels(); ++l)
        	{
        		if (abs(valOld[i*numL + l] - valNew[i*numL + l]) > 0.001)
        			sameResult = false;
        		if (abs(voteOld[i*numL + l] - voteNew[i*numL + l]) > 0.001)
        			sameResult = false;
        		double predNew = valNew[i*numL + l] > 0 ? 1.0 : 0.0;
        		double predOld = valOld[i*numL + l] > 0 ? 1.0 : 0.0;
        		if (predOld == iOrig.getData()[l + iOrig.getNumAttribs()])
        			hitsOld++;
        		if (predNew == iOrig.getData()[l + iOrig.getNumAttribs()])
        			hitsNew++;
        	}
        }
	std::cout << "Time: " << ecc->getNewTime() << std::endl;
	std::cout << "Same Result: " << std::boolalpha << sameResult << std::endl;
    std::cout << "Prediction Performance: Old " << ((float)hitsOld / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0 << "% | New " << ((float)hitsNew / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0 << "%" << std::endl;

	return ecc->getNewTime();
}

void tuneClassify() {
	auto tp_NUM_WG_CHAINS_SC = atf::tp("NUM_WG_CHAINS_SC", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WG_CHAINS_SC) { return NUM_CHAINS  % tp_NUM_WG_CHAINS_SC == 0; });
	auto tp_NUM_WG_INSTANCES_SC = atf::tp("NUM_WG_INSTANCES_SC", atf::interval(1, numInstances),
		[&](auto tp_NUM_WG_INSTANCES_SC) { return numInstances % tp_NUM_WG_INSTANCES_SC == 0; });
	auto tp_NUM_WG_TREES_SC = atf::tp("NUM_WG_TREES_SC", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WG_TREES_SC) { return NUM_TREES % tp_NUM_WG_TREES_SC == 0; });
	auto tp_NUM_WI_CHAINS_SC = atf::tp("NUM_WI_CHAINS_SC", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WI_CHAINS_SC) { return NUM_CHAINS / tp_NUM_WG_CHAINS_SC % tp_NUM_WI_CHAINS_SC == 0; });
	auto tp_NUM_WI_INSTANCES_SC = atf::tp("NUM_WI_INSTANCES_SC", atf::interval(1, numInstances),
		[&](auto tp_NUM_WI_INSTANCES_SC) { return numInstances / tp_NUM_WG_INSTANCES_SC % tp_NUM_WI_INS$
	auto tp_NUM_WI_TREES_SC = atf::tp("NUM_WI_TREES_SC", atf::interval(1, NUM_TREES),
			[&](auto tp_NUM_WI_TREES_SC) { return NUM_TREES / tp_NUM_WG_TREES_SC % tp_NUM_WI_TREES_SC == 0; });

    extraParams["NUM_INSTANCES"] = numInstances;
    extraParams["NUM_LABELS"] = numLabels;
    extraParams["NUM_ATTRIBUTES"] = numAttributes;
    extraParams["NUM_CHAINS"] = NUM_CHAINS;
    extraParams["NUM_TREES"] = NUM_TREES;
    extraParams["MAX_LEVEL"] = MAX_LEVEL;
	extraParams["NODES_PER_TREE"] = pow(2.0f, MAX_LEVEL + 1) - 1;
	extraParams["NUM_WG_CHAINS_FC"] = extraParams["NUM_WG_INSTANCES_FC"] = extraParams["NUM_WG_LABELS_FC"] = extraParams["NUM_WI_CHAINS_FC"] = extraParams["NUM_WI_INSTANCES_FC"] = extraParams["NUM_WI_LABELS_FC"] = 1;
	

	measureStep = true;

	//      auto tuner = atf::exhaustive();
	auto tuner = atf::open_tuner(atf::cond::evaluations(1000));

	auto best_config = tuner(
		G(tp_NUM_WG_CHAINS_SC, tp_NUM_WI_CHAINS_SC),
		G(tp_NUM_WG_INSTANCES_SC, tp_NUM_WI_INSTANCES_SC),
		G(tp_NUM_WG_TREES_SC, tp_NUM_WI_TREES_SC)
	)(tune);
}

int main(int argc, char* argv[]) {
	std::cout << "START" << std::endl;
        if(!PlatformUtil::init("NVIDIA", "GTX"))
	{
		PlatformUtil::deinit();
		return -1;
	}
        std::cout << "Platform created!" << std::endl;
	{
		ECCData data(14, "data/yeast.arff");
		int trainSize = 0.67 * data.getSize();
		int evalSize = data.getSize() - trainSize;
		std::vector<MultilabelInstance> inputCopy = data.getInstances();
		std::vector<MultilabelInstance> trainInstances;
		std::vector<MultilabelInstance> evalInstances;
		trainInstances.reserve(trainSize);
		evalInstances.reserve(evalSize);
		for (int i = 0; i < trainSize; ++i)
		{
			int idx = Util::randomInt(inputCopy.size());
			trainInstances.push_back(inputCopy[idx]);
			inputCopy.erase(inputCopy.begin() + idx);
		}
		for (int i = 0; i < evalSize; ++i)
		{
			int idx = Util::randomInt(inputCopy.size());
			MultilabelInstance inst = inputCopy[idx];
			evalCopy.push_back(inst);
			for (int i = inst.getNumAttribs(); i < inst.getValueCount(); ++i)
			{
				inst.getData()[i] = 0.0;
			}
			evalInstances.push_back(inst);
			inputCopy.erase(inputCopy.begin() + idx);
		}
		ECCData trainData(trainInstances, data.getAttribCount(), data.getLabelCount());
		evalData = new ECCData(evalInstances, data.getAttribCount(), data.getLabelCount());
		ecc = new ECCExecutor(MAX_LEVEL, evalInstances[0].getValueCount(), NUM_TREES);
		ecc->runBuild(trainData, NUM_TREES, NUM_CHAINS, NUM_CHAINS, 100, 50);
		ecc->runClassifyOld(*evalData, valOld, voteOld);
		numLabels = evalData->getLabelCount();
		numAttributes = evalData->getAttribCount();
		numInstances = evalData->getInstances().size();
		tuneClassify();
	}
	PlatformUtil::deinit();
	system("Pause");
	return 0;
}
