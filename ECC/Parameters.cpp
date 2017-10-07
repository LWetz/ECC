#include "ECCExecutor.h" 
#include "atf_library/atf.h" 

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

bool measureStep; //namespace atf //{ //	typedef std::map<std::string, size_t> configuration; //} 

size_t tune(atf::configuration config){
	std::map<std::string, int> params = extraParams;
	valNew.clear();
	voteNew.clear();
	for (auto it = config.begin(); it!=config.end(); ++it)
		params[it->first] = it->second;
	params["NUM_WG_CHAINS_SR"] = params["NUM_WG_CHAINS_SC"];
	params["NUM_WG_INSTANCES_SR"] = params["NUM_WG_INSTANCES_SC"];
	params["NUM_WI_CHAINS_SR"] = params["NUM_WI_CHAINS_SC"];
	params["NUM_WI_INSTANCES_SR"] = params["NUM_WI_INSTANCES_SC"];
    params["NUM_WG_LABELS_FR"] = params["NUM_WG_LABELS_FC"];
    params["NUM_WG_INSTANCES_FR"] = params["NUM_WG_INSTANCES_FC"];
    params["NUM_WI_LABELS_FR"] = params["NUM_WI_LABELS_FC"];
    params["NUM_WI_INSTANCES_FR"] = params["NUM_WI_INSTANCES_FC"];
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
        			sameResult = false;//std::cout << valOld[i*numL+l] << "|" <<  valNew[i*numL+l] << std::endl;}
        		if (abs(voteOld[i*numL + l] - voteNew[i*numL + l]) > 0.001)
				sameResult = false;//std::cout << valOld[i*numL+l] << "|" <<  valNew[i*numL+l] << std::endl;}
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
	extraParams.clear();
	auto tp_NUM_WG_CHAINS_SC = atf::tp("NUM_WG_CHAINS_SC", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WG_CHAINS_SC) { return (NUM_CHAINS % tp_NUM_WG_CHAINS_SC) == 0; });
	auto tp_NUM_WG_INSTANCES_SC = atf::tp("NUM_WG_INSTANCES_SC", atf::interval(1, numInstances),
		[&](auto tp_NUM_WG_INSTANCES_SC) { return (numInstances % tp_NUM_WG_INSTANCES_SC) == 0; });
	auto tp_NUM_WG_TREES_SC = atf::tp("NUM_WG_TREES_SC", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WG_TREES_SC) { return (NUM_TREES % tp_NUM_WG_TREES_SC) == 0; });
	auto tp_NUM_WI_CHAINS_SC = atf::tp("NUM_WI_CHAINS_SC", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WI_CHAINS_SC) { return ((NUM_CHAINS / tp_NUM_WG_CHAINS_SC) % tp_NUM_WI_CHAINS_SC) == 0; });
	auto tp_NUM_WI_INSTANCES_SC = atf::tp("NUM_WI_INSTANCES_SC", atf::interval(1, numInstances),
		[&](auto tp_NUM_WI_INSTANCES_SC) { return ((numInstances / tp_NUM_WG_INSTANCES_SC) % tp_NUM_WI_INSTANCES_SC) == 0; });
	auto tp_NUM_WI_TREES_SC = atf::tp("NUM_WI_TREES_SC", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WI_TREES_SC) { return ((NUM_TREES / tp_NUM_WG_TREES_SC) % tp_NUM_WI_TREES_SC) == 0; });
	auto tp_NUM_WI_TREES_SR = atf::tp("NUM_WI_TREES_SR", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WI_TREES_SR) { return (tp_NUM_WG_TREES_SC % tp_NUM_WI_TREES_SR) == 0; });



	extraParams["NUM_INSTANCES"] = numInstances;
	extraParams["NUM_LABELS"] = numLabels;
	extraParams["NUM_ATTRIBUTES"] = numAttributes;
	extraParams["NUM_CHAINS"] = NUM_CHAINS;
	extraParams["NUM_TREES"] = NUM_TREES;
	extraParams["MAX_LEVEL"] = MAX_LEVEL;
	extraParams["NODES_PER_TREE"] = pow(2.0f, MAX_LEVEL + 1) - 1;
	extraParams["NUM_WG_CHAINS_FC"] = extraParams["NUM_WG_INSTANCES_FC"] = extraParams["NUM_WG_LABELS_FC"] = extraParams["NUM_WI_CHAINS_FC"] = extraParams["NUM_WI_INSTANCES_FC"] = 
	extraParams["NUM_WI_LABELS_FC"] = extraParams["NUM_WI_CHAINS_FR"] = 1;

	measureStep = true; 

	/*std::map<std::string, size_t>  config;
	config["NUM_WG_CHAINS_SC"] = 2;
	config["NUM_WG_INSTANCES_SC"] = 266;
	config["NUM_WG_TREES_SC"] = 8;
	config["NUM_WI_CHAINS_SC"] = 1;
	config["NUM_WI_INSTANCES_SC"] = 1;
	config["NUM_WI_TREES_SC"] = 1;
	config["NUM_WI_TREES_SR"] = 8;*/ 
	//tune(config);
	// auto tuner = atf::exhaustive();
	auto tuner = atf::open_tuner(atf::cond::evaluations(1000));
	auto best_config = tuner(
		G(tp_NUM_WG_CHAINS_SC, tp_NUM_WI_CHAINS_SC),
		G(tp_NUM_WG_INSTANCES_SC, tp_NUM_WI_INSTANCES_SC),
		G(tp_NUM_WG_TREES_SC, tp_NUM_WI_TREES_SC, tp_NUM_WI_TREES_SR)
	)(tune);

	for (auto it = best_config.begin(); it!=best_config.end(); ++it)
                extraParams[it->first] = it->second;


        auto tp_NUM_WG_CHAINS_FC = atf::tp("NUM_WG_CHAINS_FC", atf::interval(1, NUM_CHAINS),
                [&](auto tp_NUM_WG_CHAINS_FC) { return (NUM_CHAINS % tp_NUM_WG_CHAINS_FC) == 0; });
        auto tp_NUM_WG_INSTANCES_FC = atf::tp("NUM_WG_INSTANCES_FC", atf::interval(1, numInstances),
                [&](auto tp_NUM_WG_INSTANCES_FC) { return (numInstances % tp_NUM_WG_INSTANCES_FC) == 0; });
        auto tp_NUM_WG_LABELS_FC = atf::tp("NUM_WG_LABELS_FC", atf::interval(1, numLabels),
                [&](auto tp_NUM_WG_LABELS_FC) { return (numLabels % tp_NUM_WG_LABELS_FC) == 0; });
        auto tp_NUM_WI_CHAINS_FC = atf::tp("NUM_WI_CHAINS_FC", atf::interval(1, NUM_CHAINS),
                [&](auto tp_NUM_WI_CHAINS_FC) { return ((NUM_CHAINS / tp_NUM_WG_CHAINS_FC) % tp_NUM_WI_CHAINS_FC) == 0; });
        auto tp_NUM_WI_INSTANCES_FC = atf::tp("NUM_WI_INSTANCES_FC", atf::interval(1, numInstances),
                [&](auto tp_NUM_WI_INSTANCES_FC) { return ((numInstances / tp_NUM_WG_INSTANCES_FC) % tp_NUM_WI_INSTANCES_FC) == 0; });
        auto tp_NUM_WI_LABELS_FC = atf::tp("NUM_WI_LABELS_FC", atf::interval(1, numLabels),
                [&](auto tp_NUM_WI_LABELS_FC) { return ((numLabels / tp_NUM_WG_LABELS_FC) % tp_NUM_WI_LABELS_FC) == 0; });	
        auto tp_NUM_WI_CHAINS_FR = atf::tp("NUM_WI_CHAINS_FR", atf::interval(1, NUM_CHAINS),
                [&](auto tp_NUM_WI_CHAINS_FR) { return (tp_NUM_WG_CHAINS_FC % tp_NUM_WI_CHAINS_FR) == 0; });


	measureStep = false;

	auto tuner2 = atf::open_tuner(atf::cond::evaluations(1000));
	auto best_config2 = tuner2(
		G(tp_NUM_WG_CHAINS_FC, tp_NUM_WI_CHAINS_FC, tp_NUM_WI_CHAINS_FR),
		G(tp_NUM_WG_INSTANCES_FC, tp_NUM_WI_INSTANCES_FC),
		G(tp_NUM_WG_LABELS_FC, tp_NUM_WI_LABELS_FC)
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

	std::map<std::string, size_t> dataSets;
	dataSets["data/yeast.arff"] = 14;
	dataSets["data/scene.arff"] = 6;
	dataSets["data/NNRTI.arff"] = 3;

	for(auto it=dataSets.begin(); it!=dataSets.end(); ++it)
	{
		std::cout << "Dataset: " << it->first << std::endl;
		ECCData data(it->second, it->first);
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
		delete ecc;
		delete evalData;
		valOld.clear();
		voteOld.clear();
		evalCopy.clear();
	}
	PlatformUtil::deinit();
	system("Pause");
	return 0;
}
