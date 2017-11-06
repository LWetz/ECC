#include "ECCExecutor.h" 
#include "atf_library/atf.h" 
#include <fstream>

#define NUM_CHAINS 8 
#define NUM_TREES 8 
#define MAX_LEVEL 10

ECCExecutor *ecc; 
ECCData *evalData; 
std::vector<MultilabelInstance> evalCopy; 
std::vector<double> valOld, valNew, valFixed; 
std::vector<int> voteOld, voteNew, voteFixed; 

std::ofstream outfile;

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
	size_t hitsOld = 0, hitsNew = 0, hitsFixed = 0;
        for (int i = 0; i < evalCopy.size(); ++i)
        {
        	MultilabelInstance iOrig = evalCopy[i];
        	int numL = iOrig.getNumLabels();
        	for (int l = 0; l < iOrig.getNumLabels(); ++l)
       		{
        		if (abs(valFixed[i*numL + l] - valNew[i*numL + l]) > 0.001)
        			{ sameResult = false;}// std::cout << valFixed[i*numL+l] << "|" <<  valNew[i*numL+l] << std::endl;}
        		if (abs(voteFixed[i*numL + l] - voteNew[i*numL + l]) > 0.001)
				{ sameResult = false;}// std::cout << valFixed[i*numL+l] << "|" <<  valNew[i*numL+l] << std::endl;}
			double predNew = valNew[i*numL + l] > 0 ? 1.0 : 0.0;
        		double predOld = valOld[i*numL + l] > 0 ? 1.0 : 0.0;
			double predFixed = valFixed[i*numL+l] > 0 ? 1.0 : 0.0;
        		if (predOld == iOrig.getData()[l + iOrig.getNumAttribs()])
        			hitsOld++;
        		if (predNew == iOrig.getData()[l + iOrig.getNumAttribs()])
        			hitsNew++;
			if (predFixed == iOrig.getData()[l + iOrig.getNumAttribs()])
                                hitsFixed++;
        	}
        }
	std::cout << "Time: " << ecc->getNewTime() << std::endl;
	std::cout << "Same Result: " << std::boolalpha << sameResult << std::endl;
	std::cout << "Prediction Performance: Old " << ((float)hitsOld / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0 << "% | New " << ((float)hitsNew / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0 << "% | Fixed " << ((float)hitsFixed / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0 << "%" << std::endl;
	return measureStep ? ecc->getNewTime() : ecc->getNewTotalTime();
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
	auto tuner =  atf::exhaustive();//atf::open_tuner(atf::cond::evaluations(1000));
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

	auto tuner2 = atf::exhaustive();
	auto best_config2 = tuner2(
		G(tp_NUM_WG_CHAINS_FC, tp_NUM_WI_CHAINS_FC, tp_NUM_WI_CHAINS_FR),
		G(tp_NUM_WG_INSTANCES_FC, tp_NUM_WI_INSTANCES_FC),
		G(tp_NUM_WG_LABELS_FC, tp_NUM_WI_LABELS_FC)
	)(tune);

	outfile << "Best config:" << std::endl;
	for (auto it = best_config.begin(); it!=best_config.end(); ++it)
                outfile << "    " << it->first << " = " << it->second << std::endl;

        for (auto it = best_config2.begin(); it!=best_config2.end(); ++it)
                outfile << "    " << it->first << " = " << it->second << std::endl;

	outfile << "Best time:" << ((double)tuner2.best_measured_result()) * 1e-06 << "ms" << std::endl << std::endl;

}
int main(int argc, char* argv[]) {
	std::cout << "START" << std::endl;
	
	std::string pname = "NVIDIA";
	std::string dname = "k20";
	if(argc > 1)
		dname = argv[1];
	if(argc > 2)
		pname = argv[2];

        if(!PlatformUtil::init(pname, dname))
	{
		PlatformUtil::deinit();
		return -1;
	}
        std::cout << "Platform created!" << std::endl;

	std::map<std::string, size_t> dataSets;
	dataSets["data/bibtex.arff"] = 159;
	dataSets["data/bookmarks.arff"] = 208;
	dataSets["data/CAL500.arff"] = 174;
	dataSets["data/Corel5k.arff"] = 374;
	//dataSets["data/delicious.arff"] = 983;
	dataSets["data/emotions.arff"] = 6;
	dataSets["data/enron.arff"] = 53;
	dataSets["data/flags.arff"] = 7;
	dataSets["data/genbase.arff"] = 27;
	dataSets["data/mediamill.arff"] = 101;
	dataSets["data/medical.arff"] = 45;
	dataSets["data/NNRTI.arff"] = 3;
	dataSets["data/scene.arff"] = 6;
	dataSets["data/tmc2007.arff"] = 22;
	dataSets["data/yeast.arff"] = 14;

//	dataSets.clear();
//	dataSets["data/bibtex.arff"] = 159;
	
	outfile = std::ofstream("results_"+pname+"_"+dname+".txt");
	outfile << "NUM_CHAINS = " << NUM_CHAINS << std::endl;
	outfile << "NUM_TREES = " << NUM_TREES << std::endl;
	outfile << "MAX_LEVEL = " << MAX_LEVEL << std::endl << std::endl;

	for(auto it=dataSets.begin(); it!=dataSets.end(); ++it)
	{
		std::cout << "Dataset: " << it->first << std::endl;
		outfile << "Dataset: " << it->first << std::endl;
		//try{
		ECCData data(it->second, it->first);
		outfile << "instances: " << data.getSize() << " attribute: " << data.getAttribCount() << " labels: " << data.getLabelCount() << std::endl;
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
		ecc->runClassifyOld(*evalData, valOld, voteOld, false);
		outfile << "Old time: " <<  ((double)ecc->getOldTime()) * 1e-06 << "ms" << std::endl;
                ecc->runClassifyOld(*evalData, valFixed, voteFixed, true);
                outfile << "Old time fixed: " <<  ((double)ecc->getOldTime()) * 1e-06 << "ms" << std::endl;
		numLabels = evalData->getLabelCount();
		numAttributes = evalData->getAttribCount();
		numInstances = evalData->getInstances().size();
		tuneClassify();
		delete ecc;
		delete evalData;
		valOld.clear();
		voteOld.clear();
		valFixed.clear();
		voteFixed.clear();
		evalCopy.clear();
                //}catch(std::exception e)
                //{
		//	std::cout << e.what() << std::endl;
                //        continue;
                //}

	}
	PlatformUtil::deinit();
	system("Pause");
	return 0;
}
