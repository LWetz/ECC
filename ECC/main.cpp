#ifndef _WIN32
	#include "Tuner.hpp"
#endif
#include "ECCExecutorNew.hpp"
#include "ECCExecutorOld.hpp" 
#include <fstream>

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
	char ** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

int getIntegerCmdOption(char ** begin, char ** end, int defaultVal, const std::string & option)
{
	if (char* cmd = getCmdOption(begin, end, option))
	{
		try {
			return std::stoi(cmd);
		}
		catch (...)
		{
			std::cout << option << " has to be integer" << std::endl;
			throw;
		}
	}

	return defaultVal;
}

int calcTreesPerRun(int nodeLimit, int totalTrees, int nodesPerTree)
{
	int treeLimit = nodeLimit / nodesPerTree;

	treeLimit = treeLimit > 0 ? treeLimit : 1;

	if (treeLimit >= totalTrees)
		return totalTrees;

	while (totalTrees % treeLimit != 0)
		treeLimit--;

	return treeLimit;
}

std::string makeFileName(const char* dataset, const char* pname, int maxLevel, int numChains, int numTrees)
{
	std::string datasetstr(dataset);

	size_t dot = datasetstr.find_last_of(".");
	size_t slash = datasetstr.find_last_of("/\\");

	slash = slash != std::string::npos ? slash : 0;
	dot = dot != std::string::npos ? dot : std::string::npos;

	datasetstr = datasetstr.substr(slash + 1, dot - slash - 1);

	std::stringstream fileName;
	fileName << "config_" << pname << "_" << datasetstr << "_" << maxLevel << "_" << numTrees << "_" << numChains << ".txt";
	return fileName.str();
}

void writeConfigFile(std::map<std::string, int> config, std::string fileName)
{
	std::ofstream file(fileName);
	for (auto it = config.begin(); it != config.end(); ++it)
	{
		file << it->first << "=" << it->second << std::endl;
	}
	file.close();
}

std::map<std::string, int> readConfigFile(std::string fileName)
{
	std::ifstream file(fileName);
	std::map<std::string, int> config;

	if (!file.is_open())
	{
		std::cout << "No config file found, tune before measuring" << std::endl;
		exit(-5);
	}

	try {
		std::string line;
		while (std::getline(file, line))
		{
			size_t split = line.find('=');
			if (split == std::string::npos)
			{
				throw;
			}

			config[line.substr(0, split)] = std::stoi(line.substr(split+1));
		}
	}
	catch (...)
	{
		std::cout << "Couldnt parse config file" << std::endl;
		exit(-6);
	}

	return config;
}

int main(int argc, char* argv[]) {
	if (argc < 2) { std::cout << "First argument has to be 'tune', 'measure' or 'measureold'" << std::endl; return -1; }

	const char* pname = getCmdOption(argv + 2, argv + argc, "-platform");
	const char* dname = getCmdOption(argv + 2, argv + argc, "-device");

	pname = pname ? pname : "NVIDIA";
	dname = dname ? dname : "Tesla K20m";

	if (!PlatformUtil::init(pname, dname))
	{
		PlatformUtil::deinit();
		return -2;
	}

	char* dataset = getCmdOption(argv + 2, argv + argc, "-d");
	char* labelcount = getCmdOption(argv + 2, argv + argc, "-l");
	int numLabels;

	if(!dataset && !labelcount)
	{
		std::cout << "Specify dataset with -d and labelcount with -l" << std::endl;
		return -3;
	}

	try { 
		numLabels = std::stoi(labelcount);
	}
	catch (...)
	{
		std::cout << "Label count has to be integer" << std::endl;
		return -4;
	}

	std::cout << "Preparing dataset" << std::endl;

	std::vector<MultilabelInstance> inputCopy;
	std::vector<MultilabelInstance> trainInstances;
	std::vector<MultilabelInstance> evalInstances;
	std::vector<MultilabelInstance> evalOriginal;
	int numAttributes;

	try {
		ECCData data(numLabels, dataset);
		int trainSize = 0.67 * data.getSize();
		int evalSize = data.getSize() - trainSize;
		inputCopy = data.getInstances();
		trainInstances.reserve(trainSize);
		evalInstances.reserve(evalSize);
		Util::RANDOM.setSeed(10101001);
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
			evalOriginal.push_back(inst);
			for (int i = inst.getNumAttribs(); i < inst.getValueCount(); ++i)
			{
				inst.getData()[i] = 0.0;
			}
			evalInstances.push_back(inst);
			inputCopy.erase(inputCopy.begin() + idx);
		}
		numAttributes = data.getAttribCount();
	}
	catch (...)
	{
		std::cout << "Error preparing dataset" << std::endl;
		return -7;
	}

	ECCData trainData(trainInstances, numAttributes, numLabels);
	ECCData evalData(evalInstances, numAttributes, numLabels);

	int maxLevel;
	int numTrees;
	int numChains;
	int ensembleSubSetSize;
	int forestSubSetSize;
	int nodeLimit;

	int totalTrees;
	int nodesPerTree;

	try {
		maxLevel = getIntegerCmdOption(argv + 2, argv + argc, 10, "-depth");
		nodesPerTree = (1 << (maxLevel + 1)) - 1;
		numTrees = getIntegerCmdOption(argv + 2, argv + argc, 32, "-t");
		numChains = getIntegerCmdOption(argv + 2, argv + argc, 64, "-c");
		totalTrees = numLabels * numChains * numTrees;
		ensembleSubSetSize = getIntegerCmdOption(argv + 2, argv + argc, 100, "-ie");
		forestSubSetSize = getIntegerCmdOption(argv + 2, argv + argc, 50, "-if");
		nodeLimit = getIntegerCmdOption(argv + 2, argv + argc, totalTrees * nodesPerTree, "-nl");
	}
	catch (...)
	{
		return -3;
	}

	int treesPerRun = calcTreesPerRun(nodeLimit, totalTrees, nodesPerTree);

	std::cout << "Platform: " << pname << std::endl;
	std::cout << "Device: " << dname << std::endl;
	std::cout << "Dataset: " << dataset << std::endl;
	std::cout << "NUM_LABELS: " << numLabels << std::endl;
	std::cout << "NUM_ATTRIBUTES: " << numAttributes << std::endl;
	std::cout << "MAX_LEVEL: " << maxLevel << std::endl;
	std::cout << "NUM_TREES: " << numTrees << std::endl;
	std::cout << "NUM_CHAINS: " << numChains << std::endl;
	std::cout << "TOTAL_TREES: " << totalTrees << std::endl;
	std::cout << "ENSEMBLE_SUBSET: " << ensembleSubSetSize << std::endl;
	std::cout << "FOREST_SUBSET: " << forestSubSetSize << std::endl;
	std::cout << "TREES_PER_RUN: " << treesPerRun << std::endl;

	if (std::string(argv[1]).compare("tune") == 0)
	{
#ifndef _WIN32
		ECCTuner tuner(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);
		tuner.tune(trainData, treesPerRun, evalData);
		
		auto config = tuner.getBestBuildConfig();
		auto stepConfig = tuner.getBestStepConfig();
		auto finalConfig = tuner.getBestFinalConfig();

		config.insert(stepConfig.begin(), stepConfig.end());
		config.insert(finalConfig.begin(), finalConfig.end());

		writeConfigFile(config, makeFileName(dataset, pname, maxLevel, numChains, numTrees));
#endif
				ECCExecutorNew eccEx(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);
		eccEx.prepareBuild(trainData, treesPerRun);
		for (int wi = 1; wi < treesPerRun; ++wi)
		{
			for (int wg = 1; wg < treesPerRun; ++wg)
			{
				if ((treesPerRun % wg) != 0 || ((treesPerRun / wg) % wi) != 0)
					continue;

				std::cout << "WG=" << wg << " WI=" << wi << " => ";
				try {
					std::cout << eccEx.tuneBuild(wi, wg)*1e-09 << std::endl;
				}
				catch (...)
				{
					std::cout << "ERROR" << std::endl;
				}
			}
		}
		eccEx.finishBuild();
		system("Pause");
	}
	else if (std::string(argv[1]).compare("measure") == 0)
	{
		auto config = readConfigFile(makeFileName(dataset, pname, maxLevel, numChains, numTrees));
		std::vector<double> values;
		std::vector<int> votes;
		ECCExecutorNew eccEx(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);
		eccEx.runBuild(trainData, treesPerRun, config["NUM_WI"], config["NUM_WG"]);
		eccEx.runClassifyNew(trainData, values, votes, config);
	}
	else if (std::string(argv[1]).compare("measureold") == 0)
	{
		std::vector<double> values;
		std::vector<int> votes;
		ECCExecutorOld eccEx(maxLevel, numAttributes, numTrees);
		eccEx.runBuild(trainData, 1, numChains, 1, ensembleSubSetSize, forestSubSetSize);
		eccEx.runClassifyOld(evalData, values, votes);
	}

	PlatformUtil::deinit();
	return 0;
}
