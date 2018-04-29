#ifndef _WIN32
	#include "Tuner.hpp"
#endif
#include "ECCExecutorNew.hpp"
#include "ECCExecutorOld.hpp" 
#include "PredictionPerformance.hpp"
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

size_t getIntegerCmdOption(char ** begin, char ** end, size_t defaultVal, const std::string & option)
{
	if (char* cmd = getCmdOption(begin, end, option))
	{
		try {
			return std::stoul(cmd);
		}
		catch (...)
		{
			std::cout << option << " has to be integer" << std::endl;
			throw;
		}
	}

	return defaultVal;
}

size_t calcTreesPerRun(size_t nodeLimit, size_t totalTrees, size_t nodesPerTree)
{
	size_t treeLimit = nodeLimit / nodesPerTree;

	treeLimit = treeLimit > 0 ? treeLimit : 1;

	if (treeLimit >= totalTrees)
		return totalTrees;

	while (totalTrees % treeLimit != 0)
		--treeLimit;

	return treeLimit;
}


std::string makeFileName(const char* prefix, const char* dataset, const char* pname, size_t maxLevel, size_t numChains, size_t numTrees)
{
	std::string datasetstr(dataset);

	size_t dot = datasetstr.find_last_of(".");
	size_t slash = datasetstr.find_last_of("/\\");

	slash = slash != std::string::npos ? slash : 0;
	dot = dot != std::string::npos ? dot : std::string::npos;

	datasetstr = datasetstr.substr(slash + 1, dot - slash - 1);

	std::stringstream fileName;
	fileName << prefix << pname << "_" << datasetstr << "_" << maxLevel << "_" << numTrees << "_" << numChains << ".txt";
	return fileName.str();
}

template<typename T>
void makeKeyValFile(T keyval, std::string fileName)
{
	std::ofstream file(fileName);
	for (auto it = keyval.begin(); it != keyval.end(); ++it)
	{
		file << it->first << "=" << it->second << std::endl;
	}
	file.close();
}

template<typename T>
T readKeyValFile(std::string fileName)
{
	std::ifstream file(fileName);
	T keyval;

	if (!file.is_open())
	{
		std::cout << "No Key-Value file found, tune before measuring" << std::endl;
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

			keyval[line.substr(0, split)] = std::stoi(line.substr(split+1));
		}
	}
	catch (...)
	{
		std::cout << "Couldnt parse Key-Value file" << std::endl;
		file.close();
		exit(-6);
	}

	file.close();
	return keyval;
}

template<typename T>
void updateKeyValFile(T keyval, std::string fileName)
{
	auto oldConf = readKeyValFile<T>(fileName);
	for (auto it = keyval.begin(); it != keyval.end(); ++it)
	{
		oldConf[it->first] = it->second;
	}

	makeKeyValFile<T>(oldConf, fileName);
}

int main(int argc, char* argv[]) {
	if (argc < 2) { std::cout << "First argument has to be 'tuneBuild', 'tuneStep', 'tuneFinal', 'measure', 'measureold' or 'measureoldorig'" << std::endl; return -1; }

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
	size_t numLabels;

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
	size_t numAttributes;

	try {
		ECCData data(numLabels, dataset);
		size_t trainSize = 0.67 * data.getSize();
		size_t evalSize = data.getSize() - trainSize;
		inputCopy = data.getInstances();
		trainInstances.reserve(trainSize);
		evalInstances.reserve(evalSize);
		for (size_t i = 0; i < trainSize; ++i)
		{
			int idx = Util::randomInt(inputCopy.size());
			trainInstances.push_back(inputCopy[idx]);
			inputCopy.erase(inputCopy.begin() + idx);
		}
		for (size_t i = 0; i < evalSize; ++i)
		{
			int idx = Util::randomInt(inputCopy.size());
			MultilabelInstance inst = inputCopy[idx];
			evalOriginal.push_back(inst);
			for (size_t i = inst.getNumAttribs(); i < inst.getValueCount(); ++i)
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

	size_t maxLevel;
	size_t numTrees;
	size_t numChains;
	size_t ensembleSubSetSize;
	size_t forestSubSetSize;
	size_t nodeLimit;

	size_t totalTrees;
	size_t nodesPerTree;

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

	size_t treesPerRun = calcTreesPerRun(nodeLimit, totalTrees, nodesPerTree);

	std::string configFileName = makeFileName("config_", dataset, pname, maxLevel, numChains, numTrees);
	std::string measureFileName = makeFileName("measure_", dataset, pname, maxLevel, numChains, numTrees);
	std::string oldMeasureFileName = makeFileName("oldmeasure_", dataset, pname, maxLevel, numChains, numTrees);
	std::string oldMeasureOrigFileName = makeFileName("oldmeasureorig_", dataset, pname, maxLevel, numChains, numTrees);

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

	if (std::string(argv[1]).compare("tuneBuild") == 0)
	{
#ifndef _WIN32
		ECCTuner tuner(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);
		auto config = tuner.tuneBuild(trainData, treesPerRun);
		makeKeyValFile<Configuration>(config, configFileName);
#endif
	}
	else if (std::string(argv[1]).compare("tuneStep") == 0)
	{
#ifndef _WIN32
		ECCTuner tuner(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);
		auto config = readKeyValFile<Configuration>(configFileName);
		config = tuner.tuneClassifyStep(trainData, treesPerRun, evalData, config);
		updateKeyValFile<Configuration>(config, configFileName);
#endif
	}
	else if (std::string(argv[1]).compare("tuneFinal") == 0)
	{
#ifndef _WIN32
		ECCTuner tuner(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);
		auto config = readKeyValFile<Configuration>(configFileName);
		config = tuner.tuneClassifyFinal(trainData, treesPerRun, evalData, config);
		updateKeyValFile<Configuration>(config, configFileName);
#endif
	}
	else if (std::string(argv[1]).compare("measure") == 0)
	{
		auto config = readKeyValFile<Configuration>(configFileName);
		ECCExecutorNew eccEx(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);
		eccEx.runBuild(trainData, treesPerRun, config["NUM_WI"], config["NUM_WG"]);
		auto predictions = eccEx.runClassify(evalData, config);

		Measurement performance = eccEx.getMeasurement();
		std::for_each(performance.begin(), performance.end(), [](auto &d) { std::cout << d.first << ": " << d.second << std::endl; d.second *= 1e-06; });

		PredictionPerformance predictionPerformance(numLabels, evalOriginal.size(), 0.5f);
		Measurement evaluation = predictionPerformance.calculatePerfomance(evalOriginal, predictions);
		performance.insert(evaluation.begin(), evaluation.end());

		makeKeyValFile<Measurement>(performance, measureFileName);
	}
	else if (std::string(argv[1]).compare("measureold") == 0)
	{
		ECCExecutorOld eccEx(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);		
                eccEx.runBuild(trainData, treesPerRun);
                auto predictions = eccEx.runClassify(evalData);
                Measurement performance = eccEx.getMeasurement();

                std::for_each(performance.begin(), performance.end(), [](auto &d) { std::cout << d.first << ": " << d.second << std::endl; d.second *= 1e-06; });

		PredictionPerformance predictionPerformance(numLabels, evalOriginal.size(), 0.5f);
		Measurement evaluation = predictionPerformance.calculatePerfomance(evalOriginal, predictions);
		performance.insert(evaluation.begin(), evaluation.end());

		makeKeyValFile<Measurement>(performance, oldMeasureFileName);
	}
	else if (std::string(argv[1]).compare("measureoldorig") == 0)
        {
                ECCExecutorOld eccEx(maxLevel, numAttributes, numAttributes, numTrees, numLabels, numChains, ensembleSubSetSize, forestSubSetSize);
               	eccEx.runBuild(trainData, treesPerRun);
          	auto predictions = eccEx.runClassify(evalData, false);
                Measurement performance = eccEx.getMeasurement();
		
		std::for_each(performance.begin(), performance.end(), [](auto &d) { std::cout << d.first << ": " << d.second << std::endl; d.second *= 1e-06; });

                PredictionPerformance predictionPerformance(numLabels, evalOriginal.size(), 0.5f);
                Measurement evaluation = predictionPerformance.calculatePerfomance(evalOriginal, predictions);
                performance.insert(evaluation.begin(), evaluation.end());

                makeKeyValFile<Measurement>(performance, oldMeasureOrigFileName);
        }


	PlatformUtil::deinit();
	return 0;
}
