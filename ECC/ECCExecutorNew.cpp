#pragma once

#include "ECCExecutorNew.hpp"

std::vector<int> ECCExecutorNew::partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc)
{
	std::vector<int> indicesList;
	if (ecc.getEnsembleSubSetSize() != data.getSize() || ecc.getForestSubSetSize() != data.getSize())
	{
		indicesList = ecc.partitionInstanceIndices(data.getSize());
	}
	else
	{
		for (int chain = 0; chain < ecc.getEnsembleSize(); ++chain)
		{
			for (int forest = 0; forest < ecc.getChainSize(); ++forest)
			{
				for (int tree = 0; tree < ecc.getForestSize(); ++tree)
				{
					for (int index = 0; index < data.getSize(); ++index)
					{
						indicesList.push_back(index);
					}
				}
			}
		}
	}
	return indicesList;
}

ECCExecutorNew::ECCExecutorNew(int _maxLevel, int _maxAttributes, int _numAttributes, int _numTrees, int _numLabels, int _numChains, int _ensembleSubSetSize, int _forestSubSetSize)
	: nodeValues(NULL), nodeIndices(NULL), labelOrderBuffer(NULL), maxLevel(_maxLevel),
	numTrees(_numTrees), maxAttributes(_maxAttributes), numLabels(_numLabels),
	numChains(_numChains), numAttributes(_numAttributes),
	buildSource(Util::loadFileToString("eccBuildNew.cl")),
	stepCalcSource(Util::loadFileToString("stepCalcKernel.cl")),
	stepReduceSource(Util::loadFileToString("stepReduceKernel.cl")),
	finalCalcSource(Util::loadFileToString("finalCalcKernel.cl")),
	finalReduceSource(Util::loadFileToString("finalReduceKernel.cl"))
{
	Util::RANDOM.setSeed(133713);
	Util::StopWatch stopWatch;
	stopWatch.start();
	ecc = new EnsembleOfClassifierChains(_numAttributes + numLabels, numLabels, maxLevel, numTrees, numChains, _ensembleSubSetSize, _forestSubSetSize);

	labelOrderBuffer = Buffer(sizeof(int) * ecc->getEnsembleSize() * ecc->getChainSize(), CL_MEM_READ_ONLY);
	int* labelOrders = new int[ecc->getEnsembleSize() * ecc->getChainSize()];

	int i = 0;
	for (int chain = 0; chain < ecc->getEnsembleSize(); ++chain)
	{
		for (int forest = 0; forest < ecc->getChainSize(); ++forest)
		{
			labelOrders[i++] = ecc->getChains()[chain].getLabelOrder()[forest];
		}
	}
	labelOrderBuffer.writeFrom(labelOrders, labelOrderBuffer.getSize());
	delete[] labelOrders;

	nodeValues = new double[ecc->getTotalSize()];
	nodeIndices = new int[ecc->getTotalSize()];

	measurement["SetupTotalTime"] = stopWatch.stop();
	measurement["SetupLabelOrdersWrite"] = labelOrderBuffer.getTransferTime();
}

void ECCExecutorNew::prepareBuild(ECCData& data, int treesPerRun)
{
	std::cout << std::endl << "--- BUILD ---" << std::endl;

	buildData = new BuildData;

	int nodesLastLevel = pow(2.0f, maxLevel);
	int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;

	int maxSplits = ecc->getForestSubSetSize() - 1;

	Buffer tmpNodeValueBuffer(sizeof(double) * treesPerRun * nodesPerTree, CL_MEM_READ_WRITE);
	Buffer tmpNodeIndexBuffer(sizeof(int) * treesPerRun * nodesPerTree, CL_MEM_READ_WRITE);

	Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

	double* dataArray = new double[data.getValueCount() * data.getSize()];
	int dataBuffIdx = 0;
	for (MultilabelInstance inst : data.getInstances())
	{
		memcpy(dataArray + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
		dataBuffIdx += inst.getValueCount();
	}
	dataBuffer.writeFrom(dataArray, dataBuffer.getSize());
	delete[] dataArray;

	int numValues = data.getValueCount();
	int numAttributes = data.getAttribCount();

	Buffer instancesBuffer(sizeof(int) * maxSplits*treesPerRun, CL_MEM_READ_WRITE);

	std::vector<int> indicesList(partitionInstances(data, *ecc));

	Buffer instancesNextBuffer(sizeof(int) * maxSplits*treesPerRun, CL_MEM_READ_WRITE);
	Buffer instancesLengthBuffer(sizeof(int) * nodesLastLevel*treesPerRun, CL_MEM_READ_WRITE);
	Buffer instancesNextLengthBuffer(sizeof(int) * nodesLastLevel*treesPerRun, CL_MEM_READ_WRITE);
	Buffer seedsBuffer(sizeof(int) * treesPerRun, CL_MEM_READ_ONLY);

	double totalTime = .0;

	int* seeds = new int[treesPerRun];
	for (int seed = 0; seed < treesPerRun; ++seed)
	{
		seeds[seed] = Util::randomInt(INT_MAX);
	}
	seedsBuffer.writeFrom(seeds, seedsBuffer.getSize());
	delete[] seeds;

	instancesBuffer.writeFrom(indicesList.data(), treesPerRun * maxSplits * sizeof(int));

	buildData->tmpNodeIndexBuffer = tmpNodeIndexBuffer;
	buildData->tmpNodeValueBuffer = tmpNodeValueBuffer;
	buildData->dataBuffer = dataBuffer;
	buildData->instancesBuffer = instancesBuffer;
	buildData->instancesLengthBuffer = instancesLengthBuffer;
	buildData->instancesNextBuffer = instancesNextBuffer;
	buildData->instancesNextLengthBuffer = instancesNextLengthBuffer;
	buildData->seedsBuffer = seedsBuffer;
	buildData->numTrees = treesPerRun;
	buildData->numInstances = data.getSize();
}

double ECCExecutorNew::tuneBuild(int workitems, int workgroups)
{
	Kernel* buildKernel = NULL;

	try {
		int nodesLastLevel = pow(2.0f, maxLevel);
		int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;
		int maxSplits = ecc->getForestSubSetSize() - 1;

		Configuration params;
		params["NUM_CHAINS"] = numChains;
		params["NUM_VALUES"] = numAttributes + numLabels;
		params["NUM_ATTRIBUTES"] = numAttributes;
		params["NUM_TREES"] = numTrees;
		params["TOTAL_TREES"] = buildData->numTrees;
		params["MAX_LEVEL"] = maxLevel;
		params["NODES_LAST_LEVEL"] = nodesLastLevel;
		params["NODES_PER_TREE"] = nodesPerTree;
		params["NUM_INSTANCES"] = buildData->numInstances;
		params["MAX_SPLITS"] = maxSplits;
		params["NUM_LABELS"] = numLabels;
		params["MAX_ATTRIBUTES"] = maxAttributes;
		params["NUM_WI"] = workitems;
		params["NUM_WG"] = workgroups;

		std::string optionString;
		std::stringstream strstr;
		for (auto it = params.begin(); it != params.end(); ++it)
			strstr << " -D " << it->first << "=" << it->second;
		optionString = strstr.str();

		cl_program prog;
		PlatformUtil::buildProgramFromSource(buildSource, prog, optionString);//("\\\\X-THINK\\Users\\Public\\eccBuild.cl", prog);
		buildKernel = new Kernel(prog, "eccBuild");
		clReleaseProgram(prog);

		buildKernel->setDim(1);
		buildKernel->setGlobalSize(workgroups * workitems);
		buildKernel->setLocalSize(workitems);

		buildKernel->SetArg(0, 0);
		buildKernel->SetArg(1, buildData->seedsBuffer);
		buildKernel->SetArg(2, buildData->dataBuffer);
		buildKernel->SetArg(3, labelOrderBuffer);
		buildKernel->SetArg(4, buildData->instancesBuffer);
		buildKernel->SetArg(5, buildData->instancesNextBuffer);
		buildKernel->SetArg(6, buildData->instancesLengthBuffer);
		buildKernel->SetArg(7, buildData->instancesNextLengthBuffer);
		buildKernel->SetArg(8, buildData->tmpNodeValueBuffer);
		buildKernel->SetArg(9, buildData->tmpNodeIndexBuffer);

		buildKernel->execute();
		return buildKernel->getRuntime();
	}
	catch (...)
	{
		delete buildKernel;
		throw;
	}
}

void ECCExecutorNew::finishBuild()
{
	buildData->tmpNodeIndexBuffer.clear();
	buildData->tmpNodeValueBuffer.clear();
	buildData->dataBuffer.clear();
	buildData->instancesBuffer.clear();
	buildData->instancesLengthBuffer.clear();
	buildData->instancesNextBuffer.clear();
	buildData->instancesNextLengthBuffer.clear();
	buildData->seedsBuffer.clear();

	delete buildData;
}

void ECCExecutorNew::runBuild(ECCData& data, int treesPerRun, int workitems, int workgroups)
{
	std::cout << std::endl << "--- BUILD ---" << std::endl;

	Util::StopWatch totalBuildTime;
	totalBuildTime.start();

	int nodesLastLevel = pow(2.0f, maxLevel);
	int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;

	int maxSplits = ecc->getForestSubSetSize() - 1;

	Configuration params;
	params["NUM_CHAINS"] = numChains;
	params["NUM_VALUES"] = numAttributes + numLabels;
	params["NUM_ATTRIBUTES"] = numAttributes;
	params["NUM_TREES"] = numTrees;
	params["TOTAL_TREES"] = treesPerRun;
	params["MAX_LEVEL"] = maxLevel;
	params["NODES_LAST_LEVEL"] = nodesLastLevel;
	params["NODES_PER_TREE"] = nodesPerTree;
	params["NUM_INSTANCES"] = data.getSize();
	params["MAX_SPLITS"] = maxSplits;
	params["NUM_LABELS"] = numLabels;
	params["MAX_ATTRIBUTES"] = maxAttributes;
	params["NUM_WI"] = workitems;
	params["NUM_WG"] = workgroups;

	Util::StopWatch buildCompileTime;
	buildCompileTime.start();

	std::string optionString;
	std::stringstream strstr;
	for (auto it = params.begin(); it != params.end(); ++it)
		strstr << " -D " << it->first << "=" << it->second;
	optionString = strstr.str();

	measurement["buildCompileTime"] = buildCompileTime.stop();

	cl_program prog;
	PlatformUtil::buildProgramFromSource(buildSource, prog, optionString);
	Kernel* buildKernel = new Kernel(prog, "eccBuild");
	clReleaseProgram(prog);

	Buffer tmpNodeValueBuffer(sizeof(double) * treesPerRun * nodesPerTree, CL_MEM_READ_WRITE);
	Buffer tmpNodeIndexBuffer(sizeof(int) * treesPerRun * nodesPerTree, CL_MEM_READ_WRITE);

	Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

	double* dataArray = new double[data.getValueCount() * data.getSize()];
	int dataBuffIdx = 0;
	for (MultilabelInstance inst : data.getInstances())
	{
		memcpy(dataArray + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
		dataBuffIdx += inst.getValueCount();
	}
	dataBuffer.writeFrom(dataArray, dataBuffer.getSize());
	delete[] dataArray;

	int numValues = data.getValueCount();
	int numAttributes = data.getAttribCount();

	std::vector<int> indicesList(partitionInstances(data, *ecc));

	Buffer instancesBuffer(sizeof(int) * maxSplits*treesPerRun, CL_MEM_READ_WRITE);
	Buffer instancesNextBuffer(sizeof(int) * maxSplits*treesPerRun, CL_MEM_READ_WRITE);
	Buffer instancesLengthBuffer(sizeof(int) * nodesLastLevel*treesPerRun, CL_MEM_READ_WRITE);
	Buffer instancesNextLengthBuffer(sizeof(int) * nodesLastLevel*treesPerRun, CL_MEM_READ_WRITE);
	Buffer seedsBuffer(sizeof(int) * treesPerRun, CL_MEM_READ_ONLY);

	int gidMultiplier = 0;

	buildKernel->setDim(1);
	buildKernel->setGlobalSize(workgroups * workitems);
	buildKernel->setLocalSize(workitems);

	buildKernel->SetArg(1, seedsBuffer);
	buildKernel->SetArg(2, dataBuffer);
	buildKernel->SetArg(3, labelOrderBuffer);
	buildKernel->SetArg(4, instancesBuffer);
	buildKernel->SetArg(5, instancesNextBuffer);
	buildKernel->SetArg(6, instancesLengthBuffer);
	buildKernel->SetArg(7, instancesNextLengthBuffer);
	buildKernel->SetArg(8, tmpNodeValueBuffer);
	buildKernel->SetArg(9, tmpNodeIndexBuffer);

	measurement["buildKernel"] = 0;
	measurement["buildSeedsWrite"] = 0;
	measurement["buildInstancesWrite"] = 0;
	measurement["buildNodeIndexRead"] = 0;
	measurement["buildNodeValueRead"] = 0;

	Util::StopWatch buildLoopTime;
	buildLoopTime.start();

	int* seeds = new int[treesPerRun];
	for (int tree = 0; tree < numChains * numTrees * numLabels; tree += treesPerRun)
	{
		buildKernel->SetArg(0, gidMultiplier * treesPerRun);

		for (int seed = 0; seed < treesPerRun; ++seed)
		{
			seeds[seed] = Util::randomInt(INT_MAX);
		}
		seedsBuffer.writeFrom(seeds, seedsBuffer.getSize());

		instancesBuffer.writeFrom(indicesList.data() + gidMultiplier * treesPerRun * maxSplits, treesPerRun * maxSplits * sizeof(int));

		buildKernel->execute();

		tmpNodeIndexBuffer.readTo(nodeIndices + gidMultiplier*treesPerRun*nodesPerTree, treesPerRun*nodesPerTree * sizeof(int));
		tmpNodeValueBuffer.readTo(nodeValues + gidMultiplier*treesPerRun*nodesPerTree, treesPerRun*nodesPerTree * sizeof(double));
		
		++gidMultiplier;
		measurement["buildKernel"] += buildKernel->getRuntime();
		measurement["buildSeedsWrite"] += seedsBuffer.getTransferTime();
		measurement["buildInstancesWrite"] += instancesBuffer.getTransferTime();
		measurement["buildNodeIndexRead"] += tmpNodeIndexBuffer.getTransferTime();
		measurement["buildNodeValueRead"] += tmpNodeValueBuffer.getTransferTime();
	}
	measurement["buildLoopTime"] = buildLoopTime.stop();
	measurement["buildTotalTime"] = totalBuildTime.stop();
	measurement["buildDataWrite"] = dataBuffer.getTransferTime();

	delete[] seeds;
	tmpNodeIndexBuffer.clear();
	tmpNodeValueBuffer.clear();
	dataBuffer.clear();
	instancesBuffer.clear();
	instancesLengthBuffer.clear();
	instancesNextBuffer.clear();
	instancesNextLengthBuffer.clear();
	seedsBuffer.clear();

	int* nodeIndicesTransp = new int[ecc->getTotalSize()];
	double* nodeValuesTransp = new double[ecc->getTotalSize()];
	size_t forestDataSize = numTrees * nodesPerTree;
	for (int chain = 0; chain < ecc->getEnsembleSize(); ++chain)
	{
		for (int forest = 0; forest < ecc->getChainSize(); ++forest)
		{
			memcpy(nodeIndicesTransp + (forest * numChains + chain) * forestDataSize,
				nodeIndices + (chain * numLabels + forest) * forestDataSize,
				forestDataSize * sizeof(int));

			memcpy(nodeValuesTransp + (forest * numChains + chain) * forestDataSize,
				nodeValues + (chain * numLabels + forest) * forestDataSize,
				forestDataSize * sizeof(double));
		}
	}

	delete[] nodeValues;
	delete[] nodeIndices;
	nodeValues = nodeValuesTransp;
	nodeIndices = nodeIndicesTransp;

	delete buildKernel;
}

void ECCExecutorNew::prepareClassify(ECCData& data)
{
	std::cout << std::endl << "--- NEW CLASSIFICATION ---" << std::endl;

	classifyData = new ClassifyData;

	Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

	double* dataArray = new double[data.getValueCount() * data.getSize()];
	int dataBuffIdx = 0;
	for (MultilabelInstance inst : data.getInstances())
	{
		memcpy(dataArray + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
		dataBuffIdx += inst.getValueCount();
	}
	dataBuffer.writeFrom(dataArray, dataBuffer.getSize());
	delete[] dataArray;

	Buffer resultBuffer(data.getSize() * data.getLabelCount() * sizeof(double), CL_MEM_WRITE_ONLY);
	Buffer labelBuffer(data.getSize() * data.getLabelCount() * numChains * sizeof(double), CL_MEM_WRITE_ONLY);

	size_t stepModelSize = ecc->getTotalSize() / numLabels;
	Buffer stepNodeValueBuffer(sizeof(double) * stepModelSize, CL_MEM_READ_ONLY);
	Buffer stepNodeIndexBuffer(sizeof(int) * stepModelSize, CL_MEM_READ_ONLY);

	classifyData->dataBuffer = dataBuffer;
	classifyData->resultBuffer = resultBuffer;
	classifyData->labelBuffer = labelBuffer;
	classifyData->stepNodeValueBuffer = stepNodeValueBuffer;
	classifyData->stepNodeIndexBuffer = stepNodeIndexBuffer;
	classifyData->numInstances = data.getSize();
}

double ECCExecutorNew::tuneClassifyStep(Configuration config, int oneStep)
{
	Buffer stepIntermediateBuffer;
	Kernel* stepCalcKernel = NULL;
	Kernel* stepReduceKernel = NULL;
	try {
		config["NUM_INSTANCES"] = classifyData->numInstances;
		config["NUM_LABELS"] = numLabels;
		config["NUM_ATTRIBUTES"] = numAttributes;
		config["NUM_CHAINS"] = numChains;
		config["NUM_TREES"] = numTrees;
		config["MAX_LEVEL"] = maxLevel;
		config["NODES_PER_TREE"] = pow(2.0f, maxLevel + 1) - 1;
		config["NUM_WI_INSTANCES_SR"] = config["NUM_WI_INSTANCES_SC"];
		config["NUM_WG_INSTANCES_SR"] = config["NUM_WG_INSTANCES_SC"];
		config["NUM_WI_CHAINS_SR"] = config["NUM_WI_CHAINS_SC"];
		config["NUM_WG_CHAINS_SR"] = config["NUM_WG_CHAINS_SC"];

		std::string optionString;
		std::stringstream strstr;
		for (auto it = config.begin(); it != config.end(); ++it)
			strstr << " -D " << it->first << "=" << it->second;
		optionString = strstr.str();

		cl_program prog;
		PlatformUtil::buildProgramFromSource(stepCalcSource, prog, optionString.c_str());
		stepCalcKernel = new Kernel(prog, "stepCalc");
		clReleaseProgram(prog);

		PlatformUtil::buildProgramFromSource(stepReduceSource, prog, optionString.c_str());
		stepReduceKernel = new Kernel(prog, "stepReduce");
		clReleaseProgram(prog);

		int stepIntermediateBufferSize[3] = { classifyData->numInstances, numChains, config["NUM_WG_TREES_SC"] };
		int stepIntermediateBufferTotalSize = stepIntermediateBufferSize[0] * stepIntermediateBufferSize[1] * stepIntermediateBufferSize[2];

		int localBufferSize_SC = config["NUM_WI_INSTANCES_SC"] * config["NUM_WI_CHAINS_SC"] * config["NUM_WI_TREES_SC"];
		int localBufferSize_SR = config["NUM_WI_INSTANCES_SR"] * config["NUM_WI_CHAINS_SR"] * config["NUM_WI_TREES_SR"];

		stepIntermediateBuffer = Buffer(stepIntermediateBufferTotalSize * sizeof(TreeVote), CL_MEM_READ_WRITE);

		stepCalcKernel->SetArg(0, classifyData->stepNodeValueBuffer);
		stepCalcKernel->SetArg(1, classifyData->stepNodeIndexBuffer);
		stepCalcKernel->SetArg(2, classifyData->dataBuffer);
		stepCalcKernel->SetArg(3, classifyData->labelBuffer);
		stepCalcKernel->SetLocalArg(4, localBufferSize_SC * sizeof(TreeVote));
		stepCalcKernel->SetArg(5, stepIntermediateBuffer);

		stepCalcKernel->setDim(3);
		stepCalcKernel->setGlobalSize(config["NUM_WG_INSTANCES_SC"] * config["NUM_WI_INSTANCES_SC"], config["NUM_WG_CHAINS_SC"] * config["NUM_WI_CHAINS_SC"], config["NUM_WG_TREES_SC"] * config["NUM_WI_TREES_SC"]);
		stepCalcKernel->setLocalSize(config["NUM_WI_INSTANCES_SC"], config["NUM_WI_CHAINS_SC"], config["NUM_WI_TREES_SC"]);

		stepReduceKernel->SetArg(0, stepIntermediateBuffer);
		stepReduceKernel->SetArg(1, classifyData->labelBuffer);
		stepReduceKernel->SetArg(2, labelOrderBuffer);
		stepReduceKernel->SetLocalArg(3, localBufferSize_SR * sizeof(TreeVote));

		stepReduceKernel->setDim(3);
		stepReduceKernel->setGlobalSize(config["NUM_WG_INSTANCES_SR"] * config["NUM_WI_INSTANCES_SR"], config["NUM_WG_CHAINS_SR"] * config["NUM_WI_CHAINS_SR"], config["NUM_WI_TREES_SR"]);
		stepReduceKernel->setLocalSize(config["NUM_WI_INSTANCES_SR"], config["NUM_WI_CHAINS_SR"], config["NUM_WI_TREES_SR"]);

		size_t stepModelSize = ecc->getTotalSize() / numLabels;
		double time = 0.0;

		for (int chainIndex = 0; chainIndex < (oneStep ? 1 : numLabels); ++chainIndex)
		{
			stepReduceKernel->SetArg(4, chainIndex);
			classifyData->stepNodeIndexBuffer.writeFrom(nodeIndices + chainIndex * stepModelSize, stepModelSize * sizeof(int));
			classifyData->stepNodeValueBuffer.writeFrom(nodeValues + chainIndex * stepModelSize, stepModelSize * sizeof(double));
			stepCalcKernel->execute();
			stepReduceKernel->execute();
			time += stepCalcKernel->getRuntime()
				+ stepReduceKernel->getRuntime();
		}

		stepIntermediateBuffer.clear();

		delete stepCalcKernel;
		delete stepReduceKernel;

		return time;
	}
	catch (...)
	{
		stepIntermediateBuffer.clear();

		delete stepCalcKernel;
		delete stepReduceKernel;

		throw;
	}
}

double ECCExecutorNew::tuneClassifyFinal(Configuration config)
{
	Buffer finalIntermediateBuffer;
	Kernel* finalCalcKernel = NULL;
	Kernel* finalReduceKernel = NULL;
	try {
		config["NUM_INSTANCES"] = classifyData->numInstances;
		config["NUM_LABELS"] = numLabels;
		config["NUM_ATTRIBUTES"] = numAttributes;
		config["NUM_CHAINS"] = numChains;
		config["NUM_TREES"] = numTrees;
		config["MAX_LEVEL"] = maxLevel;
		config["NODES_PER_TREE"] = pow(2.0f, maxLevel + 1) - 1;
		config["NUM_WI_INSTANCES_FR"] = config["NUM_WI_INSTANCES_FC"];
		config["NUM_WG_INSTANCES_FR"] = config["NUM_WG_INSTANCES_FC"];
		config["NUM_WI_LABELS_FR"] = config["NUM_WI_LABELS_FC"];
		config["NUM_WG_LABELS_FR"] = config["NUM_WG_LABELS_FC"];

		std::string optionString;
		std::stringstream strstr;
		for (auto it = config.begin(); it != config.end(); ++it)
			strstr << " -D " << it->first << "=" << it->second;
		optionString = strstr.str();

		cl_program prog;
		PlatformUtil::buildProgramFromSource(finalCalcSource, prog, optionString.c_str());
		finalCalcKernel = new Kernel(prog, "finalCalc");
		clReleaseProgram(prog);

		PlatformUtil::buildProgramFromSource(finalReduceSource, prog, optionString.c_str());
		finalReduceKernel = new Kernel(prog, "finalReduce");
		clReleaseProgram(prog);

		int finalIntermediateBufferSize[3] = { classifyData->numInstances, numLabels, config["NUM_WG_CHAINS_FC"] };
		int finalIntermediateBufferTotalSize = finalIntermediateBufferSize[0] * finalIntermediateBufferSize[1] * finalIntermediateBufferSize[2];

		int localBufferSize_FC = config["NUM_WI_INSTANCES_FC"] * config["NUM_WI_LABELS_FC"] * config["NUM_WI_CHAINS_FC"];
		int localBufferSize_FR = config["NUM_WI_INSTANCES_FR"] * config["NUM_WI_LABELS_FR"] * config["NUM_WI_CHAINS_FR"];

		finalIntermediateBuffer = Buffer(finalIntermediateBufferTotalSize * sizeof(double), CL_MEM_READ_WRITE);

		finalCalcKernel->SetArg(0, classifyData->labelBuffer);
		finalCalcKernel->SetLocalArg(1, localBufferSize_FC * sizeof(double));
		finalCalcKernel->SetArg(2, finalIntermediateBuffer);

		finalCalcKernel->setDim(3);
		finalCalcKernel->setGlobalSize(config["NUM_WG_INSTANCES_FC"] * config["NUM_WI_INSTANCES_FC"], config["NUM_WG_LABELS_FC"] * config["NUM_WI_LABELS_FC"], config["NUM_WG_CHAINS_FC"] * config["NUM_WI_CHAINS_FC"]);
		finalCalcKernel->setLocalSize(config["NUM_WI_INSTANCES_FC"], config["NUM_WI_LABELS_FC"], config["NUM_WI_CHAINS_FC"]);

		finalReduceKernel->SetArg(0, finalIntermediateBuffer);
		finalReduceKernel->SetArg(1, classifyData->resultBuffer);
		finalReduceKernel->SetLocalArg(2, localBufferSize_FR * sizeof(double));

		finalReduceKernel->setDim(3);
		finalReduceKernel->setGlobalSize(config["NUM_WG_INSTANCES_FR"] * config["NUM_WI_INSTANCES_FR"], config["NUM_WG_LABELS_FR"] * config["NUM_WI_LABELS_FR"], config["NUM_WI_CHAINS_FR"]);
		finalReduceKernel->setLocalSize(config["NUM_WI_INSTANCES_FR"], config["NUM_WI_LABELS_FR"], config["NUM_WI_CHAINS_FR"]);

		finalCalcKernel->execute();
		finalReduceKernel->execute();

		finalIntermediateBuffer.clear();

		double time = finalCalcKernel->getRuntime()
			+ finalReduceKernel->getRuntime();

		delete finalCalcKernel;
		delete finalReduceKernel;

		return time;
	}
	catch (...)
	{
		finalIntermediateBuffer.clear();

		delete finalCalcKernel;
		delete finalReduceKernel;

		throw;
	}
}

void ECCExecutorNew::finishClassify()
{
	classifyData->dataBuffer.clear();
	classifyData->resultBuffer.clear();
	classifyData->labelBuffer.clear();
	classifyData->stepNodeValueBuffer.clear();
	classifyData->stepNodeIndexBuffer.clear();

	delete classifyData;
}

std::vector<MultilabelPrediction> ECCExecutorNew::runClassify(ECCData& data, Configuration config)
{
	std::cout << std::endl << "--- NEW CLASSIFICATION ---" << std::endl;
	
	Util::StopWatch totalClassifyTime;
	totalClassifyTime.start();

	int numInstances = data.getSize();
	size_t stepModelSize = ecc->getTotalSize() / numLabels;
	Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

	double* dataArray = new double[data.getValueCount() * data.getSize()];
	int dataBuffIdx = 0;
	for (MultilabelInstance inst : data.getInstances())
	{
		memcpy(dataArray + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
		dataBuffIdx += inst.getValueCount();
	}
	dataBuffer.writeFrom(dataArray, dataBuffer.getSize());
	delete[] dataArray;

	Buffer resultBuffer(data.getSize() * data.getLabelCount() * sizeof(double), CL_MEM_WRITE_ONLY);
	Buffer labelBuffer(data.getSize() * data.getLabelCount() * numChains * sizeof(double), CL_MEM_WRITE_ONLY);

	Buffer stepNodeValueBuffer(sizeof(double) * stepModelSize, CL_MEM_READ_ONLY);
	Buffer stepNodeIndexBuffer(sizeof(int) * stepModelSize, CL_MEM_READ_ONLY);

	config["NUM_INSTANCES"] = numInstances;
	config["NUM_LABELS"] = numLabels;
	config["NUM_ATTRIBUTES"] = numAttributes;
	config["NUM_CHAINS"] = numChains;
	config["NUM_TREES"] = numTrees;
	config["MAX_LEVEL"] = maxLevel;
	config["NODES_PER_TREE"] = pow(2.0f, maxLevel + 1) - 1;
	config["NUM_WI_INSTANCES_SR"] = config["NUM_WI_INSTANCES_SC"];
	config["NUM_WG_INSTANCES_SR"] = config["NUM_WG_INSTANCES_SC"];
	config["NUM_WI_CHAINS_SR"] = config["NUM_WI_CHAINS_SC"];
	config["NUM_WG_CHAINS_SR"] = config["NUM_WG_CHAINS_SC"];
	config["NUM_WI_INSTANCES_FR"] = config["NUM_WI_INSTANCES_FC"];
	config["NUM_WG_INSTANCES_FR"] = config["NUM_WG_INSTANCES_FC"];
	config["NUM_WI_LABELS_FR"] = config["NUM_WI_LABELS_FC"];
	config["NUM_WG_LABELS_FR"] = config["NUM_WG_LABELS_FC"];

	std::string optionString;
	std::stringstream strstr;
	for (auto it = config.begin(); it != config.end(); ++it)
		strstr << " -D " << it->first << "=" << it->second;
	optionString = strstr.str();

	Util::StopWatch classifyCompileTime;
	classifyCompileTime.start();

	cl_program prog;
	PlatformUtil::buildProgramFromSource(stepCalcSource, prog, optionString.c_str());
	Kernel* stepCalcKernel = new Kernel(prog, "stepCalc");
	clReleaseProgram(prog);

	PlatformUtil::buildProgramFromSource(stepReduceSource, prog, optionString.c_str());
	Kernel* stepReduceKernel = new Kernel(prog, "stepReduce");
	clReleaseProgram(prog);

	PlatformUtil::buildProgramFromSource(finalCalcSource, prog, optionString.c_str());
	Kernel* finalCalcKernel = new Kernel(prog, "finalCalc");
	clReleaseProgram(prog);

	PlatformUtil::buildProgramFromSource(finalReduceSource, prog, optionString.c_str());
	Kernel* finalReduceKernel = new Kernel(prog, "finalReduce");
	clReleaseProgram(prog);

	measurement["classifyCompileTime"] = classifyCompileTime.stop();

	int stepIntermediateBufferSize[3] = { numInstances, numChains, config["NUM_WG_TREES_SC"] };
	int stepIntermediateBufferTotalSize = stepIntermediateBufferSize[0] * stepIntermediateBufferSize[1] * stepIntermediateBufferSize[2];

	int localBufferSize_SC = config["NUM_WI_INSTANCES_SC"] * config["NUM_WI_CHAINS_SC"] * config["NUM_WI_TREES_SC"];
	int localBufferSize_SR = config["NUM_WI_INSTANCES_SR"] * config["NUM_WI_CHAINS_SR"] * config["NUM_WI_TREES_SR"];

	Buffer stepIntermediateBuffer(stepIntermediateBufferTotalSize * sizeof(TreeVote), CL_MEM_READ_WRITE);

	stepCalcKernel->SetArg(0, stepNodeValueBuffer);
	stepCalcKernel->SetArg(1, stepNodeIndexBuffer);
	stepCalcKernel->SetArg(2, dataBuffer);
	stepCalcKernel->SetArg(3, labelBuffer);
	stepCalcKernel->SetLocalArg(4, localBufferSize_SC * sizeof(TreeVote));
	stepCalcKernel->SetArg(5, stepIntermediateBuffer);

	stepCalcKernel->setDim(3);
	stepCalcKernel->setGlobalSize(config["NUM_WG_INSTANCES_SC"] * config["NUM_WI_INSTANCES_SC"], config["NUM_WG_CHAINS_SC"] * config["NUM_WI_CHAINS_SC"], config["NUM_WG_TREES_SC"] * config["NUM_WI_TREES_SC"]);
	stepCalcKernel->setLocalSize(config["NUM_WI_INSTANCES_SC"], config["NUM_WI_CHAINS_SC"], config["NUM_WI_TREES_SC"]);

	stepReduceKernel->SetArg(0, stepIntermediateBuffer);
	stepReduceKernel->SetArg(1, labelBuffer);
	stepReduceKernel->SetArg(2, labelOrderBuffer);
	stepReduceKernel->SetLocalArg(3, localBufferSize_SR * sizeof(TreeVote));

	stepReduceKernel->setDim(3);
	stepReduceKernel->setGlobalSize(config["NUM_WG_INSTANCES_SR"] * config["NUM_WI_INSTANCES_SR"], config["NUM_WG_CHAINS_SR"] * config["NUM_WI_CHAINS_SR"], config["NUM_WI_TREES_SR"]);
	stepReduceKernel->setLocalSize(config["NUM_WI_INSTANCES_SR"], config["NUM_WI_CHAINS_SR"], config["NUM_WI_TREES_SR"]);

	int finalIntermediateBufferSize[3] = { numInstances, numLabels, config["NUM_WG_CHAINS_FC"] };
	int finalIntermediateBufferTotalSize = finalIntermediateBufferSize[0] * finalIntermediateBufferSize[1] * finalIntermediateBufferSize[2];

	int localBufferSize_FC = config["NUM_WI_INSTANCES_FC"] * config["NUM_WI_LABELS_FC"] * config["NUM_WI_CHAINS_FC"];
	int localBufferSize_FR = config["NUM_WI_INSTANCES_FR"] * config["NUM_WI_LABELS_FR"] * config["NUM_WI_CHAINS_FR"];

	Buffer finalIntermediateBuffer(finalIntermediateBufferTotalSize * sizeof(double), CL_MEM_READ_WRITE);

	finalCalcKernel->SetArg(0, labelBuffer);
	finalCalcKernel->SetLocalArg(1, localBufferSize_FC * sizeof(double));
	finalCalcKernel->SetArg(2, finalIntermediateBuffer);

	finalCalcKernel->setDim(3);
	finalCalcKernel->setGlobalSize(config["NUM_WG_INSTANCES_FC"] * config["NUM_WI_INSTANCES_FC"], config["NUM_WG_LABELS_FC"] * config["NUM_WI_LABELS_FC"], config["NUM_WG_CHAINS_FC"] * config["NUM_WI_CHAINS_FC"]);
	finalCalcKernel->setLocalSize(config["NUM_WI_INSTANCES_FC"], config["NUM_WI_LABELS_FC"], config["NUM_WI_CHAINS_FC"]);

	finalReduceKernel->SetArg(0, finalIntermediateBuffer);
	finalReduceKernel->SetArg(1, resultBuffer);
	finalReduceKernel->SetLocalArg(2, localBufferSize_FR * sizeof(double));

	finalReduceKernel->setDim(3);
	finalReduceKernel->setGlobalSize(config["NUM_WG_INSTANCES_FR"] * config["NUM_WI_INSTANCES_FR"], config["NUM_WG_LABELS_FR"] * config["NUM_WI_LABELS_FR"], config["NUM_WI_CHAINS_FR"]);
	finalReduceKernel->setLocalSize(config["NUM_WI_INSTANCES_FR"], config["NUM_WI_LABELS_FR"], config["NUM_WI_CHAINS_FR"]);

	measurement["classifyStepCalcKernel"] = 0;
	measurement["classifyStepReduceKernel"] = 0;
	measurement["classifyNodeIndexWrite"] = 0;
	measurement["classifyNodeValueWrite"] = 0;

	Util::StopWatch classifyLoopTime;
	classifyLoopTime.start();
	for (int chainIndex = 0; chainIndex < numLabels; ++chainIndex)
	{
		stepReduceKernel->SetArg(4, chainIndex);
		stepNodeIndexBuffer.writeFrom(nodeIndices + chainIndex * stepModelSize, stepModelSize * sizeof(int));
		stepNodeValueBuffer.writeFrom(nodeValues + chainIndex * stepModelSize, stepModelSize * sizeof(double));
		stepCalcKernel->execute();
		stepReduceKernel->execute();

		measurement["classifyStepCalcKernel"] += stepCalcKernel->getRuntime();
		measurement["classifyStepReduceKernel"] += stepReduceKernel->getRuntime();
		measurement["classifyNodeIndexWrite"] += stepNodeIndexBuffer.getTransferTime();
		measurement["classifyNodeValueWrite"] += stepNodeValueBuffer.getTransferTime();
	}
	finalCalcKernel->execute();
	finalReduceKernel->execute();

	double* results = new double[numInstances * numLabels];
	resultBuffer.readTo(results, resultBuffer.getSize());
	std::vector<MultilabelPrediction> predictions;

	for (int d = 0; d < numInstances; ++d)
	{
		predictions.push_back(MultilabelPrediction(results + d * numLabels, results + (d + 1) * numLabels));
	}
	delete[] results;

	measurement["classifyFinalCalcKernel"] = finalCalcKernel->getRuntime();
	measurement["classifyFinalReduceKernel"] = finalReduceKernel->getRuntime();
	measurement["classifyDataWrite"] = dataBuffer.getTransferTime();
	measurement["classifyLoopTime"] = classifyLoopTime.stop();
	measurement["classifyTotalTime"] = totalClassifyTime.stop();
	measurement["classifyValuesRead"] = resultBuffer.getTransferTime();

	stepIntermediateBuffer.clear();
	finalIntermediateBuffer.clear();

	delete stepCalcKernel;
	delete stepReduceKernel;
	delete finalCalcKernel;
	delete finalReduceKernel;
	
	dataBuffer.clear();
	resultBuffer.clear();
	labelBuffer.clear();
	stepNodeValueBuffer.clear();
	stepNodeIndexBuffer.clear();

	return predictions;
}

Measurement ECCExecutorNew::getMeasurement()
{
	return measurement;
}

ECCExecutorNew::~ECCExecutorNew()
{
	delete[] nodeIndices;
	delete[] nodeValues;
	labelOrderBuffer.clear();
	delete ecc;
}
