#include "ECCExecutorOld.hpp"

std::vector<int> ECCExecutorOld::partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc)
{
	std::vector<int> indicesList;
	if (ecc.getEnsembleSubSetSize() != data.getSize() || ecc.getForestSubSetSize() != data.getSize())
	{
		indicesList = ecc.partitionInstanceIndices(data.getSize());
	}
	else
	{
		for (size_t chain = 0; chain < ecc.getEnsembleSize(); ++chain)
		{
			for (size_t forest = 0; forest < ecc.getChainSize(); ++forest)
			{
				for (size_t tree = 0; tree < ecc.getForestSize(); ++tree)
				{
					for (size_t index = 0; index < data.getSize(); ++index)
					{
						indicesList.push_back(index);
					}
				}
			}
		}
	}
	return indicesList;
}

ECCExecutorOld::ECCExecutorOld(size_t _maxLevel, size_t _maxAttributes, size_t _numAttributes, size_t _numTrees, size_t _numLabels, size_t _numChains, size_t _ensembleSubSetSize, size_t _forestSubSetSize)
	: nodeValues(NULL), nodeIndices(NULL), labelOrderBuffer(NULL), maxLevel(_maxLevel),
	numTrees(_numTrees), maxAttributes(_maxAttributes), numLabels(_numLabels),
	numChains(_numChains), numAttributes(_numAttributes),
	buildSource(Util::loadFileToString("eccBuild.cl")),
	classifySource(Util::loadFileToString("eccClassify.cl")),
	classifyFixSource(Util::loadFileToString("eccClassify_fix.cl"))
{
	Util::RANDOM.setSeed(133713);
	Util::StopWatch stopWatch;
	stopWatch.start();
	ecc = new EnsembleOfClassifierChains(_numAttributes + numLabels, numLabels, maxLevel, numTrees, numChains, _ensembleSubSetSize, _forestSubSetSize);

	labelOrderBuffer = Buffer(sizeof(int) * ecc->getEnsembleSize() * ecc->getChainSize(), CL_MEM_READ_ONLY);
	int* labelOrders = new int[ecc->getEnsembleSize() * ecc->getChainSize()];

	size_t i = 0;
	for (size_t chain = 0; chain < ecc->getEnsembleSize(); ++chain)
	{
		for (size_t forest = 0; forest < ecc->getChainSize(); ++forest)
		{
			labelOrders[i++] = ecc->getChains()[chain].getLabelOrder()[forest];
		}
	}
	labelOrderBuffer.writeFrom(labelOrders, labelOrderBuffer.getSize());
	delete[] labelOrders;

	nodeValues = new double[ecc->getTotalSize()];
	nodeIndices = new int[ecc->getTotalSize()];

	measurement["oldSetupTotalTime"] = stopWatch.stop();
	measurement["oldSetupLabelOrdersWrite"] = labelOrderBuffer.getTransferTime();
}

void ECCExecutorOld::runBuild(ECCData& data, size_t treeLimit)
{
	std::cout << std::endl << "--- BUILD ---" << std::endl;

	Util::StopWatch totalBuildTime;
	totalBuildTime.start();

	size_t totalTrees = numChains * numLabels * numTrees;
	while (treeLimit % numLabels != 0 || totalTrees % treeLimit != 0)
		--treeLimit;

	size_t globalSize = treeLimit;
	size_t chunkSize = globalSize / numLabels;

	size_t nodesLastLevel = pow(2.0f, maxLevel);
	size_t nodesPerTree = pow(2.0f, maxLevel + 1) - 1;

	size_t maxSplits = ecc->getForestSubSetSize() - 1;

	Util::StopWatch buildCompileTime;
	buildCompileTime.start();

	cl_program prog;
	PlatformUtil::buildProgramFromFile("eccBuild.cl", prog);
	Kernel* buildKernel = new Kernel(prog, "eccBuild");
	clReleaseProgram(prog);

	measurement["oldBuildCompileTime"] = buildCompileTime.stop();

	Buffer tmpNodeValueBuffer(sizeof(double) * globalSize * nodesPerTree, CL_MEM_READ_WRITE);
	Buffer tmpNodeIndexBuffer(sizeof(int) * globalSize * nodesPerTree, CL_MEM_READ_WRITE);

	Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

	double* dataArray = new double[data.getValueCount() * data.getSize()];
	size_t dataBuffIdx = 0;
	for (MultilabelInstance inst : data.getInstances())
	{
		memcpy(dataArray + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
		dataBuffIdx += inst.getValueCount();
	}
	dataBuffer.writeFrom(dataArray, dataBuffer.getSize());
	delete[] dataArray;

	size_t numValues = data.getValueCount();
	size_t numAttributes = data.getAttribCount();

	Buffer instancesBuffer(sizeof(int) * maxSplits*globalSize, CL_MEM_READ_WRITE);

	std::vector<int> indicesList(partitionInstances(data, *ecc));

	Buffer instancesNextBuffer(sizeof(int) * maxSplits*globalSize, CL_MEM_READ_WRITE);
	Buffer instancesLengthBuffer(sizeof(int) * nodesLastLevel*globalSize, CL_MEM_READ_WRITE);
	Buffer instancesNextLengthBuffer(sizeof(int) * nodesLastLevel*globalSize, CL_MEM_READ_WRITE);
	Buffer seedsBuffer(sizeof(int) * globalSize, CL_MEM_READ_ONLY);
	Buffer voteBuffer(sizeof(int) * nodesLastLevel*globalSize, CL_MEM_READ_WRITE);

	ConstantBuffer pGidMultiplier(0);
	ConstantBuffer pDataSize(data.getSize());
	ConstantBuffer pSubSetSize(ecc->getForestSubSetSize());
	ConstantBuffer pNumValues(numValues);
	ConstantBuffer pNumAttributes(numAttributes);
	ConstantBuffer pMaxAttributes(maxAttributes);
	ConstantBuffer pMaxLevel(maxLevel);
	ConstantBuffer pChainSize(numLabels);
	ConstantBuffer pMaxSplits(maxSplits);
	ConstantBuffer pForestSize(numTrees);

	int gidMultiplier = 0;

	buildKernel->setDim(1);
	buildKernel->setGlobalSize({ globalSize });
	buildKernel->setLocalSize({ numLabels });

	buildKernel->SetArg(0, pGidMultiplier);
	buildKernel->SetArg(1, seedsBuffer);
	buildKernel->SetArg(2, dataBuffer);
	buildKernel->SetArg(3, pDataSize);
	buildKernel->SetArg(4, pSubSetSize);
	buildKernel->SetArg(5, labelOrderBuffer);
	buildKernel->SetArg(6, pNumValues);
	buildKernel->SetArg(7, pNumAttributes);
	buildKernel->SetArg(8, pMaxAttributes);
	buildKernel->SetArg(9, pMaxLevel);
	buildKernel->SetArg(10, pChainSize);
	buildKernel->SetArg(11, pMaxSplits);
	buildKernel->SetArg(12, pForestSize);
	buildKernel->SetArg(13, instancesBuffer);
	buildKernel->SetArg(14, instancesNextBuffer);
	buildKernel->SetArg(15, instancesLengthBuffer);
	buildKernel->SetArg(16, instancesNextLengthBuffer);
	buildKernel->SetArg(17, tmpNodeValueBuffer);
	buildKernel->SetArg(18, tmpNodeIndexBuffer);
	buildKernel->SetArg(19, voteBuffer);

	int* seeds = new int[globalSize];

	measurement["oldBuildValuesZero"] = 0;
	measurement["oldBuildKernel"] = 0;
	measurement["oldBuildSeedsWrite"] = 0;
	measurement["oldBuildInstancesWrite"] = 0;
	measurement["oldBuildNodeIndexRead"] = 0;
	measurement["oldBuildNodeValueRead"] = 0;

	Util::StopWatch buildLoopTime;
	buildLoopTime.start();
	for (size_t chunk = 0; chunk < numChains * numTrees; chunk += chunkSize)
	{
		pGidMultiplier.write(gidMultiplier);

		tmpNodeValueBuffer.zero();
		for (size_t seed = 0; seed < globalSize; ++seed)
		{
			seeds[seed] = Util::randomInt(INT_MAX);
		}
		seedsBuffer.writeFrom(seeds, seedsBuffer.getSize());

		instancesBuffer.writeFrom(indicesList.data() + gidMultiplier * globalSize * maxSplits, globalSize * maxSplits * sizeof(int));

		buildKernel->execute();

		tmpNodeIndexBuffer.readTo(nodeIndices + gidMultiplier*globalSize*nodesPerTree, globalSize*nodesPerTree * sizeof(int));
		tmpNodeValueBuffer.readTo(nodeValues + gidMultiplier*globalSize*nodesPerTree, globalSize*nodesPerTree * sizeof(double));

		++gidMultiplier;
		measurement["oldBuildValuesZero"] += tmpNodeValueBuffer.getTransferTime();
		measurement["oldBuildKernel"] += buildKernel->getRuntime();
		measurement["oldBuildSeedsWrite"] += seedsBuffer.getTransferTime();
		measurement["oldBuildInstancesWrite"] += instancesBuffer.getTransferTime();
		measurement["oldBuildNodeIndexRead"] += tmpNodeIndexBuffer.getTransferTime();
		measurement["oldBuildNodeValueRead"] += tmpNodeValueBuffer.getTransferTime();
	}
	measurement["oldBuildDataWrite"] = dataBuffer.getTransferTime();
	measurement["oldBuildLoopTime"] = buildLoopTime.stop();
	measurement["oldBuildTotalTime"] = totalBuildTime.stop();

	delete[] seeds;
	tmpNodeIndexBuffer.clear();
	tmpNodeValueBuffer.clear();
	dataBuffer.clear();
	instancesBuffer.clear();
	instancesLengthBuffer.clear();
	instancesNextBuffer.clear();
	instancesNextLengthBuffer.clear();
	seedsBuffer.clear();
	voteBuffer.clear();

	delete buildKernel;
}

std::vector<MultilabelPrediction> ECCExecutorOld::runClassify(ECCData& data, bool fix)
{
	std::cout << std::endl << "--- " << (fix ? "FIXED" : "OLD") << " CLASSIFICATION ---" << std::endl;

	Util::StopWatch totalClassifyTime, classifyCompileTime;
	totalClassifyTime.start();
	classifyCompileTime.start();

	cl_program prog;
	PlatformUtil::buildProgramFromFile(fix ? "eccClassify_fix.cl" : "eccClassify.cl", prog);
	Kernel* classifyKernel = new Kernel(prog, "eccClassify");
	clReleaseProgram(prog);

	measurement["oldClassifyCompileTime"] = classifyCompileTime.stop();

	size_t dataSize = data.getSize();
	Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_WRITE);

	double* dataArray = new double[data.getValueCount() * dataSize];
	size_t dataBuffIdx = 0;
	for (MultilabelInstance inst : data.getInstances())
	{
		memcpy(dataArray + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
		dataBuffIdx += inst.getValueCount();
	}
	dataBuffer.writeFrom(dataArray, dataBuffer.getSize());
	delete[] dataArray;

	Buffer resultBuffer(dataSize * data.getLabelCount() * sizeof(double), CL_MEM_READ_WRITE);
	resultBuffer.zero();
	Buffer voteBuffer(dataSize * data.getLabelCount() * sizeof(int), CL_MEM_WRITE_ONLY);

	Buffer nodeValueBuffer(ecc->getTotalSize() * sizeof(double), CL_MEM_READ_ONLY);
	Buffer nodeIndexBuffer(ecc->getTotalSize() * sizeof(int), CL_MEM_READ_ONLY);
	nodeValueBuffer.writeFrom(nodeValues, nodeValueBuffer.getSize());
	nodeIndexBuffer.writeFrom(nodeIndices, nodeIndexBuffer.getSize());

	size_t numValues = data.getValueCount();

	ConstantBuffer maxLevelBuffer(maxLevel);
	ConstantBuffer forestSizeBuffer(numTrees);
	ConstantBuffer chainSizeBuffer(numLabels);
	ConstantBuffer ensembleSizeBuffer(numChains);
	ConstantBuffer numValuesBuffer(numValues);

	classifyKernel->SetArg(0, nodeValueBuffer);
	classifyKernel->SetArg(1, nodeIndexBuffer);
	classifyKernel->SetArg(2, labelOrderBuffer);
	classifyKernel->SetArg(3, maxLevelBuffer);
	classifyKernel->SetArg(4, forestSizeBuffer);
	classifyKernel->SetArg(5, chainSizeBuffer);
	classifyKernel->SetArg(6, ensembleSizeBuffer);
	classifyKernel->SetArg(7, dataBuffer);
	classifyKernel->SetArg(8, numValuesBuffer);
	classifyKernel->SetArg(9, resultBuffer);
	classifyKernel->SetArg(10, voteBuffer);

	classifyKernel->setDim(1);

	int numWG = 32;
	for(;data.getSize() % numWG != 0; numWG--);


	classifyKernel->setGlobalSize({ data.getSize() });
	classifyKernel->setLocalSize({ data.getSize() / numWG });

	classifyKernel->execute();

	measurement["oldClassifyResultZero"] = resultBuffer.getTransferTime();
	Util::StopWatch readBackTime;
	readBackTime.start();

	double* results = new double[dataSize * numLabels];
	int* votes = new int[dataSize * numLabels];
	resultBuffer.readTo(results, resultBuffer.getSize());
	voteBuffer.readTo(votes, voteBuffer.getSize());
	std::vector<MultilabelPrediction> predictions;

	for (size_t d = 0; d < dataSize; ++d)
	{
		for (size_t l = 0; l < numLabels; ++l)
		{
			results[d * numLabels + l] = (1.0 + results[d * numLabels + l] / votes[d * numLabels + l]) / 2.0;
		}
		predictions.push_back(MultilabelPrediction(results + d * numLabels, results + (d + 1) * numLabels));
	}
	delete[] results;
	delete[] votes;

	measurement["classifyReadBackTime"] = readBackTime.stop();
	measurement["oldClassifyNodeIndexWrite"] += nodeIndexBuffer.getTransferTime();
	measurement["oldClassifyNodeValueWrite"] += nodeValueBuffer.getTransferTime();
	measurement["oldClassifyKernel"] = classifyKernel->getRuntime();
	measurement["oldClassifyDataWrite"] = dataBuffer.getTransferTime();
	measurement["oldClassifyTotalTime"] = totalClassifyTime.stop();
	measurement["oldClassifyVotesRead"] = voteBuffer.getTransferTime();
	measurement["oldClassifyResultRead"] = resultBuffer.getTransferTime();

	dataBuffer.clear();
	resultBuffer.clear();
	voteBuffer.clear();
	maxLevelBuffer.clear();
	forestSizeBuffer.clear();
	chainSizeBuffer.clear();
	ensembleSizeBuffer.clear();
	numValuesBuffer.clear();

	nodeValueBuffer.clear();
	nodeIndexBuffer.clear();

	delete classifyKernel;

	return predictions;
}

Measurement ECCExecutorOld::getMeasurement()
{
	return measurement;
}

void getResults(std::vector<double> values, std::vector<int> votes)
{

}

ECCExecutorOld::~ECCExecutorOld()
{
	delete[] nodeIndices;
	delete[] nodeValues;
	labelOrderBuffer.clear();
	delete ecc;
}
