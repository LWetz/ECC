#include "ECCExecutorOld.hpp"

std::vector<int> ECCExecutorOld::partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc)
{
	std::vector<int> indicesList;
	if (partitionInstance)
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

ECCExecutorOld::ECCExecutorOld(int _maxLevel, int _maxAttributes, int _forestSize)
	: nodeValues(NULL), nodeIndices(NULL), labelOrderBuffer(NULL), partitionInstance(true),
	maxLevel(_maxLevel), forestSize(_forestSize), maxAttributes(_maxAttributes), oldKernelMode(false)
{
	Util::RANDOM.setSeed(133713);
}

void ECCExecutorOld::runBuild(ECCData& data, int treeLimit, int ensembleSize, int ensembleSubSetSize, int forestSubSetSize)
{
	this->chainSize = data.getLabelCount();
	this->ensembleSize = ensembleSize;

	int totalTrees = ensembleSize * chainSize * forestSize;
	while (treeLimit % chainSize != 0 || totalTrees % treeLimit != 0)
		--treeLimit;

	int chunkSize = treeLimit / chainSize;

	delete[] nodeValues;
	delete[] nodeIndices;
	labelOrderBuffer.clear();

	std::cout << std::endl << "--- BUILD ---" << std::endl;
	cl_program prog;
	PlatformUtil::buildProgramFromFile("eccBuild.cl", prog);//("\\\\X-THINK\\Users\\Public\\eccBuild.cl", prog);
	Kernel* buildKernel = new Kernel(prog, "eccBuild");
	clReleaseProgram(prog);

	if (data.getSize() == ensembleSubSetSize && data.getSize() == forestSubSetSize)
		partitionInstance = false;
	else
		partitionInstance = true;

	ecc = new EnsembleOfClassifierChains(data.getValueCount(), data.getLabelCount(), maxLevel, forestSize, ensembleSize, ensembleSubSetSize, forestSubSetSize);
	int globalSize = ecc->getChainSize() * chunkSize;
	int nodesLastLevel = pow(2.0f, maxLevel);
	int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;

	labelOrderBuffer = Buffer(sizeof(int) * ecc->getEnsembleSize() * ecc->getChainSize(), CL_MEM_READ_ONLY);

	int i = 0;
	for (int chain = 0; chain < ecc->getEnsembleSize(); ++chain)
	{
		for (int forest = 0; forest < ecc->getChainSize(); ++forest)
		{
			static_cast<int*>(labelOrderBuffer.getData())[i++] = ecc->getChains()[chain].getLabelOrder()[forest];
		}
	}

	nodeValues = new double[ecc->getTotalSize()];
	nodeIndices = new int[ecc->getTotalSize()];
	Buffer tmpNodeValueBuffer(sizeof(double) * globalSize * nodesPerTree, CL_MEM_READ_WRITE);
	Buffer tmpNodeIndexBuffer(sizeof(int) * globalSize * nodesPerTree, CL_MEM_READ_WRITE);

	int dataSize = data.getSize();
	Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

	int dataBuffIdx = 0;
	for (MultilabelInstance inst : data.getInstances())
	{
		memcpy(static_cast<double*>(dataBuffer.getData()) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
		dataBuffIdx += inst.getValueCount();
	}

	int subSetSize = ecc->getForestSubSetSize();
	int numValues = data.getValueCount();
	int numAttributes = data.getAttribCount();

	int maxSplits = ecc->getForestSubSetSize() - 1;
	Buffer instancesBuffer(sizeof(int) * maxSplits*globalSize, CL_MEM_READ_WRITE);

	std::vector<int> indicesList(partitionInstances(data, *ecc));

	Buffer instancesNextBuffer(sizeof(int) * maxSplits*globalSize, CL_MEM_READ_WRITE);
	Buffer instancesLengthBuffer(sizeof(int) * nodesLastLevel*globalSize, CL_MEM_READ_WRITE);
	Buffer instancesNextLengthBuffer(sizeof(int) * nodesLastLevel*globalSize, CL_MEM_READ_WRITE);
	Buffer seedsBuffer(sizeof(int) * globalSize, CL_MEM_READ_ONLY);
	Buffer voteBuffer(sizeof(int) * nodesLastLevel*globalSize, CL_MEM_READ_WRITE);

	ConstantBuffer pGidMultiplier(0);
	ConstantBuffer pDataSize(dataSize);
	ConstantBuffer pSubSetSize(subSetSize);
	ConstantBuffer pNumValues(numValues);
	ConstantBuffer pNumAttributes(numAttributes);
	ConstantBuffer pMaxAttributes(maxAttributes);
	ConstantBuffer pMaxLevel(maxLevel);
	ConstantBuffer pChainSize(chainSize);
	ConstantBuffer pMaxSplits(maxSplits);
	ConstantBuffer pForestSize(forestSize);

	int gidMultiplier = 0;

	buildKernel->setDim(1);
	buildKernel->setGlobalSize(globalSize);
	buildKernel->setLocalSize(3);

	double totalTime = .0;

	buildKernel->SetArg(0, pGidMultiplier);
	buildKernel->SetArg(1, seedsBuffer, true);
	buildKernel->SetArg(2, dataBuffer, true);
	buildKernel->SetArg(3, pDataSize, true);
	buildKernel->SetArg(4, pSubSetSize, true);
	buildKernel->SetArg(5, labelOrderBuffer, true);
	buildKernel->SetArg(6, pNumValues, true);
	buildKernel->SetArg(7, pNumAttributes, true);
	buildKernel->SetArg(8, pMaxAttributes, true);
	buildKernel->SetArg(9, pMaxLevel, true);
	buildKernel->SetArg(10, pChainSize, true);
	buildKernel->SetArg(11, pMaxSplits, true);
	buildKernel->SetArg(12, pForestSize, true);
	buildKernel->SetArg(13, instancesBuffer);
	buildKernel->SetArg(14, instancesNextBuffer);
	buildKernel->SetArg(15, instancesLengthBuffer);
	buildKernel->SetArg(16, instancesNextLengthBuffer);
	buildKernel->SetArg(17, tmpNodeValueBuffer);
	buildKernel->SetArg(18, tmpNodeIndexBuffer);
	buildKernel->SetArg(19, voteBuffer, true);

	Util::StopWatch stopWatch;
	stopWatch.start();
	for (int chunk = 0; chunk < ensembleSize * forestSize; chunk += chunkSize)
	{
		pGidMultiplier.writeFrom(&gidMultiplier, sizeof(int));

		for (int seed = 0; seed < globalSize; ++seed)
		{
			int rnd = Util::randomInt(INT_MAX);
			static_cast<int*>(seedsBuffer.getData())[seed] = rnd;
		}
		seedsBuffer.write();

		memcpy(instancesBuffer.getData(), indicesList.data() + gidMultiplier * globalSize * maxSplits, globalSize * maxSplits * sizeof(int));
		instancesBuffer.write();

		buildKernel->execute();
		totalTime += buildKernel->getRuntime();

		tmpNodeIndexBuffer.read();
		tmpNodeValueBuffer.read();

		memcpy(((uint8_t*)nodeIndices) + gidMultiplier*tmpNodeIndexBuffer.getSize(), tmpNodeIndexBuffer.getData(), tmpNodeIndexBuffer.getSize());
		memcpy(((uint8_t*)nodeValues) + gidMultiplier*tmpNodeValueBuffer.getSize(), tmpNodeValueBuffer.getData(), tmpNodeValueBuffer.getSize());

		++gidMultiplier;
	}

	std::cout << "Build took " << ((double)stopWatch.stop())*1e-06 << " ms total." << std::endl;
	std::cout << "Build took " << ((double)totalTime)*1e-06 << " ms kernel time." << std::endl;

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
	PlatformUtil::finish();
}

void ECCExecutorOld::runClassify(ECCData& data, std::vector<double>& values, std::vector<int>& votes, bool fix)
{
	std::cout << std::endl << "--- " << (fix ? "FIXED" : "OLD") << " CLASSIFICATION ---" << std::endl;
	cl_program prog;
	PlatformUtil::buildProgramFromFile(fix ? "eccClassify_fix.cl" : "eccClassify.cl", prog);
	Kernel* classifyKernel = new Kernel(prog, "eccClassify");
	clReleaseProgram(prog);

	int dataSize = data.getSize();
	Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_WRITE);

	int dataBuffIdx = 0;
	for (MultilabelInstance inst : data.getInstances())
	{
		memcpy(static_cast<double*>(dataBuffer.getData()) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
		dataBuffIdx += inst.getValueCount();
	}

	Buffer resultBuffer(dataSize * data.getLabelCount() * sizeof(double), CL_MEM_READ_WRITE);
	Buffer voteBuffer(dataSize * data.getLabelCount() * sizeof(int), CL_MEM_WRITE_ONLY);

	Buffer nodeValueBuffer(ecc->getTotalSize() * sizeof(double), CL_MEM_READ_ONLY);
	Buffer nodeIndexBuffer(ecc->getTotalSize() * sizeof(int), CL_MEM_READ_ONLY);
	memcpy(nodeValueBuffer.getData(), nodeValues, nodeValueBuffer.getSize());
	memcpy(nodeIndexBuffer.getData(), nodeIndices, nodeIndexBuffer.getSize());

	int numValues = data.getValueCount();

	ConstantBuffer maxLevelBuffer(maxLevel);
	ConstantBuffer forestSizeBuffer(forestSize);
	ConstantBuffer chainSizeBuffer(chainSize);
	ConstantBuffer ensembleSizeBuffer(ensembleSize);
	ConstantBuffer numValuesBuffer(numValues);

	classifyKernel->SetArg(0, nodeValueBuffer, true);
	classifyKernel->SetArg(1, nodeIndexBuffer, true);
	classifyKernel->SetArg(2, labelOrderBuffer, true);
	classifyKernel->SetArg(3, maxLevelBuffer, true);
	classifyKernel->SetArg(4, forestSizeBuffer, true);
	classifyKernel->SetArg(5, chainSizeBuffer, true);
	classifyKernel->SetArg(6, ensembleSizeBuffer, true);
	classifyKernel->SetArg(7, dataBuffer, true);
	classifyKernel->SetArg(8, numValuesBuffer, true);
	classifyKernel->SetArg(9, resultBuffer, true);
	classifyKernel->SetArg(10, voteBuffer, true);

	classifyKernel->setDim(1);
	classifyKernel->setGlobalSize(data.getSize());
	classifyKernel->setLocalSize(1);

	classifyKernel->execute();

	resultBuffer.read();
	voteBuffer.read();

	for (int n = 0; n < data.getLabelCount()*data.getSize(); ++n)
	{
		values.push_back(static_cast<double*>(resultBuffer.getData())[n]);
		votes.push_back(static_cast<int*>(voteBuffer.getData())[n]);
	}

	voteBuffer.read();

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
}

Measurement ECCExecutorOld::getMeasurement()
{
	return measurement;
}

ECCExecutorOld::~ECCExecutorOld()
{
	delete[] nodeIndices;
	delete[] nodeValues;
	labelOrderBuffer.clear();
	delete ecc;
}
