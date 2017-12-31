#pragma once

#include<CL/cl.h>

#include"EnsembleOfClassifierChains.h"
#include"PlatformUtil.h"
#include"ECCData.h"
#include "Kernel.h"
#include <chrono>
#include <climits>
#include "Util.h"
#include "Parameters.h"

class ECCExecutorOld
{
	EnsembleOfClassifierChains *ecc;

	bool partitionInstance;
	bool oldKernelMode;

	long time;

	double* nodeValues;
	int* nodeIndices;
	Buffer labelOrderBuffer;
	int maxLevel;
	int forestSize;
	int chainSize;
	int ensembleSize;
	int maxAttributes;

	size_t oldTime;

	std::vector<int> partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc)
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

public:
	ECCExecutorOld(int _maxLevel, int _maxAttributes, int _forestSize)
		: nodeValues(NULL), nodeIndices(NULL), labelOrderBuffer(NULL), partitionInstance(true),
		maxLevel(_maxLevel), forestSize(_forestSize), maxAttributes(_maxAttributes), oldKernelMode(false)
	{
		Util::RANDOM.setSeed(133713);
	}

	void runBuild(ECCData& data, int treesPerRun, int ensembleSize, int chainsPerRun, int ensembleSubSetSize, int forestSubSetSize)
	{
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
		int globalSize = ecc->getChainSize() * chainsPerRun * treesPerRun;
		int nodesLastLevel = pow(2.0f, maxLevel);
		int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;

		this->chainSize = ecc->getChainSize();
		this->ensembleSize = ecc->getEnsembleSize();

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

		int gidMultiplier = 0;
		int ensembleRuns = ensembleSize / chainsPerRun;
		int forestRuns = forestSize / treesPerRun;

		buildKernel->setDim(1);
		buildKernel->setGlobalSize(globalSize);
		buildKernel->setLocalSize(4);

		double totalTime = .0;

		buildKernel->SetArg(1, seedsBuffer, true);
		buildKernel->SetArg(2, dataBuffer, true);
		buildKernel->SetArg(3, dataSize);
		buildKernel->SetArg(4, subSetSize);
		buildKernel->SetArg(5, labelOrderBuffer, true);
		buildKernel->SetArg(6, numValues);
		buildKernel->SetArg(7, numAttributes);
		buildKernel->SetArg(8, maxAttributes);
		buildKernel->SetArg(9, maxLevel);
		buildKernel->SetArg(10, chainSize);
		buildKernel->SetArg(11, maxSplits);
		buildKernel->SetArg(12, forestSize);
		buildKernel->SetArg(13, instancesBuffer, true);
		buildKernel->SetArg(14, instancesNextBuffer, true);
		buildKernel->SetArg(15, instancesLengthBuffer, true);
		buildKernel->SetArg(16, instancesNextLengthBuffer, true);
		buildKernel->SetArg(17, tmpNodeValueBuffer, true);
		buildKernel->SetArg(18, tmpNodeIndexBuffer, true);
		buildKernel->SetArg(19, voteBuffer, true);

		Util::StopWatch stopWatch;
		stopWatch.start();
		for (int chain = 0; chain < ensembleSize; chain += chainsPerRun)
		{
			for (int tree = 0; tree < forestSize; tree += treesPerRun)
			{
				buildKernel->SetArg(0, gidMultiplier);

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

	template<typename T>
	void printBuffer(Buffer buff, std::ofstream& ofs)
	{
		int len = buff.getSize() / sizeof(T);

		for (int n = 0; n < len; ++n)
		{
			ofs << static_cast<T*>(buff.getData())[n] << "\n";
		}
	}

public:
	void runClassifyOld(ECCData& data, std::vector<double>& values, std::vector<int>& votes, bool fix = true)
	{
		std::cout << std::endl << "--- " << (fix ? "FIXED" : "OLD") << " CLASSIFICATION ---" << std::endl;
		cl_program prog;
		PlatformUtil::buildProgramFromFile(fix ? "OldKernels/eccClassify_fix.cl" : "OldKernels/eccClassify.cl", prog);
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

		oldTime = classifyKernel->getRuntime();

		resultBuffer.read();
		voteBuffer.read();
		std::cout << "Classification kernel took " << ((double)oldTime * 1e-06) << " ms." << std::endl;

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

	size_t getTime()
	{
		return oldTime;
	}

	~ECCExecutorOld()
	{
		delete[] nodeIndices;
		delete[] nodeValues;
		labelOrderBuffer.clear();
		delete ecc;
	}
};

