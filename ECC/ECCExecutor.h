#pragma once

#include<CL/cl.h>

#include"EnsembleOfClassifierChains.h"
#include"PlatformUtil.h"
#include"ECCData.h"
#include "Kernel.h"
#include <chrono>
#include "Util.h"

class ECCExecutor
{
	EnsembleOfClassifierChains *ecc;

	Kernel* classifyKernel;
	Kernel* buildKernel;

	bool partitionInstance;
	bool oldKernelMode;

	long time;

	struct Buffer
	{
		void* data;
		size_t size;
		cl_mem memObj;

		Buffer(size_t _size) : data(new uint8_t[_size]), size(_size), memObj(NULL)
		{
			memset(data, 0, size);
		}

		void clear()
		{
			delete[] data;
			if(memObj != NULL)
				clReleaseMemObject(memObj);
		}
	};

	Buffer nodeValueBuffer;
	Buffer nodeIndexBuffer;
	Buffer labelOrderBuffer;
	int maxLevel;
	int forestSize;
	int chainSize;
	int ensembleSize;
	int maxAttributes;

	size_t treesPerThread;
	size_t chainsPerThread;
	size_t instancesPerThread;

	size_t treesPerGroup;
	size_t chainsPerGroup;
	size_t instancesPerGroup;

	std::vector<std::vector<int>> partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc)
	{
		std::vector<std::vector<int>> indicesList;
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
						indicesList.push_back(std::vector<int>());
						for (int index = 0; index < data.getSize(); ++index)
						{
							indicesList.back().push_back(index);
						}
					}
				}
			}
		}
		return indicesList;
	}

	std::string buildOptionString()
	{
		std::stringstream strstr;
		strstr << "-D INSTANCES_PER_ITEM=" << instancesPerThread << " -D CHAINS_PER_ITEM=" << chainsPerThread << " -D TREES_PER_ITEM=" << treesPerThread << std::endl;
		return strstr.str();
	}

public:
	ECCExecutor(bool build, bool classify, int _maxLevel, int _maxAttributes, int _forestSize,
		int _instancesPerThread, int _chainsPerThread, int _treesPerThread,
		int _instancesPerGroup, int _chainsPerGroup, int _treesPerGroup)
		: nodeIndexBuffer(NULL), nodeValueBuffer(NULL), labelOrderBuffer(NULL), partitionInstance(true),
		maxLevel(_maxLevel), forestSize(_forestSize), maxAttributes(_maxAttributes), oldKernelMode(false),
		instancesPerThread(_instancesPerThread), chainsPerThread(_chainsPerThread), treesPerThread(_treesPerThread),
		instancesPerGroup(_instancesPerGroup), chainsPerGroup(_chainsPerGroup), treesPerGroup(_treesPerGroup)
	{
		if (build)
		{
			cl_program prog;
			PlatformUtil::buildProgramFromFile("eccBuild.cl", prog);//("\\\\X-THINK\\Users\\Public\\eccBuild.cl", prog);
			buildKernel = new Kernel(prog, "eccBuild");
			clReleaseProgram(prog);
		}

		if (classify)
		{
			cl_program prog;
			
			PlatformUtil::buildProgramFromFile(oldKernelMode ? "OldKernels/eccClassify.cl" : "eccClassify.cl", prog, buildOptionString().c_str());
			classifyKernel = new Kernel(prog, "eccClassify");
			clReleaseProgram(prog);
		}
	}

	void runBuild(ECCData& data, int forestsPerRun, int ensembleSize, int ensemblesPerRun, int ensembleSubSetSize, int forestSubSetSize)
	{
		std::cout << std::endl << "--- BUILD ---" << std::endl;
		if (data.getSize() == ensembleSubSetSize && data.getSize() == forestSubSetSize)
			partitionInstance = false;
		else
			partitionInstance = true;

		ecc = new EnsembleOfClassifierChains(data.getValueCount(), data.getLabelCount(), maxLevel, forestSize, ensembleSize, ensembleSubSetSize, forestSubSetSize);
		int globalSize = ecc->getChainSize() * ensemblesPerRun * forestsPerRun;
		int nodesLastLevel = pow(2.0f, maxLevel);
		int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;

		this->chainSize = ecc->getChainSize();
		this->ensembleSize = ecc->getEnsembleSize();

		labelOrderBuffer = Buffer(sizeof(int) * ecc->getEnsembleSize() * ecc->getChainSize());

		int i = 0;
		for (ClassifierChain chain : ecc->getChains())
		{
			for (int label : chain.getLabelOrder())
			{
				static_cast<int*>(labelOrderBuffer.data)[i++] = label;
			}
		}

		nodeValueBuffer = Buffer(sizeof(double) * ecc->getTotalSize());
		nodeIndexBuffer = Buffer(sizeof(int) * ecc->getTotalSize());
		Buffer tmpNodeValueBuffer(sizeof(double) * globalSize * nodesPerTree);
		Buffer tmpNodeIndexBuffer(sizeof(int) * globalSize * nodesPerTree);


		int dataSize = data.getSize();
		Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double));

		int dataBuffIdx = 0;
		for (MultilabelInstance inst : data.getInstances())
		{
			memcpy(static_cast<double*>(dataBuffer.data) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
			dataBuffIdx += inst.getValueCount();
		}

		int subSetSize = ecc->getForestSubSetSize();
		int numValues = data.getValueCount();
		int numAttributes = data.getAttribCount();

		int maxSplits = ecc->getForestSubSetSize() - 1;
		Buffer instancesBuffer(sizeof(int) * maxSplits*globalSize);

		int instBuffIdx = 0;
		std::vector<std::vector<int>> indicesList(partitionInstances(data, *ecc));
		for (std::vector<int>& indices : indicesList)
		{
			int intsAdded = 0;
			int lpos = 0;

			while (intsAdded < maxSplits)
			{
				if (lpos >= indices.size())
				{
					lpos = 0;
				}

				static_cast<int*>(instancesBuffer.data)[instBuffIdx++] = indices[lpos++];
				++intsAdded;
			}
		}

		Buffer instancesNextBuffer(sizeof(int) * maxSplits*globalSize);
		Buffer instancesLengthBuffer(sizeof(int) * nodesLastLevel*globalSize);
		Buffer instancesNextLengthBuffer(sizeof(int) * nodesLastLevel*globalSize);
		Buffer seedsBuffer(sizeof(int) * globalSize);
		Buffer voteBuffer(sizeof(int) * nodesLastLevel*globalSize);

		int gidMultiplier = 1;
		int ensembleRuns = ensembleSize / ensemblesPerRun;
		int forestRuns = forestSize / forestsPerRun;

		buildKernel->setDim(1);
		buildKernel->setGlobalSize(globalSize);
		buildKernel->setLocalSize(4);

		double totalTime = .0;

		buildKernel->SetInputArg(1, seedsBuffer.data, seedsBuffer.size);
		buildKernel->SetInputArg(2, dataBuffer.data, dataBuffer.size);
		buildKernel->SetInputArg(3, dataSize);
		buildKernel->SetInputArg(4, subSetSize);
		buildKernel->SetInputArg(5, labelOrderBuffer.data, labelOrderBuffer.size);
		buildKernel->SetInputArg(6, numValues);
		buildKernel->SetInputArg(7, numAttributes);
		buildKernel->SetInputArg(8, maxAttributes);
		buildKernel->SetInputArg(9, maxLevel);
		buildKernel->SetInputArg(10, chainSize);
		buildKernel->SetInputArg(11, maxSplits);
		buildKernel->SetInputArg(12, forestSize);
		buildKernel->SetInputOutputArg(13, instancesBuffer.data, instancesBuffer.size);
		buildKernel->SetInputOutputArg(14, instancesNextBuffer.data, instancesNextBuffer.size);
		buildKernel->SetInputOutputArg(15, instancesLengthBuffer.data, instancesLengthBuffer.size);
		buildKernel->SetInputOutputArg(16, instancesNextLengthBuffer.data, instancesNextLengthBuffer.size);
		tmpNodeValueBuffer.memObj = 
			buildKernel->SetInputOutputArg(17, tmpNodeValueBuffer.data, tmpNodeValueBuffer.size);
		tmpNodeIndexBuffer.memObj = 
			buildKernel->SetInputOutputArg(18, tmpNodeIndexBuffer.data, tmpNodeIndexBuffer.size);
		buildKernel->SetInputOutputArg(19, voteBuffer.data, voteBuffer.size);

		Util::StopWatch stopWatch;
		stopWatch.start();
		for (int chain = 0; chain < ensembleSize; chain += ensemblesPerRun)
		{
			for (int forest = 0; forest < ensembleSize; forest += ensemblesPerRun)
			{
				buildKernel->SetInputArg(0, &gidMultiplier, sizeof(gidMultiplier));

				for (int seed = 0; seed < globalSize; ++seed)
				{
					int rnd = Util::randomInt(INT_MAX);
					static_cast<int*>(seedsBuffer.data)[seed] = rnd;
				}

				//std::ofstream ofs("cmp2.txt");

				//ofs << gidMultiplier << "\n";
				//printBuffer<int>(seedsBuffer, ofs);
				//printBuffer<double>(dataBuffer, ofs);
				//ofs << dataSize << "\n";
				//ofs << subSetSize << "\n";
				//printBuffer<int>(labelOrderBuffer, ofs);
				//ofs << numValues << "\n";
				//ofs << numAttributes << "\n";
				//ofs << maxAttributes << "\n";
				//ofs << maxLevel << "\n";
				//ofs << chainSize << "\n";
				//ofs << maxSplits << "\n";
				//ofs << forestSize << "\n";
				//printBuffer<int>(instancesBuffer, ofs);
				//printBuffer<int>(instancesNextBuffer, ofs);
				//printBuffer<int>(instancesLengthBuffer, ofs);
				//printBuffer<int>(instancesNextLengthBuffer, ofs);
				//printBuffer<double>(tmpNodeValueBuffer, ofs);
				//printBuffer<int>(tmpNodeIndexBuffer, ofs);
				//printBuffer<int>(voteBuffer, ofs);
				//ofs.close();

				buildKernel->execute();
				totalTime += buildKernel->getRuntime();

				//std::ofstream ofs("cmp2.txt");

				//printBuffer<double>(tmpNodeValueBuffer, ofs);
				//printBuffer<int>(tmpNodeIndexBuffer, ofs);
				buildKernel->readResult(tmpNodeIndexBuffer.memObj, tmpNodeIndexBuffer.size, tmpNodeIndexBuffer.data);
				buildKernel->readResult(tmpNodeValueBuffer.memObj, tmpNodeValueBuffer.size, tmpNodeValueBuffer.data);

				memcpy(static_cast<uint8_t*>(nodeIndexBuffer.data) + (gidMultiplier - 1)*tmpNodeIndexBuffer.size, tmpNodeIndexBuffer.data, tmpNodeIndexBuffer.size);
				memcpy(static_cast<uint8_t*>(nodeValueBuffer.data) + (gidMultiplier - 1)*tmpNodeIndexBuffer.size, tmpNodeValueBuffer.data, tmpNodeValueBuffer.size);

				++gidMultiplier;
			}
		}
		std::cout << "Build took " << stopWatch.stop() << " ms total." << std::endl;
		std::cout << "Build took " << totalTime << " ms kernel time." << std::endl;

		tmpNodeIndexBuffer.clear();
		tmpNodeValueBuffer.clear();
		dataBuffer.clear();
		instancesBuffer.clear();
		instancesLengthBuffer.clear();
		instancesNextBuffer.clear();
		instancesNextLengthBuffer.clear();
		seedsBuffer.clear();
		voteBuffer.clear();
	}

	template<typename T>
	void printBuffer(Buffer buff, std::ofstream& ofs)
	{
		int len = buff.size / sizeof(T);

		for (int n = 0; n < len; ++n)
		{
			ofs << static_cast<T*>(buff.data)[n] << "\n";
		}
	}

	void runClassify(ECCData& data, std::vector<double>& values, std::vector<int>& votes)
	{
		runClassifyOld(data, values, votes);
		runClassifyNew(data, values, votes);
		//oldKernelMode ? runClassifyOld(data, values, votes) : runClassifyNew(data, values, votes);
	}

private:
	int* votesCmp;
	double* resultCmp;

	void validateResults(const ECCData& data, const Buffer& voteBuffer, const Buffer& resultBuffer)
	{
		bool correct = true;
		for (int inst = 0; inst < data.getSize() && correct; ++inst)
		{
			for (int label = 0; label < data.getLabelCount() && correct; ++label)
			{
				int* vote = static_cast<int*>(voteBuffer.data) + (inst * data.getLabelCount() + label);
				double* result = static_cast<double*>(resultBuffer.data) + (inst * data.getLabelCount() + label);

				if (abs(*vote - votesCmp[inst * data.getLabelCount() + label]) > 1e-06)
				{
					correct = false;
				}

				if (abs(*result - resultCmp[inst * data.getLabelCount() + label]) > 1e-06)
				{
					correct = false;
				}
			}
		}

		if (!correct)
			std::cout << "Results not correct!" << std::endl;
	}

	void runClassifyNew(ECCData& data, std::vector<double>& values, std::vector<int>& votes)
	{
		std::cout << std::endl << "--- NEW CLASSIFICATION ---" << std::endl;
		cl_program prog;
		PlatformUtil::buildProgramFromFile("eccClassify.cl", prog, buildOptionString().c_str());
		classifyKernel = new Kernel(prog, "eccClassify");
		clReleaseProgram(prog);

		PlatformUtil::buildProgramFromFile("eccClassifyReduce.cl", prog, buildOptionString().c_str());
		Kernel* classifyReduceKernel = new Kernel(prog, "eccClassifyReduce");
		clReleaseProgram(prog);

		int dataSize = data.getSize();
		Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double));

		int dataBuffIdx = 0;
		for (MultilabelInstance inst : data.getInstances())
		{
			memcpy(static_cast<double*>(dataBuffer.data) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
			dataBuffIdx += inst.getValueCount();
		}

		Buffer resultBuffer(dataSize * data.getLabelCount() * sizeof(double));
		Buffer voteBuffer(dataSize * data.getLabelCount() * sizeof(int));

		int numValues = data.getValueCount();
		
		Buffer labelBuffer(data.getSize() * data.getLabelCount() * ensembleSize * sizeof(double));

		int localBufferSize[3] = { instancesPerGroup, chainsPerGroup, treesPerGroup };
		int localBufferTotalSize = localBufferSize[0] * localBufferSize[1] * localBufferSize[2];

		classifyKernel->SetInputArg(0, nodeValueBuffer.data, nodeValueBuffer.size);
		classifyKernel->SetInputArg(1, nodeIndexBuffer.data, nodeIndexBuffer.size);
		classifyKernel->SetInputArg(2, labelOrderBuffer.data, labelOrderBuffer.size);
		classifyKernel->SetInputArg(3, maxLevel);
		classifyKernel->SetInputArg(4, forestSize);
		classifyKernel->SetInputArg(5, chainSize);
		classifyKernel->SetInputArg(6, ensembleSize);
		classifyKernel->SetInputArg(7, dataBuffer.data, dataBuffer.size);
		classifyKernel->SetInputArg(8, numValues);
		classifyKernel->SetOutputArg(10, resultBuffer.data, resultBuffer.size);
		classifyKernel->SetOutputArg(11, voteBuffer.data, voteBuffer.size);
		classifyKernel->SetInputArg(12, labelBuffer.data, labelBuffer.size);

		classifyKernel->setDim(3);
		classifyKernel->setGlobalSize(data.getSize() / instancesPerThread, ensembleSize / chainsPerThread, forestSize / treesPerThread);
		classifyKernel->setLocalSize(1, ensembleSize / chainsPerThread, forestSize / treesPerThread);

		classifyKernel->execute();
	
		for (int chainIndex = 0; chainIndex < chainSize; ++chainIndex)
		{
			classifyKernel->SetInputArg(9, chainIndex);
			classifyKernel->SetInputArg(13, NULL, localBufferTotalSize * sizeof(double));
			classifyKernel->SetInputArg(14, NULL, localBufferTotalSize * sizeof(int));
			classifyKernel->SetInputArg(15, localBufferSize, sizeof(int) * 3);
		}
		std::cout << "Classification kernel took " << classifyKernel->getRuntime() << " ms." << std::endl;
		
		classifyKernel->readResult(voteBuffer.memObj, voteBuffer.size, voteBuffer.data);
		classifyKernel->readResult(resultBuffer.memObj, resultBuffer.size, resultBuffer.data);
		validateResults(data, voteBuffer, resultBuffer);

		resultBuffer.clear();
		voteBuffer.clear();
	}

	void runClassifyOld(ECCData& data, std::vector<double>& values, std::vector<int>& votes)
	{
		std::cout << std::endl << "--- OLD CLASSIFICATION ---" << std::endl;
		cl_program prog;
		PlatformUtil::buildProgramFromFile("OldKernels/eccClassify.cl", prog);
		classifyKernel = new Kernel(prog, "eccClassify");
		clReleaseProgram(prog);

		int dataSize = data.getSize();
		Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double));

		int dataBuffIdx = 0;
		for (MultilabelInstance inst : data.getInstances())
		{
			memcpy(static_cast<double*>(dataBuffer.data) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
			dataBuffIdx += inst.getValueCount();
		}

		Buffer resultBuffer(dataSize * data.getLabelCount() * sizeof(double));
		Buffer voteBuffer(dataSize * data.getLabelCount() * sizeof(int));

		int numValues = data.getValueCount();

		classifyKernel->SetInputArg(0, nodeValueBuffer.data, nodeValueBuffer.size);
		classifyKernel->SetInputArg(1, nodeIndexBuffer.data, nodeIndexBuffer.size);
		classifyKernel->SetInputArg(2, labelOrderBuffer.data, labelOrderBuffer.size);
		classifyKernel->SetInputArg(3, maxLevel);
		classifyKernel->SetInputArg(4, forestSize);
		classifyKernel->SetInputArg(5, chainSize);
		classifyKernel->SetInputArg(6, ensembleSize);
		classifyKernel->SetInputArg(7, dataBuffer.data, dataBuffer.size);
		classifyKernel->SetInputArg(8, numValues);
		resultBuffer.memObj = classifyKernel->SetOutputArg(9, resultBuffer.data, resultBuffer.size);
		voteBuffer.memObj = classifyKernel->SetOutputArg(10, voteBuffer.data, voteBuffer.size);

		classifyKernel->setDim(1);
		classifyKernel->setGlobalSize(data.getSize());
		classifyKernel->setLocalSize(1);

		classifyKernel->execute();
		std::cout << "Classification kernel took " << classifyKernel->getRuntime() << " ms." << std::endl;

		votesCmp = new int[dataSize * data.getLabelCount()];
		resultCmp = new double[dataSize * data.getLabelCount()];

		classifyKernel->readResult(resultBuffer.memObj, resultBuffer.size, resultBuffer.data);
		classifyKernel->readResult(voteBuffer.memObj, voteBuffer.size, voteBuffer.data);

		memcpy(votesCmp, voteBuffer.data, voteBuffer.size);
		memcpy(resultCmp, resultBuffer.data, resultBuffer.size);

		resultBuffer.clear();
		voteBuffer.clear();
	}

public:
	~ECCExecutor()
	{
		nodeIndexBuffer.clear();
		nodeValueBuffer.clear();
		labelOrderBuffer.clear();
		delete classifyKernel;
		delete buildKernel;
	}
};

