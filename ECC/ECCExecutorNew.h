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

class ECCExecutorNew
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

	size_t newLoopStepTime;
	size_t newLoopFinalTime;
	size_t newRemainStepTime;
	size_t newRemainFinalTime;
	size_t newKernelTime;
	size_t newCPUTime;

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

	static std::string buildOptionString(int numChains, int numInstances, int numTrees, int numAttributes, int numLabels, int maxLevel, int nodesPerTree)
	{
		std::stringstream strstr;
		strstr << " -D NUM_CHAINS=" << numChains
			<< " -D NUM_INSTANCES=" << numInstances
			<< " -D NUM_TREES=" << numTrees
			<< " -D NUM_LABELS=" << numLabels
			<< " -D NUM_ATTRIBUTES=" << numAttributes
			<< " -D MAX_LEVEL=" << maxLevel
			<< " -D NODES_PER_TREE=" << nodesPerTree

			<< " -D NUM_WG_CHAINS_SC=" << NUM_WG_CHAINS_SC
			<< " -D NUM_WG_INSTANCES_SC=" << NUM_WG_INSTANCES_SC
			<< " -D NUM_WG_TREES_SC=" << NUM_WG_TREES_SC

			<< " -D NUM_WG_CHAINS_SR=" << NUM_WG_CHAINS_SR
			<< " -D NUM_WG_INSTANCES_SR=" << NUM_WG_INSTANCES_SR

			<< " -D NUM_WG_LABELS_FC=" << NUM_WG_LABELS_FC
			<< " -D NUM_WG_CHAINS_FC=" << NUM_WG_CHAINS_FC
			<< " -D NUM_WG_INSTANCES_FC=" << NUM_WG_INSTANCES_FC

			<< " -D NUM_WG_LABELS_FR=" << NUM_WG_LABELS_FR
			<< " -D NUM_WG_INSTANCES_FR=" << NUM_WG_INSTANCES_FR

			<< " -D NUM_WI_CHAINS_SC=" << NUM_WI_CHAINS_SC
			<< " -D NUM_WI_INSTANCES_SC=" << NUM_WI_INSTANCES_SC
			<< " -D NUM_WI_TREES_SC=" << NUM_WI_TREES_SC

			<< " -D NUM_WI_CHAINS_SR=" << NUM_WI_CHAINS_SR
			<< " -D NUM_WI_INSTANCES_SR=" << NUM_WI_INSTANCES_SR
			<< " -D NUM_WI_TREES_SR=" << NUM_WI_TREES_SR

			<< " -D NUM_WI_LABELS_FC=" << NUM_WI_LABELS_FC
			<< " -D NUM_WI_INSTANCES_FC=" << NUM_WI_INSTANCES_FC
			<< " -D NUM_WI_CHAINS_FC=" << NUM_WI_CHAINS_FC

			<< " -D NUM_WI_LABELS_FR=" << NUM_WI_LABELS_FR
			<< " -D NUM_WI_INSTANCES_FR=" << NUM_WI_INSTANCES_FR
			<< " -D NUM_WI_CHAINS_FR=" << NUM_WI_CHAINS_FR;
		return strstr.str();
	}

public:
	ECCExecutorNew(int _maxLevel, int _maxAttributes, int _forestSize)
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
		PlatformUtil::buildProgramFromFile("eccBuildNew.cl", prog);//("\\\\X-THINK\\Users\\Public\\eccBuild.cl", prog);
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

		int* nodeIndicesTransp = new int[ecc->getTotalSize()];
		double* nodeValuesTransp = new double[ecc->getTotalSize()];
		size_t forestDataSize = forestSize * nodesPerTree;
		for (int chain = 0; chain < ecc->getEnsembleSize(); ++chain)
		{
			for (int forest = 0; forest < ecc->getChainSize(); ++forest)
			{
				memcpy(nodeIndicesTransp + (forest * ensembleSize + chain) * forestDataSize,
					nodeIndices + (chain * chainSize + forest) * forestDataSize,
					forestDataSize * sizeof(int));

				memcpy(nodeValuesTransp + (forest * ensembleSize + chain) * forestDataSize,
					nodeValues + (chain * chainSize + forest) * forestDataSize,
					forestDataSize * sizeof(double));
			}
		}

		delete[] nodeValues;
		delete[] nodeIndices;
		nodeValues = nodeValuesTransp;
		nodeIndices = nodeIndicesTransp;

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

private:
	typedef struct OutputAtom
	{
		double result;
		int vote;
	}OutputAtom;

public:
	void runClassifyNew(ECCData& data, std::vector<double>& values, std::vector<int>& votes, std::map<std::string, int>& params)
	{
		std::cout << std::endl << "--- NEW CLASSIFICATION ---" << std::endl;
		std::string optionString;
		std::stringstream strstr;
		for (auto it = params.begin(); it != params.end(); ++it)
			strstr << " -D " << it->first << "=" << it->second;
		optionString = strstr.str();

		std::cout << optionString << std::endl;

		cl_program prog;
		PlatformUtil::buildProgramFromFile("stepCalcKernel.cl", prog, optionString.c_str());
		Kernel* stepCalcKernel = new Kernel(prog, "stepCalc");
		clReleaseProgram(prog);

		PlatformUtil::buildProgramFromFile("stepReduceKernel.cl", prog, optionString.c_str());
		Kernel* stepReduceKernel = new Kernel(prog, "stepReduce");
		clReleaseProgram(prog);

		PlatformUtil::buildProgramFromFile("finalCalcKernel.cl", prog, optionString.c_str());
		Kernel* finalCalcKernel = new Kernel(prog, "finalCalc");
		clReleaseProgram(prog);

		PlatformUtil::buildProgramFromFile("finalReduceKernel.cl", prog, optionString.c_str());
		Kernel* finalReduceKernel = new Kernel(prog, "finalReduce");
		clReleaseProgram(prog);

		Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

		int dataBuffIdx = 0;
		for (MultilabelInstance inst : data.getInstances())
		{
			memcpy(((double*)dataBuffer.getData()) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
			dataBuffIdx += inst.getValueCount();
		}

		Buffer resultBuffer(data.getSize() * data.getLabelCount() * sizeof(OutputAtom), CL_MEM_WRITE_ONLY);
		Buffer labelBuffer(data.getSize() * data.getLabelCount() * ensembleSize * sizeof(OutputAtom), CL_MEM_WRITE_ONLY);

		int stepIntermediateBufferSize[3] = { data.getSize(), ensembleSize, params["NUM_WG_TREES_SC"] };
		int stepIntermediateBufferTotalSize = stepIntermediateBufferSize[0] * stepIntermediateBufferSize[1] * stepIntermediateBufferSize[2];

		int finalIntermediateBufferSize[3] = { data.getSize(), data.getLabelCount(), params["NUM_WG_CHAINS_FC"] };
		int finalIntermediateBufferTotalSize = finalIntermediateBufferSize[0] * finalIntermediateBufferSize[1] * finalIntermediateBufferSize[2];

		int localBufferSize_SC = params["NUM_WI_INSTANCES_SC"] * params["NUM_WI_CHAINS_SC"] * params["NUM_WI_TREES_SC"];
		int localBufferSize_SR = params["NUM_WI_INSTANCES_SR"] * params["NUM_WI_CHAINS_SR"] * params["NUM_WI_TREES_SR"];
		int localBufferSize_FC = params["NUM_WI_INSTANCES_FC"] * params["NUM_WI_LABELS_FC"] * params["NUM_WI_CHAINS_FC"];
		int localBufferSize_FR = params["NUM_WI_INSTANCES_FR"] * params["NUM_WI_LABELS_FR"] * params["NUM_WI_CHAINS_FR"];

		Buffer stepIntermediateBuffer(stepIntermediateBufferTotalSize * sizeof(OutputAtom), CL_MEM_READ_WRITE);
		Buffer finalIntermediateBuffer(finalIntermediateBufferTotalSize * sizeof(OutputAtom), CL_MEM_READ_WRITE);

		size_t stepModelSize = ecc->getTotalSize() / chainSize;
		Buffer stepNodeValueBuffer(sizeof(double) * stepModelSize, CL_MEM_READ_ONLY);
		Buffer stepNodeIndexBuffer(sizeof(int) * stepModelSize, CL_MEM_READ_ONLY);

		stepCalcKernel->SetArg(0, stepNodeValueBuffer, true);
		stepCalcKernel->SetArg(1, stepNodeIndexBuffer, true);
		stepCalcKernel->SetArg(2, dataBuffer, true);
		stepCalcKernel->SetArg(3, labelBuffer, true);
		stepCalcKernel->SetLocalArg(4, localBufferSize_SC * sizeof(OutputAtom));
		stepCalcKernel->SetArg(5, stepIntermediateBuffer);

		stepCalcKernel->setDim(3);
		stepCalcKernel->setGlobalSize(params["NUM_WG_INSTANCES_SC"] * params["NUM_WI_INSTANCES_SC"], params["NUM_WG_CHAINS_SC"] * params["NUM_WI_CHAINS_SC"], params["NUM_WG_TREES_SC"] * params["NUM_WI_TREES_SC"]);
		stepCalcKernel->setLocalSize(params["NUM_WI_INSTANCES_SC"], params["NUM_WI_CHAINS_SC"], params["NUM_WI_TREES_SC"]);

		stepReduceKernel->SetArg(0, stepIntermediateBuffer);
		stepReduceKernel->SetArg(1, labelBuffer);
		stepReduceKernel->SetArg(2, labelOrderBuffer, true);
		stepReduceKernel->SetLocalArg(3, localBufferSize_SR * sizeof(OutputAtom));

		stepReduceKernel->setDim(3);
		stepReduceKernel->setGlobalSize(params["NUM_WG_INSTANCES_SR"] * params["NUM_WI_INSTANCES_SR"], params["NUM_WG_CHAINS_SR"] * params["NUM_WI_CHAINS_SR"], params["NUM_WI_TREES_SR"]);
		stepReduceKernel->setLocalSize(params["NUM_WI_INSTANCES_SR"], params["NUM_WI_CHAINS_SR"], params["NUM_WI_TREES_SR"]);

		finalCalcKernel->SetArg(0, labelBuffer);
		finalCalcKernel->SetLocalArg(1, localBufferSize_FC * sizeof(OutputAtom));
		finalCalcKernel->SetArg(2, finalIntermediateBuffer);

		finalCalcKernel->setDim(3);
		finalCalcKernel->setGlobalSize(params["NUM_WG_INSTANCES_FC"] * params["NUM_WI_INSTANCES_FC"], params["NUM_WG_LABELS_FC"] * params["NUM_WI_LABELS_FC"], params["NUM_WG_CHAINS_FC"] * params["NUM_WI_CHAINS_FC"]);
		finalCalcKernel->setLocalSize(params["NUM_WI_INSTANCES_FC"], params["NUM_WI_LABELS_FC"], params["NUM_WI_CHAINS_FC"]);

		finalReduceKernel->SetArg(0, finalIntermediateBuffer);
		finalReduceKernel->SetArg(1, resultBuffer);
		finalReduceKernel->SetLocalArg(2, localBufferSize_FR * sizeof(OutputAtom));

		finalReduceKernel->setDim(3);
		finalReduceKernel->setGlobalSize(params["NUM_WG_INSTANCES_FR"] * params["NUM_WI_INSTANCES_FR"], params["NUM_WG_LABELS_FR"] * params["NUM_WI_LABELS_FR"], params["NUM_WI_CHAINS_FR"]);
		finalReduceKernel->setLocalSize(params["NUM_WI_INSTANCES_FR"], params["NUM_WI_LABELS_FR"], params["NUM_WI_CHAINS_FR"]);

		bool sizesok = stepCalcKernel->verifyWorkgroupSize();
		sizesok = sizesok && stepReduceKernel->verifyWorkgroupSize();
		sizesok = sizesok && finalCalcKernel->verifyWorkgroupSize();
		sizesok = sizesok && finalReduceKernel->verifyWorkgroupSize();

		if (!sizesok)
		{
			std::cout << "Workgroup too large" << std::endl;
			newCPUTime = newKernelTime = newLoopFinalTime = newLoopStepTime = newRemainFinalTime = newRemainStepTime = std::numeric_limits<int>::max();

			for (int n = 0; n < data.getLabelCount()*data.getSize(); ++n)
			{
				values.push_back(0.0);
				votes.push_back(0);
			}

			dataBuffer.clear();
			resultBuffer.clear();
			labelBuffer.clear();
			stepIntermediateBuffer.clear();
			finalIntermediateBuffer.clear();
			stepNodeValueBuffer.clear();
			stepNodeIndexBuffer.clear();

			delete stepCalcKernel;
			delete stepReduceKernel;
			delete finalCalcKernel;
			delete finalReduceKernel;

			return;
		}

		double SCTime = 0.0, SRTime = 0.0, FCTime = 0.0, FRTime = 0.0;
		Util::StopWatch stopWatch;
		stopWatch.start();

		for (int chainIndex = 0; chainIndex < chainSize; ++chainIndex)
		{
			stepReduceKernel->SetArg(4, chainIndex);
			stepNodeIndexBuffer.writeFrom(nodeIndices + chainIndex * stepModelSize, stepModelSize * sizeof(int));
			stepNodeValueBuffer.writeFrom(nodeValues + chainIndex * stepModelSize, stepModelSize * sizeof(double));
			stepCalcKernel->execute();
			stepReduceKernel->execute();
			SCTime += stepCalcKernel->getRuntime();
			SRTime += stepReduceKernel->getRuntime();
		}
		finalCalcKernel->execute();
		finalReduceKernel->execute();

		FCTime += finalCalcKernel->getRuntime();
		FRTime += finalReduceKernel->getRuntime();
		resultBuffer.read();

		newCPUTime = stopWatch.stop();
		newKernelTime = SCTime + SRTime + FCTime + FRTime;
		newLoopStepTime = SCTime + SRTime;
		newLoopFinalTime = FCTime + FRTime;
		std::cout << "Classification kernel took " << ((double)newKernelTime * 1e-06) << " ms."
			<< "\n\tstepCalc: " << ((double)SCTime * 1e-06)
			<< "\n\tstepReduce: " << ((double)SRTime * 1e-06)
			<< "\n\tfinalCalc: " << ((double)FCTime * 1e-06)
			<< "\n\tfinalReduce: " << ((double)FRTime * 1e-06)
			<< std::endl;
		std::cout << "Total time: " << ((double)newCPUTime*1e-06) << std::endl;

		PlatformUtil::finish();

		bool all0 = true;
		for (int n = 0; n < data.getLabelCount()*data.getSize(); ++n)
		{
			values.push_back(static_cast<OutputAtom*>(resultBuffer.getData())[n].result);
			votes.push_back(static_cast<OutputAtom*>(resultBuffer.getData())[n].vote);
			if (votes[n] != 0) all0 = false;
		}

		std::cout << "all0: " << (all0 ? "true" : "false") << std::endl;

		dataBuffer.clear();
		resultBuffer.clear();
		labelBuffer.clear();
		stepIntermediateBuffer.clear();
		finalIntermediateBuffer.clear();
		stepNodeValueBuffer.clear();
		stepNodeIndexBuffer.clear();

		delete stepCalcKernel;
		delete stepReduceKernel;
		delete finalCalcKernel;
		delete finalReduceKernel;
	}

	size_t getNewLoopStepTime()
	{
		return newLoopStepTime;
	}

	size_t getNewLoopFinalTime()
	{
		return newLoopFinalTime;
	}

	size_t getNewRemainStepTime()
	{
		return newRemainStepTime;
	}

	size_t getNewRemainFinalTime()
	{
		return newRemainFinalTime;
	}

	size_t getNewKernelTime()
	{
		return newKernelTime;
	}

	size_t getNewCPUTime()
	{
		return newCPUTime;
	}

	~ECCExecutorNew()
	{
		delete[] nodeIndices;
		delete[] nodeValues;
		labelOrderBuffer.clear();
		delete ecc;
	}
};

