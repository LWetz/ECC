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

	long time;

	double* nodeValues;
	int* nodeIndices;
	Buffer labelOrderBuffer;
	int maxLevel;
	int numTrees;
	int numLabels;
	int numChains;
	int maxAttributes;
	int numAttributes;

	std::string buildSource;
	std::string stepCalcSource;
	std::string stepReduceSource;
	std::string finalCalcSource;
	std::string finalReduceSource;

	size_t stepTime;
	size_t finalTime;
	size_t kernelTime;
	size_t CPUTime;

	std::vector<int> partitionInstances(ECCData& data, EnsembleOfClassifierChains& ecc)
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

	struct BuildData
	{
		Buffer tmpNodeIndexBuffer;
		Buffer tmpNodeValueBuffer;
		Buffer dataBuffer;
		Buffer instancesBuffer;
		Buffer instancesLengthBuffer;
		Buffer instancesNextBuffer;
		Buffer instancesNextLengthBuffer;
		Buffer seedsBuffer;

		int numTrees;
		int numInstances;
	};

	BuildData *buildData;

	struct ClassifyData
	{
		Buffer dataBuffer;
		Buffer resultBuffer;
		Buffer labelBuffer;
		Buffer stepNodeValueBuffer;
		Buffer stepNodeIndexBuffer;

		int numInstances;
	};

	ClassifyData *classifyData;

public:
	ECCExecutorNew(int _maxLevel, int _maxAttributes, int _numAttributes, int _numTrees, int _numLabels, int _numChains, int _ensembleSubSetSize, int _forestSubSetSize)
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
		ecc = new EnsembleOfClassifierChains(_numAttributes + numLabels, numLabels, maxLevel, numTrees, numChains, _ensembleSubSetSize, _forestSubSetSize);
	
		labelOrderBuffer = Buffer(sizeof(int) * ecc->getEnsembleSize() * ecc->getChainSize(), CL_MEM_READ_ONLY);
		labelOrderBuffer.write();

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
	}

	void prepareBuild(ECCData& data, int treesPerRun)
	{
		std::cout << std::endl << "--- BUILD ---" << std::endl;

		buildData = new BuildData;

		int nodesLastLevel = pow(2.0f, maxLevel);
		int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;

		int maxSplits = ecc->getForestSubSetSize() - 1;

		Buffer tmpNodeValueBuffer(sizeof(double) * treesPerRun * nodesPerTree, CL_MEM_READ_WRITE);
		Buffer tmpNodeIndexBuffer(sizeof(int) * treesPerRun * nodesPerTree, CL_MEM_READ_WRITE);

		Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

		int dataBuffIdx = 0;
		for (MultilabelInstance inst : data.getInstances())
		{
			memcpy(static_cast<double*>(dataBuffer.getData()) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
			dataBuffIdx += inst.getValueCount();
		}

		int numValues = data.getValueCount();
		int numAttributes = data.getAttribCount();

		Buffer instancesBuffer(sizeof(int) * maxSplits*treesPerRun, CL_MEM_READ_WRITE);

		std::vector<int> indicesList(partitionInstances(data, *ecc));

		Buffer instancesNextBuffer(sizeof(int) * maxSplits*treesPerRun, CL_MEM_READ_WRITE);
		Buffer instancesLengthBuffer(sizeof(int) * nodesLastLevel*treesPerRun, CL_MEM_READ_WRITE);
		Buffer instancesNextLengthBuffer(sizeof(int) * nodesLastLevel*treesPerRun, CL_MEM_READ_WRITE);
		Buffer seedsBuffer(sizeof(int) * treesPerRun, CL_MEM_READ_ONLY);

		double totalTime = .0;

		for (int seed = 0; seed < treesPerRun; ++seed)
		{
			int rnd = Util::randomInt(INT_MAX);
			static_cast<int*>(seedsBuffer.getData())[seed] = rnd;
		}
		seedsBuffer.write();

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

	double tuneBuild(int workitems, int workgroups)
	{
		int nodesLastLevel = pow(2.0f, maxLevel);
		int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;
		int maxSplits = ecc->getForestSubSetSize() - 1;

		std::map<std::string, int> params;
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
		Kernel* buildKernel = new Kernel(prog, "eccBuild");
		clReleaseProgram(prog);

		buildKernel->setDim(1);
		buildKernel->setGlobalSize(workgroups * workitems);
		buildKernel->setLocalSize(workitems);

		buildKernel->SetArg(0, buildData->numTrees);
		buildKernel->SetArg(1, buildData->seedsBuffer);
		buildKernel->SetArg(2, buildData->dataBuffer, true);
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

	void finishBuild()
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

	void runBuild(ECCData& data, int treesPerRun, int workitems, int workgroups)
	{
		std::cout << std::endl << "--- BUILD ---" << std::endl;

		int nodesLastLevel = pow(2.0f, maxLevel);
		int nodesPerTree = pow(2.0f, maxLevel + 1) - 1;

		int maxSplits = ecc->getForestSubSetSize() - 1;

		std::map<std::string, int> params;
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

		std::string optionString;
		std::stringstream strstr;
		for (auto it = params.begin(); it != params.end(); ++it)
			strstr << " -D " << it->first << "=" << it->second;
		optionString = strstr.str();

		cl_program prog;
		PlatformUtil::buildProgramFromSource(buildSource, prog, optionString);//("\\\\X-THINK\\Users\\Public\\eccBuild.cl", prog);
		Kernel* buildKernel = new Kernel(prog, "eccBuild");
		clReleaseProgram(prog);

		labelOrderBuffer = Buffer(sizeof(int) * ecc->getEnsembleSize() * ecc->getChainSize(), CL_MEM_READ_ONLY);

		int i = 0;
		for (int chain = 0; chain < ecc->getEnsembleSize(); ++chain)
		{
			for (int forest = 0; forest < ecc->getChainSize(); ++forest)
			{
				static_cast<int*>(labelOrderBuffer.getData())[i++] = ecc->getChains()[chain].getLabelOrder()[forest];
			}
		}

		Buffer tmpNodeValueBuffer(sizeof(double) * treesPerRun * nodesPerTree, CL_MEM_READ_WRITE);
		Buffer tmpNodeIndexBuffer(sizeof(int) * treesPerRun * nodesPerTree, CL_MEM_READ_WRITE);

		Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

		int dataBuffIdx = 0;
		for (MultilabelInstance inst : data.getInstances())
		{
			memcpy(static_cast<double*>(dataBuffer.getData()) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
			dataBuffIdx += inst.getValueCount();
		}

		int numValues = data.getValueCount();
		int numAttributes = data.getAttribCount();

		Buffer instancesBuffer(sizeof(int) * maxSplits*treesPerRun, CL_MEM_READ_WRITE);

		std::vector<int> indicesList(partitionInstances(data, *ecc));

		Buffer instancesNextBuffer(sizeof(int) * maxSplits*treesPerRun, CL_MEM_READ_WRITE);
		Buffer instancesLengthBuffer(sizeof(int) * nodesLastLevel*treesPerRun, CL_MEM_READ_WRITE);
		Buffer instancesNextLengthBuffer(sizeof(int) * nodesLastLevel*treesPerRun, CL_MEM_READ_WRITE);
		Buffer seedsBuffer(sizeof(int) * treesPerRun, CL_MEM_READ_ONLY);

		int gidMultiplier = 0;

		buildKernel->setDim(1);
		buildKernel->setGlobalSize(workgroups * workitems);
		buildKernel->setLocalSize(workitems);

		double totalTime = .0;

		buildKernel->SetArg(1, seedsBuffer);
		buildKernel->SetArg(2, dataBuffer, true);
		buildKernel->SetArg(3, labelOrderBuffer, true);
		buildKernel->SetArg(4, instancesBuffer);
		buildKernel->SetArg(5, instancesNextBuffer);
		buildKernel->SetArg(6, instancesLengthBuffer);
		buildKernel->SetArg(7, instancesNextLengthBuffer);
		buildKernel->SetArg(8, tmpNodeValueBuffer);
		buildKernel->SetArg(9, tmpNodeIndexBuffer);

		Util::StopWatch stopWatch;
		stopWatch.start();

		for (int tree = 0; tree < numChains * numTrees * numLabels; tree += treesPerRun)
		{
			buildKernel->SetArg(0, gidMultiplier * treesPerRun);

			for (int seed = 0; seed < treesPerRun; ++seed)
			{
				int rnd = Util::randomInt(INT_MAX);
				static_cast<int*>(seedsBuffer.getData())[seed] = rnd;
			}
			seedsBuffer.write();

			instancesBuffer.writeFrom(indicesList.data() + gidMultiplier * treesPerRun * maxSplits, treesPerRun * maxSplits * sizeof(int));

			buildKernel->execute();
			totalTime += buildKernel->getRuntime();

			tmpNodeIndexBuffer.readTo(nodeIndices + gidMultiplier*treesPerRun*nodesPerTree, treesPerRun*nodesPerTree * sizeof(int));
			tmpNodeValueBuffer.readTo(nodeValues + gidMultiplier*treesPerRun*nodesPerTree, treesPerRun*nodesPerTree * sizeof(double));
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
	void prepareClassify(ECCData& data)
	{
		std::cout << std::endl << "--- NEW CLASSIFICATION ---" << std::endl;

		classifyData = new ClassifyData;

		Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

		int dataBuffIdx = 0;
		for (MultilabelInstance inst : data.getInstances())
		{
			memcpy(((double*)dataBuffer.getData()) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
			dataBuffIdx += inst.getValueCount();
		}

		Buffer resultBuffer(data.getSize() * data.getLabelCount() * sizeof(OutputAtom), CL_MEM_WRITE_ONLY);
		Buffer labelBuffer(data.getSize() * data.getLabelCount() * numChains * sizeof(OutputAtom), CL_MEM_WRITE_ONLY);

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

	double tuneClassifyStep(std::map<std::string, int> config, int oneStep = false)
	{
		config["NUM_INSTANCES"] = classifyData->numInstances;
		config["NUM_LABELS"] = numLabels;
		config["NUM_ATTRIBUTES"] = numAttributes;
		config["NUM_CHAINS"] = numChains;
		config["NUM_TREES"] = numTrees;
		config["MAX_LEVEL"] = maxLevel;
		config["NODES_PER_TREE"] = pow(2.0f, maxLevel + 1) - 1;

		std::string optionString;
		std::stringstream strstr;
		for (auto it = config.begin(); it != config.end(); ++it)
			strstr << " -D " << it->first << "=" << it->second;
		optionString = strstr.str();

		cl_program prog;
		PlatformUtil::buildProgramFromSource(stepCalcSource, prog, optionString.c_str());
		Kernel* stepCalcKernel = new Kernel(prog, "stepCalc");
		clReleaseProgram(prog);

		PlatformUtil::buildProgramFromSource(stepReduceSource, prog, optionString.c_str());
		Kernel* stepReduceKernel = new Kernel(prog, "stepReduce");
		clReleaseProgram(prog);

		int stepIntermediateBufferSize[3] = { classifyData->numInstances, numChains, config["NUM_WG_TREES_SC"] };
		int stepIntermediateBufferTotalSize = stepIntermediateBufferSize[0] * stepIntermediateBufferSize[1] * stepIntermediateBufferSize[2];

		int localBufferSize_SC = config["NUM_WI_INSTANCES_SC"] * config["NUM_WI_CHAINS_SC"] * config["NUM_WI_TREES_SC"];
		int localBufferSize_SR = config["NUM_WI_INSTANCES_SR"] * config["NUM_WI_CHAINS_SR"] * config["NUM_WI_TREES_SR"];

		Buffer stepIntermediateBuffer(stepIntermediateBufferTotalSize * sizeof(OutputAtom), CL_MEM_READ_WRITE);
		
		stepCalcKernel->SetArg(0, classifyData->stepNodeValueBuffer, true);
		stepCalcKernel->SetArg(1, classifyData->stepNodeIndexBuffer, true);
		stepCalcKernel->SetArg(2, classifyData->dataBuffer, true);
		stepCalcKernel->SetArg(3, classifyData->labelBuffer, true);
		stepCalcKernel->SetLocalArg(4, localBufferSize_SC * sizeof(OutputAtom));
		stepCalcKernel->SetArg(5, stepIntermediateBuffer);

		stepCalcKernel->setDim(3);
		stepCalcKernel->setGlobalSize(config["NUM_WG_INSTANCES_SC"] * config["NUM_WI_INSTANCES_SC"], config["NUM_WG_CHAINS_SC"] * config["NUM_WI_CHAINS_SC"], config["NUM_WG_TREES_SC"] * config["NUM_WI_TREES_SC"]);
		stepCalcKernel->setLocalSize(config["NUM_WI_INSTANCES_SC"], config["NUM_WI_CHAINS_SC"], config["NUM_WI_TREES_SC"]);

		stepReduceKernel->SetArg(0, stepIntermediateBuffer);
		stepReduceKernel->SetArg(1, classifyData->labelBuffer);
		stepReduceKernel->SetArg(2, labelOrderBuffer, true);
		stepReduceKernel->SetLocalArg(3, localBufferSize_SR * sizeof(OutputAtom));

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

	double tuneClassifyFinal(std::map<std::string, int>& config)
	{
		config["NUM_INSTANCES"] = classifyData->numInstances;
		config["NUM_LABELS"] = numLabels;
		config["NUM_ATTRIBUTES"] = numAttributes;
		config["NUM_CHAINS"] = numChains;
		config["NUM_TREES"] = numTrees;
		config["MAX_LEVEL"] = maxLevel;
		config["NODES_PER_TREE"] = pow(2.0f, maxLevel + 1) - 1;

		std::string optionString;
		std::stringstream strstr;
		for (auto it = config.begin(); it != config.end(); ++it)
			strstr << " -D " << it->first << "=" << it->second;
		optionString = strstr.str();

		cl_program prog;
		PlatformUtil::buildProgramFromSource(finalCalcSource, prog, optionString.c_str());
		Kernel* finalCalcKernel = new Kernel(prog, "finalCalc");
		clReleaseProgram(prog);

		PlatformUtil::buildProgramFromSource(finalReduceSource, prog, optionString.c_str());
		Kernel* finalReduceKernel = new Kernel(prog, "finalReduce");
		clReleaseProgram(prog);

		int finalIntermediateBufferSize[3] = { classifyData->numInstances, numLabels, config["NUM_WG_CHAINS_FC"] };
		int finalIntermediateBufferTotalSize = finalIntermediateBufferSize[0] * finalIntermediateBufferSize[1] * finalIntermediateBufferSize[2];

		int localBufferSize_FC = config["NUM_WI_INSTANCES_FC"] * config["NUM_WI_LABELS_FC"] * config["NUM_WI_CHAINS_FC"];
		int localBufferSize_FR = config["NUM_WI_INSTANCES_FR"] * config["NUM_WI_LABELS_FR"] * config["NUM_WI_CHAINS_FR"];

		Buffer finalIntermediateBuffer(finalIntermediateBufferTotalSize * sizeof(OutputAtom), CL_MEM_READ_WRITE);
		
		finalCalcKernel->SetArg(0, classifyData->labelBuffer);
		finalCalcKernel->SetLocalArg(1, localBufferSize_FC * sizeof(OutputAtom));
		finalCalcKernel->SetArg(2, finalIntermediateBuffer);

		finalCalcKernel->setDim(3);
		finalCalcKernel->setGlobalSize(config["NUM_WG_INSTANCES_FC"] * config["NUM_WI_INSTANCES_FC"], config["NUM_WG_LABELS_FC"] * config["NUM_WI_LABELS_FC"], config["NUM_WG_CHAINS_FC"] * config["NUM_WI_CHAINS_FC"]);
		finalCalcKernel->setLocalSize(config["NUM_WI_INSTANCES_FC"], config["NUM_WI_LABELS_FC"], config["NUM_WI_CHAINS_FC"]);

		finalReduceKernel->SetArg(0, finalIntermediateBuffer);
		finalReduceKernel->SetArg(1, classifyData->resultBuffer);
		finalReduceKernel->SetLocalArg(2, localBufferSize_FR * sizeof(OutputAtom));

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

	void finishClassify()
	{
		classifyData->dataBuffer.clear();
		classifyData->resultBuffer.clear();
		classifyData->labelBuffer.clear();
		classifyData->stepNodeValueBuffer.clear();
		classifyData->stepNodeIndexBuffer.clear();
		
		delete classifyData;
	}

	void runClassifyNew(ECCData& data, std::vector<double>& values, std::vector<int>& votes, std::map<std::string, int>& config)
	{
		config["NUM_INSTANCES"] = classifyData->numInstances;
		config["NUM_LABELS"] = numLabels;
		config["NUM_ATTRIBUTES"] = numAttributes;
		config["NUM_CHAINS"] = numChains;
		config["NUM_TREES"] = numTrees;
		config["MAX_LEVEL"] = maxLevel;
		config["NODES_PER_TREE"] = pow(2.0f, maxLevel + 1) - 1;

		std::cout << std::endl << "--- NEW CLASSIFICATION ---" << std::endl;
		std::string optionString;
		std::stringstream strstr;
		for (auto it = config.begin(); it != config.end(); ++it)
			strstr << " -D " << it->first << "=" << it->second;
		optionString = strstr.str();

		std::cout << optionString << std::endl;

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

		Buffer dataBuffer(data.getValueCount() * data.getSize() * sizeof(double), CL_MEM_READ_ONLY);

		int dataBuffIdx = 0;
		for (MultilabelInstance inst : data.getInstances())
		{
			memcpy(((double*)dataBuffer.getData()) + dataBuffIdx, inst.getData().data(), inst.getValueCount() * sizeof(double));
			dataBuffIdx += inst.getValueCount();
		}

		Buffer resultBuffer(data.getSize() * data.getLabelCount() * sizeof(OutputAtom), CL_MEM_WRITE_ONLY);
		Buffer labelBuffer(data.getSize() * data.getLabelCount() * numChains * sizeof(OutputAtom), CL_MEM_WRITE_ONLY);

		int stepIntermediateBufferSize[3] = { data.getSize(), numChains, config["NUM_WG_TREES_SC"] };
		int stepIntermediateBufferTotalSize = stepIntermediateBufferSize[0] * stepIntermediateBufferSize[1] * stepIntermediateBufferSize[2];

		int finalIntermediateBufferSize[3] = { data.getSize(), data.getLabelCount(), config["NUM_WG_CHAINS_FC"] };
		int finalIntermediateBufferTotalSize = finalIntermediateBufferSize[0] * finalIntermediateBufferSize[1] * finalIntermediateBufferSize[2];

		int localBufferSize_SC = config["NUM_WI_INSTANCES_SC"] * config["NUM_WI_CHAINS_SC"] * config["NUM_WI_TREES_SC"];
		int localBufferSize_SR = config["NUM_WI_INSTANCES_SR"] * config["NUM_WI_CHAINS_SR"] * config["NUM_WI_TREES_SR"];
		int localBufferSize_FC = config["NUM_WI_INSTANCES_FC"] * config["NUM_WI_LABELS_FC"] * config["NUM_WI_CHAINS_FC"];
		int localBufferSize_FR = config["NUM_WI_INSTANCES_FR"] * config["NUM_WI_LABELS_FR"] * config["NUM_WI_CHAINS_FR"];

		Buffer stepIntermediateBuffer(stepIntermediateBufferTotalSize * sizeof(OutputAtom), CL_MEM_READ_WRITE);
		Buffer finalIntermediateBuffer(finalIntermediateBufferTotalSize * sizeof(OutputAtom), CL_MEM_READ_WRITE);

		size_t stepModelSize = ecc->getTotalSize() / numLabels;
		Buffer stepNodeValueBuffer(sizeof(double) * stepModelSize, CL_MEM_READ_ONLY);
		Buffer stepNodeIndexBuffer(sizeof(int) * stepModelSize, CL_MEM_READ_ONLY);

		stepCalcKernel->SetArg(0, stepNodeValueBuffer, true);
		stepCalcKernel->SetArg(1, stepNodeIndexBuffer, true);
		stepCalcKernel->SetArg(2, dataBuffer, true);
		stepCalcKernel->SetArg(3, labelBuffer, true);
		stepCalcKernel->SetLocalArg(4, localBufferSize_SC * sizeof(OutputAtom));
		stepCalcKernel->SetArg(5, stepIntermediateBuffer);

		stepCalcKernel->setDim(3);
		stepCalcKernel->setGlobalSize(config["NUM_WG_INSTANCES_SC"] * config["NUM_WI_INSTANCES_SC"], config["NUM_WG_CHAINS_SC"] * config["NUM_WI_CHAINS_SC"], config["NUM_WG_TREES_SC"] * config["NUM_WI_TREES_SC"]);
		stepCalcKernel->setLocalSize(config["NUM_WI_INSTANCES_SC"], config["NUM_WI_CHAINS_SC"], config["NUM_WI_TREES_SC"]);

		stepReduceKernel->SetArg(0, stepIntermediateBuffer);
		stepReduceKernel->SetArg(1, labelBuffer);
		stepReduceKernel->SetArg(2, labelOrderBuffer, true);
		stepReduceKernel->SetLocalArg(3, localBufferSize_SR * sizeof(OutputAtom));

		stepReduceKernel->setDim(3);
		stepReduceKernel->setGlobalSize(config["NUM_WG_INSTANCES_SR"] * config["NUM_WI_INSTANCES_SR"], config["NUM_WG_CHAINS_SR"] * config["NUM_WI_CHAINS_SR"], config["NUM_WI_TREES_SR"]);
		stepReduceKernel->setLocalSize(config["NUM_WI_INSTANCES_SR"], config["NUM_WI_CHAINS_SR"], config["NUM_WI_TREES_SR"]);

		finalCalcKernel->SetArg(0, labelBuffer);
		finalCalcKernel->SetLocalArg(1, localBufferSize_FC * sizeof(OutputAtom));
		finalCalcKernel->SetArg(2, finalIntermediateBuffer);

		finalCalcKernel->setDim(3);
		finalCalcKernel->setGlobalSize(config["NUM_WG_INSTANCES_FC"] * config["NUM_WI_INSTANCES_FC"], config["NUM_WG_LABELS_FC"] * config["NUM_WI_LABELS_FC"], config["NUM_WG_CHAINS_FC"] * config["NUM_WI_CHAINS_FC"]);
		finalCalcKernel->setLocalSize(config["NUM_WI_INSTANCES_FC"], config["NUM_WI_LABELS_FC"], config["NUM_WI_CHAINS_FC"]);

		finalReduceKernel->SetArg(0, finalIntermediateBuffer);
		finalReduceKernel->SetArg(1, resultBuffer);
		finalReduceKernel->SetLocalArg(2, localBufferSize_FR * sizeof(OutputAtom));

		finalReduceKernel->setDim(3);
		finalReduceKernel->setGlobalSize(config["NUM_WG_INSTANCES_FR"] * config["NUM_WI_INSTANCES_FR"], config["NUM_WG_LABELS_FR"] * config["NUM_WI_LABELS_FR"], config["NUM_WI_CHAINS_FR"]);
		finalReduceKernel->setLocalSize(config["NUM_WI_INSTANCES_FR"], config["NUM_WI_LABELS_FR"], config["NUM_WI_CHAINS_FR"]);

		double SCTime = 0.0, SRTime = 0.0, FCTime = 0.0, FRTime = 0.0;
		Util::StopWatch stopWatch;
		stopWatch.start();

		for (int chainIndex = 0; chainIndex < numLabels; ++chainIndex)
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

		CPUTime = stopWatch.stop();
		kernelTime = SCTime + SRTime + FCTime + FRTime;
		stepTime = SCTime + SRTime;
		finalTime = FCTime + FRTime;
		std::cout << "Classification kernel took " << ((double)kernelTime * 1e-06) << " ms."
			<< "\n\tstepCalc: " << ((double)SCTime * 1e-06)
			<< "\n\tstepReduce: " << ((double)SRTime * 1e-06)
			<< "\n\tfinalCalc: " << ((double)FCTime * 1e-06)
			<< "\n\tfinalReduce: " << ((double)FRTime * 1e-06)
			<< std::endl;
		std::cout << "Total time: " << ((double)CPUTime*1e-06) << std::endl;

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

	size_t getStepTime()
	{
		return stepTime;
	}

	size_t getFinalTime()
	{
		return finalTime;
	}

	size_t getKernelTime()
	{
		return kernelTime;
	}

	size_t getCPUTime()
	{
		return CPUTime;
	}

	~ECCExecutorNew()
	{
		delete[] nodeIndices;
		delete[] nodeValues;
		labelOrderBuffer.clear();
		delete ecc;
	}
};

