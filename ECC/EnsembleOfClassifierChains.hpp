#pragma once
#include<vector>
#include<algorithm>
#include"ClassifierChain.hpp"
#include"Util.hpp"

class EnsembleOfClassifierChains
{
private:
	const size_t numValues;
	const size_t numAttributes;
	const size_t numLabels;
	const size_t maxLevel;
	const size_t treeSize;
	const size_t forestSize;
	const size_t chainSize;
	const size_t ensembleSize;
	const size_t totalSize;
	const size_t ensembleSubSetSize;
	const size_t forestSubSetSize;

	std::vector<ClassifierChain> chains;
public:
	EnsembleOfClassifierChains(size_t _numValues, size_t _numLabels, size_t _maxLevel, size_t _forestSize, size_t _ensembleSize, size_t _ensembleSubSetSize, size_t _forestSubSetSize);
	std::vector<int> partitionInstanceIndices(size_t maxIndex);

	const std::vector<ClassifierChain>& getChains();
	size_t getNumValues();
	size_t getNumAttributes();
	size_t getNumLabels();
	size_t getEnsembleSubSetSize();
	size_t getForestSubSetSize();
	size_t getMaxLevel();
	size_t getTreeSize();
	size_t getForestSize();
	size_t getChainSize();
	size_t getEnsembleSize();
	size_t getTotalSize();
	size_t getSize();
};

