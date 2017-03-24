#pragma once
#include<vector>
#include<algorithm>
#include"ClassifierChain.h"
#include"Util.h"

class EnsembleOfClassifierChains
{
private:
	const int numValues;
	const int numAttributes;
	const int numLabels;
	const int maxLevel;
	const int treeSize;
	const int forestSize;
	const int chainSize;
	const int ensembleSize;
	const int totalSize;
	const int ensembleSubSetSize;
	const int forestSubSetSize;

	std::vector<ClassifierChain> chains;
public:
	EnsembleOfClassifierChains(int _numValues, int _numLabels, int _maxLevel, int _forestSize, int _ensembleSize, int _ensembleSubSetSize, int _forestSubSetSize);
	std::vector<std::vector<int>> partitionInstanceIndices(int maxIndex);

	const std::vector<ClassifierChain>& getChains();
	int getNumValues();
	int getNumAttributes();
	int getNumLabels();
	int getEnsembleSubSetSize();
	int getForestSubSetSize();
	int getMaxLevel();
	int getTreeSize();
	int getForestSize();
	int getChainSize();
	int getEnsembleSize();
	int getTotalSize();
	int getSize();
};

