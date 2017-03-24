#pragma once
#include <vector>
#include <array>
#include "RForest.h"

class ClassifierChain
{
private:
	const int maxLevel;
	const int treeSize;
	const int forestSize;
	const int chainSize;
	const int totalSize;

	std::vector<RForest> forests;
    std::vector<int> orderedLabels;
public:
	ClassifierChain(int _numValues, std::vector<int> &_orderedLabels, int _maxLevel, int _forestSize);
	ClassifierChain(int numValues, int numLabels, int maxLevel, int forestSize);

	const std::vector<RForest>& getForests();
	const std::vector<int>& getLabelOrder();
	int getMaxLevel();
	int getTreeSize();
	int getForestSize();
	int getChainSize();
	int getTotalSize();
	int size();

	static std::vector<int> standardLabelOrder(int numLabels);
};

