#include "RForest.h"
RForest::RForest(int _numValues, int _maxLevel, int _label, std::vector<int>& _excludeLabelIndices, int _forestSize)
	: maxLevel(_maxLevel), label(_label), excludeLabels(_excludeLabelIndices), forestSize(_forestSize), forest(),
	treeSize(pow(2, maxLevel + 1) - 1), totalSize(treeSize*forestSize)
{
	for (int n = 0; n < forestSize; n++)
	{
		forest.push_back(RTree(_numValues, maxLevel, label, _excludeLabelIndices));
	}
}

const std::vector<RTree>& RForest::getTrees()
{
	return forest;
}

int RForest::getMaxLevel()
{
	return maxLevel;
}

int RForest::getTreeSize()
{
	return treeSize;
}

int RForest::getForestSize()
{
	return forestSize;
}

int RForest::getTotalSize()
{
	return totalSize;
}

int RForest::size()
{
	return forestSize*treeSize;
}

int RForest::getLabel()
{
	return label;
}

const std::vector<int>& RForest::getExcludeLabels()
{
	return excludeLabels;
}

