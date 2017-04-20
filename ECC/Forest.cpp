#include "Forest.h"
Forest::Forest(int _numValues, int _maxLevel, int _label, std::vector<int>& _excludeLabelIndices, int _forestSize)
	: maxLevel(_maxLevel), label(_label), excludeLabels(_excludeLabelIndices), forestSize(_forestSize), forest(),
	treeSize(pow(2, maxLevel + 1) - 1), totalSize(treeSize*forestSize)
{
	for (int n = 0; n < forestSize; n++)
	{
		forest.push_back(Tree(_numValues, maxLevel, label, _excludeLabelIndices));
	}
}

const std::vector<Tree>& Forest::getTrees()
{
	return forest;
}

int Forest::getMaxLevel()
{
	return maxLevel;
}

int Forest::getTreeSize()
{
	return treeSize;
}

int Forest::getForestSize()
{
	return forestSize;
}

int Forest::getTotalSize()
{
	return totalSize;
}

int Forest::size()
{
	return forestSize*treeSize;
}

int Forest::getLabel()
{
	return label;
}

const std::vector<int>& Forest::getExcludeLabels()
{
	return excludeLabels;
}

