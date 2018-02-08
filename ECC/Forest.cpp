#include "Forest.hpp"
Forest::Forest(size_t _numValues, size_t _maxLevel, size_t _label, std::vector<int>& _excludeLabelIndices, size_t _forestSize)
	: maxLevel(_maxLevel), label(_label), excludeLabels(_excludeLabelIndices), forestSize(_forestSize), forest(),
	treeSize(pow(2, maxLevel + 1) - 1), totalSize(treeSize*forestSize)
{
	for (size_t n = 0; n < forestSize; n++)
	{
		forest.push_back(Tree(_numValues, maxLevel, label, _excludeLabelIndices));
	}
}

const std::vector<Tree>& Forest::getTrees()
{
	return forest;
}

size_t Forest::getMaxLevel()
{
	return maxLevel;
}

size_t Forest::getTreeSize()
{
	return treeSize;
}

size_t Forest::getForestSize()
{
	return forestSize;
}

size_t Forest::getTotalSize()
{
	return totalSize;
}

size_t Forest::size()
{
	return forestSize*treeSize;
}

size_t Forest::getLabel()
{
	return label;
}

const std::vector<int>& Forest::getExcludeLabels()
{
	return excludeLabels;
}

