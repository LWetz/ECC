#include "RTree.h"

RTree::RTree(int _numValues, int _maxLevel, int _label, std::vector<int> _excludeLabelIndices)
	: maxLevel(_maxLevel), label(_label), excludeValues(_excludeLabelIndices), size(pow(2, _maxLevel + 1) - 1)
{
}

int RTree::getLabel()
{
	return label;
}

const std::vector<int>& RTree::getExcludedValues()
{
	return excludeValues;
}

int RTree::getMaxLevel()
{
	return maxLevel;
}

int RTree::getTotalSize()
{
	return size;
}
