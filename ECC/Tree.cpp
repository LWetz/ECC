#include "Tree.h"

Tree::Tree(int _numValues, int _maxLevel, int _label, std::vector<int> _excludeLabelIndices)
	: maxLevel(_maxLevel), label(_label), excludeValues(_excludeLabelIndices), size(pow(2, _maxLevel + 1) - 1)
{
}

int Tree::getLabel()
{
	return label;
}

const std::vector<int>& Tree::getExcludedValues()
{
	return excludeValues;
}

int Tree::getMaxLevel()
{
	return maxLevel;
}

int Tree::getTotalSize()
{
	return size;
}
