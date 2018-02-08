#include "Tree.hpp"

Tree::Tree(size_t _numValues, size_t _maxLevel, size_t _label, std::vector<int> _excludeLabelIndices)
	: maxLevel(_maxLevel), label(_label), excludeValues(_excludeLabelIndices), size(pow(2, _maxLevel + 1) - 1)
{
}

size_t Tree::getLabel()
{
	return label;
}

const std::vector<int>& Tree::getExcludedValues()
{
	return excludeValues;
}

size_t Tree::getMaxLevel()
{
	return maxLevel;
}

size_t Tree::getTotalSize()
{
	return size;
}
