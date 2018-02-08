#pragma once
#include <vector>
#include <math.h>
#include <stddef.h>

class Tree
{
private:
	const size_t maxLevel;
	const size_t size;

	const size_t label;
	std::vector<int> excludeValues;
public:
	Tree(size_t _numValues, size_t _maxLevel, size_t _label, std::vector<int> _excludeLabelIndices);
	size_t getLabel();
	const std::vector<int>& getExcludedValues();
	size_t getMaxLevel();
	size_t getTotalSize();
};

