#pragma once
#include <vector>

class Tree
{
private:
	const int maxLevel;
	const int size;

	const int label;
	std::vector<int> excludeValues;
public:
	Tree(int _numValues, int _maxLevel, int _label, std::vector<int> _excludeLabelIndices);
	int getLabel();
	const std::vector<int>& getExcludedValues();
	int getMaxLevel();
	int getTotalSize();
};

