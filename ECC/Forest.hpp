#pragma once
#include<vector>
#include"Tree.hpp"

class Forest
{
	const size_t maxLevel;
	const size_t treeSize;
	const size_t forestSize;
	const size_t totalSize;

	const int label;
	std::vector<int> excludeLabels;
	std::vector<Tree> forest;

public:
	Forest(size_t _numValues, size_t _maxLevel, size_t _label, std::vector<int>& _excludeLabelIndices, size_t _forestSize);

	const std::vector<Tree>& getTrees();
	size_t getMaxLevel();
	size_t getTreeSize();
	size_t getForestSize();
	size_t getTotalSize();
	size_t size();
	size_t getLabel();
	const std::vector<int>& getExcludeLabels();
};

