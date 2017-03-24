#pragma once
#include<vector>
#include"RTree.h"

class RForest
{
	const int maxLevel;
	const int treeSize;
	const int forestSize;
	const int totalSize;

	const int label;
	std::vector<int> excludeLabels;
	std::vector<RTree> forest;

public:
	RForest(int _numValues, int _maxLevel, int _label, std::vector<int>& _excludeLabelIndices, int _forestSize);

	const std::vector<RTree>& getTrees();
	int getMaxLevel();
	int getTreeSize();
	int getForestSize();
	int getTotalSize();
	int size();
	int getLabel();
	const std::vector<int>& getExcludeLabels();
};

