#pragma once
#include<vector>
#include"Tree.h"

class Forest
{
	const int maxLevel;
	const int treeSize;
	const int forestSize;
	const int totalSize;

	const int label;
	std::vector<int> excludeLabels;
	std::vector<Tree> forest;

public:
	Forest(int _numValues, int _maxLevel, int _label, std::vector<int>& _excludeLabelIndices, int _forestSize);

	const std::vector<Tree>& getTrees();
	int getMaxLevel();
	int getTreeSize();
	int getForestSize();
	int getTotalSize();
	int size();
	int getLabel();
	const std::vector<int>& getExcludeLabels();
};

