#pragma once
#include <vector>
#include <array>
#include "Forest.hpp"

class ClassifierChain
{
private:
	const size_t maxLevel;
	const size_t treeSize;
	const size_t forestSize;
	const size_t chainSize;
	const size_t totalSize;

	std::vector<Forest> forests;
    std::vector<int> orderedLabels;
public:
	ClassifierChain(size_t _numValues, std::vector<int> _orderedLabels, size_t _maxLevel, size_t _forestSize);
	ClassifierChain(size_t numValues, size_t numLabels, size_t maxLevel, size_t forestSize);

	const std::vector<Forest>& getForests();
	const std::vector<int> getLabelOrder() const;
	size_t getMaxLevel();
	size_t getTreeSize();
	size_t getForestSize();
	size_t getChainSize();
	size_t getTotalSize();
	size_t size();

	static std::vector<int> standardLabelOrder(size_t numLabels);
};

