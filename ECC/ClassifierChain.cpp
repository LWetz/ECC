#include "ClassifierChain.hpp"

ClassifierChain::ClassifierChain(size_t _numValues, std::vector<int> _orderedLabels, size_t _maxLevel, size_t _forestSize)
	: maxLevel(_maxLevel), forestSize(_forestSize), chainSize(_orderedLabels.size()), orderedLabels(_orderedLabels),
	forests(), treeSize(pow(2, maxLevel + 1) - 1), totalSize(forestSize * treeSize * chainSize)
{
	std::vector<int> excludeLabels(orderedLabels);

	for (int label = 0; label < orderedLabels.size(); ++label)
	{
		forests.push_back(Forest(_numValues, maxLevel, orderedLabels[label], excludeLabels, forestSize));
		excludeLabels.erase(excludeLabels.begin());
	}
}

ClassifierChain::ClassifierChain(size_t numValues, size_t numLabels, size_t maxLevel, size_t forestSize)
	: ClassifierChain(numValues, standardLabelOrder(numLabels), maxLevel, forestSize)
{
}

const std::vector<Forest>& ClassifierChain::getForests()
{
	return forests;
}

const std::vector<int> ClassifierChain::getLabelOrder() const
{
	return orderedLabels;
}

size_t ClassifierChain::getMaxLevel()
{
	return maxLevel;
}

size_t ClassifierChain::getTreeSize()
{
	return treeSize;
}

size_t ClassifierChain::getForestSize()
{
	return forestSize;
}

size_t ClassifierChain::getChainSize()
{
	return chainSize;
}

size_t ClassifierChain::getTotalSize()
{
	return totalSize;
}

size_t ClassifierChain::size()
{
	return forestSize*treeSize*chainSize;
}

std::vector<int> ClassifierChain::standardLabelOrder(size_t numLabels)
{
	std::vector<int> labels(numLabels);
	for (size_t n = 0; n < numLabels; n++)
	{
		labels[n] = n;
	}
	return labels;
}