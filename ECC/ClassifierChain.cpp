#include "ClassifierChain.h"

ClassifierChain::ClassifierChain(int _numValues, std::vector<int> _orderedLabels, int _maxLevel, int _forestSize)
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

ClassifierChain::ClassifierChain(int numValues, int numLabels, int maxLevel, int forestSize)
	: ClassifierChain(numValues, standardLabelOrder(numLabels), maxLevel, forestSize)
{
}

const std::vector<Forest>& ClassifierChain::getForests()
{
	return forests;
}

const std::vector<int> ClassifierChain::getLabelOrder()
{
	return orderedLabels;
}

int ClassifierChain::getMaxLevel()
{
	return maxLevel;
}

int ClassifierChain::getTreeSize()
{
	return treeSize;
}

int ClassifierChain::getForestSize()
{
	return forestSize;
}

int ClassifierChain::getChainSize()
{
	return chainSize;
}

int ClassifierChain::getTotalSize()
{
	return totalSize;
}

int ClassifierChain::size()
{
	return forestSize*treeSize*chainSize;
}

std::vector<int> ClassifierChain::standardLabelOrder(int numLabels)
{
	std::vector<int> labels(numLabels);
	for (int n = 0; n < numLabels; n++)
	{
		labels[n] = n;
	}
	return labels;
}