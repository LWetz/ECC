#include "EnsembleOfClassifierChains.hpp"

EnsembleOfClassifierChains::EnsembleOfClassifierChains(size_t _numValues, size_t _numLabels, size_t _maxLevel, size_t _forestSize, size_t _ensembleSize, size_t _ensembleSubSetSize, size_t _forestSubSetSize)
	: numValues(_numValues), numAttributes(numValues - numLabels), numLabels(_numLabels), maxLevel(_maxLevel), forestSize(_forestSize), chainSize(_numLabels), ensembleSize(_ensembleSize),
	treeSize(pow(2, _maxLevel + 1) - 1), totalSize(treeSize*forestSize*chainSize*ensembleSize), ensembleSubSetSize(_ensembleSubSetSize), forestSubSetSize(_forestSubSetSize), chains()
{
	for (int ensemble = 0; ensemble < ensembleSize; ++ensemble)
	{
		std::vector<int> randomLabelOrder(numLabels);
		std::vector<int> labels(numLabels);
		for (int n = 0; n < numLabels; n++)
			labels[n] = n;

		for (int label = 0; label < randomLabelOrder.size(); ++label)
		{
			int idx = Util::randomInt(labels.size());
			randomLabelOrder[label] = labels[idx];
			labels.erase(labels.begin() + idx);
		}

		chains.push_back(ClassifierChain(numValues, randomLabelOrder, maxLevel, forestSize));
	}
}

std::vector<int> EnsembleOfClassifierChains::partitionInstanceIndices(size_t maxIndex)
{
	auto instances = std::vector<int>((forestSubSetSize-1) * chainSize * forestSize * ensembleSize);

	for (size_t chain = 0; chain < ensembleSize; chain++)
	{
		std::vector<int> chainIndices;
		chainIndices.reserve(ensembleSubSetSize);

		for (size_t i = 0; i < ensembleSubSetSize; i++)
		{
			chainIndices.push_back(Util::randomInt(maxIndex, chainIndices));
		}

		for (size_t forest = 0; forest < chainSize; forest++)
		{
			for (size_t tree = 0; tree < forestSize; tree++)
			{
				std::vector<int> treeInstances(chainIndices);

				while (treeInstances.size() >= forestSubSetSize)
				{
					treeInstances.erase(treeInstances.begin() + Util::randomInt(treeInstances.size()));
				}

				std::copy(treeInstances.begin(), treeInstances.end(), instances.begin() + (forestSubSetSize-1) * ((chain * chainSize + forest) * forestSize + tree));
			}
		}
	}

	return instances;
}

const std::vector<ClassifierChain>& EnsembleOfClassifierChains::getChains()
{
	return chains;
}

size_t EnsembleOfClassifierChains::getNumValues()
{
	return numValues;
}

size_t EnsembleOfClassifierChains::getNumAttributes()
{
	return numAttributes;
}

size_t EnsembleOfClassifierChains::getNumLabels()
{
	return numLabels;
}

size_t EnsembleOfClassifierChains::getEnsembleSubSetSize()
{
	return ensembleSubSetSize;
}

size_t EnsembleOfClassifierChains::getForestSubSetSize()
{
	return forestSubSetSize;
}

size_t EnsembleOfClassifierChains::getMaxLevel()
{
	return maxLevel;
}

size_t EnsembleOfClassifierChains::getTreeSize()
{
	return treeSize;
}

size_t EnsembleOfClassifierChains::getForestSize()
{
	return forestSize;
}

size_t EnsembleOfClassifierChains::getChainSize()
{
	return chainSize;
}

size_t EnsembleOfClassifierChains::getEnsembleSize()
{
	return ensembleSize;
}

size_t EnsembleOfClassifierChains::getTotalSize()
{
	return totalSize;
}

size_t EnsembleOfClassifierChains::getSize()
{
	return forestSize * treeSize * chainSize * ensembleSize;
}

