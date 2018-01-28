#include "EnsembleOfClassifierChains.hpp"

EnsembleOfClassifierChains::EnsembleOfClassifierChains(int _numValues, int _numLabels, int _maxLevel, int _forestSize, int _ensembleSize, int _ensembleSubSetSize, int _forestSubSetSize)
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

std::vector<int> EnsembleOfClassifierChains::partitionInstanceIndices(int maxIndex)
{
	auto instances = std::vector<int>((forestSubSetSize-1) * chainSize * forestSize * ensembleSize);

	for (int chain = 0; chain < ensembleSize; chain++)
	{
		std::vector<int> chainIndices;
		chainIndices.reserve(ensembleSubSetSize);

		for (int i = 0; i < ensembleSubSetSize; i++)
		{
			chainIndices.push_back(Util::randomInt(maxIndex, chainIndices));
		}

		for (int forest = 0; forest < chainSize; forest++)
		{
			for (int tree = 0; tree < forestSize; tree++)
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

int EnsembleOfClassifierChains::getNumValues()
{
	return numValues;
}

int EnsembleOfClassifierChains::getNumAttributes()
{
	return numAttributes;
}

int EnsembleOfClassifierChains::getNumLabels()
{
	return numLabels;
}

int EnsembleOfClassifierChains::getEnsembleSubSetSize()
{
	return ensembleSubSetSize;
}

int EnsembleOfClassifierChains::getForestSubSetSize()
{
	return forestSubSetSize;
}

int EnsembleOfClassifierChains::getMaxLevel()
{
	return maxLevel;
}

int EnsembleOfClassifierChains::getTreeSize()
{
	return treeSize;
}

int EnsembleOfClassifierChains::getForestSize()
{
	return forestSize;
}

int EnsembleOfClassifierChains::getChainSize()
{
	return chainSize;
}

int EnsembleOfClassifierChains::getEnsembleSize()
{
	return ensembleSize;
}

int EnsembleOfClassifierChains::getTotalSize()
{
	return totalSize;
}

int EnsembleOfClassifierChains::getSize()
{
	return forestSize * treeSize * chainSize * ensembleSize;
}

