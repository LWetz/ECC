//32768
constant double MAX_RND = 4294967296;

typedef struct
{
	int index;
	double value;
} splitStruct;

double random(int* seed)
{
	ulong useed = 0 + (*seed);
	useed = (useed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
	double result = useed >> 16;
	*seed = useed;

	return result / MAX_RND;
}

int randomInt(int max, int* seed)
{
	int rnd = (random(seed) * max);

	return rnd;
}

double entropy(double total, double positives, double negatives)
{
	double sum = 0;

	if (positives != 0)
	{
		double prob = positives / total;
		sum += prob * native_log2((float)prob);
	}

	if (negatives != 0)
	{
		double prob = negatives / total;
		sum += prob * native_log2((float)prob);
	}

	return sum * -1;
}

double informationGain(
	global double* data,
	double randomValue,
	int randomIndex,
	int labelIndex,
	int numValues,
	int numAttributes,
	global int* instances,
	int instancesStart,
	int splitStack,
	int splitSize
)
{
	double gPositives = 0;
	double gNegatives = 0;
	double lPositives = 0;
	double lNegatives = 0;

	for (int dataIndex = 0; dataIndex < splitSize; ++dataIndex)
	{
		int instanceIndex = instances[instancesStart + splitStack + dataIndex];
		int labelValue = data[instanceIndex * numValues + numAttributes + labelIndex];

		if (data[instanceIndex * numValues + randomIndex] > randomValue)
		{

			if (labelValue == 1)
			{
				gPositives += 1;
			}
			else
			{
				gNegatives += 1;
			}

		}
		else if (labelValue == 1)
		{
			lPositives += 1;
		}
		else
		{
			lNegatives += 1;
		}
	}

	double totalPositives = gPositives + lPositives;
	double totalNegatives = gNegatives + lNegatives;
	double totalEntropy = entropy(splitSize, totalPositives, totalNegatives);

	double gTotal = gPositives + gNegatives;
	double gProb = 0;
	double gEntropy = 0;

	if (gTotal > 0)
	{
		gProb = gTotal / splitSize;
		gEntropy = entropy(gTotal, gPositives, gNegatives);
	}

	double lTotal = lPositives + lNegatives;
	double lProb = 0;
	double lEntropy = 0;

	if (lTotal > 0)
	{
		lProb = lTotal / splitSize;
		lEntropy = entropy(lTotal, lPositives, lNegatives);
	}

	return totalEntropy - (gProb * gEntropy + lProb * lEntropy);
}

splitStruct findSplit(
	global double* data,
	global int* instances,
	global int* instancesNext,
	global int* instancesLength,
	global int* instancesNextLength,
	int instancesStart,
	int instancesLengthIndex,
	int instancesNextLengthIndex,
	int numValues,
	int numAttributes,
	int labelAt,
	int chainSize,
	global int* labelOrder,
	int splitStack,
	int maxAttributes,
	int* seed
)
{
	double bestGain = 0;
	double bestValue = 0;
	int bestIndex = 0;
	int splitSize = instancesLength[instancesLengthIndex];
	int lpred = labelAt % chainSize;
	int lstart = labelAt - lpred;

	if (splitSize > 0)
	{
		for (int attributeCounter = 0; attributeCounter < maxAttributes; ++attributeCounter)
		{
			int randomInstance = instances[instancesStart + splitStack + randomInt(splitSize, seed)];

			int randomIndex;
			if (lpred <= 0)
			{
				randomIndex = randomInt(numAttributes, seed);
			}
			else
			{
				randomIndex = randomInt(numValues, seed);
			}

			if (randomIndex >= numAttributes)
			{
				int lran = randomInt(lpred, seed);
				randomIndex = numAttributes + labelOrder[lstart + lran];
			}

			double randomValue = data[(randomInstance * numValues) + randomIndex];

			double gain = informationGain(data, randomValue, randomIndex, labelOrder[labelAt], numValues, numAttributes, instances, instancesStart, splitStack, splitSize);
			if (gain > bestGain)
			{
				bestGain = gain;
				bestValue = randomValue;
				bestIndex = randomIndex;
			}
		}

		int instancesIndex = 0;

		//left split
		int leftSize = 0;
		for (int index = 0; index < splitSize; ++index)
		{
			if (data[instances[instancesStart + splitStack + index] * numValues + bestIndex] <= bestValue)
			{
				instancesNext[instancesStart + splitStack + instancesIndex] = instances[instancesStart + splitStack + index];
				++instancesIndex;
				++leftSize;
			}
		}

		//right split
		int rightSize = 0;
		for (int index = 0; index < splitSize; ++index)
		{
			if (data[instances[instancesStart + splitStack + index] * numValues + bestIndex] > bestValue)
			{
				instancesNext[instancesStart + splitStack + instancesIndex] = instances[instancesStart + splitStack + index];
				++instancesIndex;
				++rightSize;
			}
		}

		instancesNextLength[instancesNextLengthIndex] = leftSize;
		instancesNextLength[instancesNextLengthIndex + 1] = rightSize;
	}
	else
	{
		instancesNextLength[instancesNextLengthIndex] = 0;
		instancesNextLength[instancesNextLengthIndex + 1] = 0;
	}


	splitStruct s;
	s.value = bestValue;
	s.index = bestIndex;
	return s;
}

void train(
	global double* data,
	global double* nodeValues,
	global int* attributeIndices,
	global int* numVotes,
	int gid,
	int numValues,
	int numAttributes,
	int label,
	int maxLevel,
	int instance,
	int root
)
{
	int right;
	int nodeIndex = 0;

	double nodeValue;
	int attributeIndex;
	double value;

	for (int level = 0; level < maxLevel; ++level)
	{
		nodeValue = nodeValues[root + nodeIndex];
		attributeIndex = attributeIndices[root + nodeIndex];

		value = data[instance * numValues + attributeIndex];

		if (value > nodeValue)
		{
			right = 2;
		}
		else
		{
			right = 1;
		}

		nodeIndex = nodeIndex * 2 + right;
	}

	int vote;

	if (data[instance * numValues + numAttributes + label] == 1)
	{
		vote = 1;
	}
	else
	{
		vote = -1;
	}

	nodeValues[root + nodeIndex] += vote;

	int numLeaves = pown(2.f, maxLevel);
	numVotes[gid * numLeaves + nodeIndex - (numLeaves - 1)] += 1;
}

void reduceToSigns(
	global double* nodeValues,
	global int* numVotes,
	int gid,
	int maxLevel
)
{
	int nodesPerTree = pown(2.f, maxLevel + 1) - 1;
	int numLeaves = pown(2.f, maxLevel);
	int nodesLastLevel = pown(2.f, maxLevel - 1) - 1;

	int nodesStart = gid * nodesPerTree + numLeaves - 1;

	for (int node = 0; node < numLeaves; ++node)
	{
		double nodeValue = nodeValues[nodesStart + node];

		if (nodeValue > 0)
		{
			nodeValues[nodesStart + node] = 1;
		}
		else if (nodeValue < 0)
		{
			nodeValues[nodesStart + node] = -1;
		}
		else
		{
			nodeValues[nodesStart + node] = 0;
		}

	}
}

kernel void eccBuild(       //input (read-only)
	global int* gidMultiplier,
	global int* seeds,
	global double* data,
	global int* pDataSize,
	global int* pSubSetSize,
	global int* labelOrder,
	global int* pNumValues,
	global int* pNumAttributes,
	global int* pMaxAttributes,
	global int* pMaxLevel,
	global int* pChainSize,
	global int* pMaxSplits,
	global int* pForestSize,
	//read-write
	global int* instances,
	global int* instancesNext,
	global int* instancesLength,
	global int* instancesNextLength,
	global double* nodeValues,
	global int* attributeIndices,
	global int* numVotes
)
{
	int gid = get_global_id(0);
	int dataSize = *pDataSize;
	int subSetSize = *pSubSetSize;
	int maxAttributes = *pMaxAttributes;
	int numValues = *pNumValues;
	int numAttributes = *pNumAttributes;
	int maxLevel = *pMaxLevel;
	int chainSize = *pChainSize;
	int maxSplits = *pMaxSplits;
	int nodesLastLevel = pown(2.f, maxLevel);
	int nodesPerTree = pown(2.f, maxLevel + 1) - 1;
	int instancesStart = maxSplits * gid;
	int instancesLengthStart = nodesLastLevel * gid;
	int rootNode = nodesPerTree * gid;
	int forestSize = *pForestSize;
	int ensembleSize = get_global_size(0) / (forestSize * chainSize);
	int labelAt = (int)floor(((float)(gid * *gidMultiplier) / (float)get_global_size(0)) * (chainSize * ensembleSize));
	int seed = seeds[gid];

	instancesLength[instancesLengthStart] = maxSplits;

	int nodesSoFar = 0;
	for (int level = 0; level < maxLevel; ++level)
	{
		int maxNodes = pown(2.f, level);
		int splitStack = 0;
		for (int node = 0; node < maxNodes; ++node)
		{

			splitStruct split = findSplit(
				data,
				instances,
				instancesNext,
				instancesLength,
				instancesNextLength,
				instancesStart,
				instancesLengthStart + node, //instancesLengthIndex
				instancesLengthStart + (node * 2), //instancesNextLengthIndex
				numValues,
				numAttributes,
				labelAt,
				chainSize,
				labelOrder,
				splitStack,
				maxAttributes,
				&seed
			);

			splitStack += instancesLength[instancesLengthStart + node];

			nodeValues[rootNode + nodesSoFar + node] = split.value;
			attributeIndices[rootNode + nodesSoFar + node] = split.index;
		}

		for (int i = 0; i < maxSplits; ++i)
		{
			instances[instancesStart + i] = instancesNext[instancesStart + i];
		}

		for (int i = 0; i < maxNodes * 2; ++i)
		{
			instancesLength[instancesLengthStart + i] = instancesNextLength[instancesLengthStart + i];
		}

		nodesSoFar += maxNodes;
	}

	for (int instance = 0; instance < dataSize; ++instance)
	{
		train(
			data,
			nodeValues,
			attributeIndices,
			numVotes,
			gid,
			numValues,
			numAttributes,
			labelOrder[labelAt],
			maxLevel,
			instance,
			nodesPerTree * gid
		);
	}

	reduceToSigns(
		nodeValues,
		numVotes,
		gid,
		maxLevel
	);


}