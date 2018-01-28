#define TREES_PER_ITEM (TOTAL_TREES/(NUM_WI*NUM_WG))

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
		int labelValue = data[instanceIndex * NUM_VALUES + NUM_ATTRIBUTES + labelIndex];

		if (data[instanceIndex * NUM_VALUES + randomIndex] > randomValue)
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
	int labelAt,
	global int* labelOrder,
	int splitStack,
	int* seed
)
{
	double bestGain = 0;
	double bestValue = 0;
	int bestIndex = 0;
	int splitSize = instancesLength[instancesLengthIndex];
	int lpred = labelAt % NUM_LABELS;
	int lstart = labelAt - lpred;

	if (splitSize > 0)
	{
		for (int attributeCounter = 0; attributeCounter < MAX_ATTRIBUTES; ++attributeCounter)
		{
			int randomInstance = instances[instancesStart + splitStack + randomInt(splitSize, seed)];

			int randomIndex;
			if (lpred <= 0)
			{
				randomIndex = randomInt(NUM_ATTRIBUTES, seed);
			}
			else
			{
				randomIndex = randomInt(NUM_VALUES, seed);
			}

			if (randomIndex >= NUM_ATTRIBUTES)
			{
				int lran = randomInt(lpred, seed);
				randomIndex = NUM_ATTRIBUTES + labelOrder[lstart + lran];
			}

			double randomValue = data[(randomInstance * NUM_VALUES) + randomIndex];

			double gain = informationGain(data, randomValue, randomIndex, labelOrder[labelAt], instances, instancesStart, splitStack, splitSize);
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
			if (data[instances[instancesStart + splitStack + index] * NUM_VALUES + bestIndex] <= bestValue)
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
			if (data[instances[instancesStart + splitStack + index] * NUM_VALUES + bestIndex] > bestValue)
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
	int gid,
	int label,
	int instance,
	int root
)
{
	int right;
	int nodeIndex = 0;

	double nodeValue;
	int attributeIndex;
	double value;

	for (int level = 0; level < MAX_LEVEL; ++level)
	{
		nodeValue = nodeValues[root + nodeIndex];
		attributeIndex = attributeIndices[root + nodeIndex];

		value = data[instance * NUM_VALUES + attributeIndex];

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

	if (data[instance * NUM_VALUES + NUM_ATTRIBUTES + label] == 1)
	{
		vote = 1;
	}
	else
	{
		vote = -1;
	}

	nodeValues[root + nodeIndex] += vote;
}

void reduceToSigns(
	global double* nodeValues,
	int gid
)
{
	int nodesStart = gid * NODES_PER_TREE + NODES_LAST_LEVEL - 1;

	for (int node = 0; node < NODES_LAST_LEVEL; ++node)
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

void buildTree(
	int gid,
	int gidOffset,
	global int* seeds,
	global double* data,
	global int* labelOrder,
	global int* instances,
	global int* instancesNext,
	global int* instancesLength,
	global int* instancesNextLength,
	global double* nodeValues,
	global int* attributeIndices
)
{
	int realGid = gid + gidOffset;
	int instancesStart = MAX_SPLITS * gid;
	int instancesLengthStart = NODES_LAST_LEVEL * gid;
	int rootNode = NODES_PER_TREE * gid;
	int labelAt = realGid / NUM_TREES;
	int seed = seeds[gid];

	instancesLength[instancesLengthStart] = MAX_SPLITS;

	for (int n = NODES_PER_TREE / 2; n < NODES_PER_TREE; ++n)
	{
		nodeValues[gid * NODES_PER_TREE + n] = 0;
	}

	int nodesSoFar = 0;
	for (int level = 0; level < MAX_LEVEL; ++level)
	{
		int maxNodes = 1 << level;
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
				labelAt,
				labelOrder,
				splitStack,
				&seed
			);

			splitStack += instancesLength[instancesLengthStart + node];

			nodeValues[rootNode + nodesSoFar + node] = split.value;
			attributeIndices[rootNode + nodesSoFar + node] = split.index;
		}

		for (int i = 0; i < MAX_SPLITS; ++i)
		{
			instances[instancesStart + i] = instancesNext[instancesStart + i];
		}

		for (int i = 0; i < maxNodes * 2; ++i)
		{
			instancesLength[instancesLengthStart + i] = instancesNextLength[instancesLengthStart + i];
		}

		nodesSoFar += maxNodes;
	}

	for (int instance = 0; instance < NUM_INSTANCES; ++instance)
	{
		train(
			data,
			nodeValues,
			attributeIndices,
			gid,
			labelOrder[labelAt],
			instance,
			NODES_PER_TREE * gid
		);
	}

	reduceToSigns(
		nodeValues,
		gid
	);
}

kernel void eccBuild(
		int gidOffset,
		global int* seeds,
		global double* data,
		global int* labelOrder,
		global int* instances,
		global int* instancesNext,
		global int* instancesLength,
		global int* instancesNextLength,
		global double* nodeValues,
		global int* attributeIndices
	)
{
	int i_wg_tree = get_group_id(0);

	int i_wi_tree = get_local_id(0);

	for (int t = 0; t < TREES_PER_ITEM; ++t)
	{
		int tree = i_wi_tree + i_wg_tree * NUM_WI + t * NUM_WG * NUM_WI;

		buildTree(tree,
			gidOffset,
			seeds,
			data,
			labelOrder,
			instances,
			instancesNext,
			instancesLength,
			instancesNextLength,
			nodeValues,
			attributeIndices);
	}
}