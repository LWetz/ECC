#ifndef INSTANCES_PER_ITEM
#define INSTANCES_PER_ITEM 2
#endif

#ifndef CHAINS_PER_ITEM
#define CHAINS_PER_ITEM 2
#endif

#ifndef TREES_PER_ITEM
#define TREES_PER_ITEM 2
#endif

double traverse(
                        global double* data,
                        global double* nodeValues,
                        int forestSize,
                        int maxLevel,
                        int nodesPerTree,
                        global int* attributeIndices,
                        int gid,
                        int numValues,
                        int treeIndex,
                        int chainIndex,
                        int chainSize,
                        int ensembleIndex,
						int numAttributes,
						local labelBuffer
                ) 
{ 
    int startNodeIndex = (ensembleIndex * nodesPerTree * forestSize * chainSize) 
        + (nodesPerTree * forestSize * chainIndex);
    int right = 0;
    int nodeIndex = 0;

    for(int level = 0; level < maxLevel; ++level)
    {
        int tmpNodeIndex = startNodeIndex + (treeIndex * nodesPerTree + nodeIndex);
        double nodeValue = nodeValues[tmpNodeIndex];
        int attributeIndex = attributeIndices[tmpNodeIndex];

		double value;
		if (attributeIndex < numAttributes)
			value = data[gid * numValues + attributeIndex];
		else
		{
			value = labelBuffer[chainSize * ensembleSize * gid + chainSize * chainIndex + attributeIndex - numAttributes];
			value = select(0.0, 1.0, value > 0),
		}

        if(value > nodeValue) 
        {
            right = 2;
        }
        else
        {   
            right = 1;
        }

        nodeIndex = nodeIndex * 2 + right;

    }

    return nodeValues[startNodeIndex + treeIndex * nodesPerTree + nodeIndex];
    
}

void reduceForests(local double* groupResultBuffer, local int* groupVoteBuffer, int* localBufferSize, int localTree, int ensembleOff, int localGidOff,
	global int* globalVoteBuffer, global double* globalResultBuffer, int* globalBufferSize, int localEnsembleOff, int gidOff)
{
	int groupSize = localBufferSize[2];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = groupSize >> 1; i > 0; i >>= 1)
	{
		if (localTree < i)
		{
			for (int gid = localGidOff; gid < localGidOff + INSTANCES_PER_ITEM; ++gid)
			{
				for (int ensembleIndex = localEnsembleOff; ensembleIndex < localEnsembleOff + CHAINS_PER_ITEM; ++ensembleIndex)
				{
					int localIndex = localBufferSize[1] * localBufferSize[2] * gid + localBufferSize[1] * ensembleIndex + lid;
					groupResultBuffer[localIndex] += groupResultBuffer[localIndex + i];
					groupVoteBuffer[localIndex] += groupVoteBuffer[localIndex + i];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localTree == 0)
	{
		for (int gid = 0; gid < INSTANCES_PER_ITEM; ++gid)
		{
			for (int ensembleIndex = 0; ensembleIndex < CHAINS_PER_ITEM; ++ensembleIndex)
			{
				int globalIndex = globalBufferSize[1] * globalBufferSize[2] * (gid + gidOff) + globalBufferSize[1] * (ensembleIndex + ensembleOff) + get_group_id(2);
				int localIndex = localBufferSize[1] * localBufferSize[2] * (gid + localGidOff) + localBufferSize[1] * (ensembleIndex + localEnsembleOff);

				globalVoteBuffer[globalIndex] = groupVoteBuffer[localIndex];
				globalResultBuffer[globalIndex] = groupVoteBuffer[localIndex];
			} 
		}
	}
}

kernel void eccClassify(
	//input (read-only)
	global double* nodeValues,
	global int* attributeIndices,
	global int* labelOrders,
	global int* pMaxLevel,
	global int* pForestSize,
	global int* pChainSize, //equal to numLabels
	global int* pEnsembleSize,
	global double* data,
	global int* pNumValues,
	global int* pChainIndex,
	//output (read-write)
	global double* results,
	//output (write-only)
	global int* votes,
	//new params
	global* double* labelBuffer,
	local double* groupResultBuffer,
	local int* groupVoteBuffer,
	global int* localBufferSize,

	global double* globalResultBuffer,
	global int* globalVoteBuffer,
	global int* globalBufferSize,
	)
{
	int gidOff = get_global_id(0) * INSTANCES_PER_ITEM;
	int ensembleOff = get_global_id(1) * CHAINS_PER_ITEM;
	int treeOff = get_global_id(2) * TREES_PER_ITEM;

	int localTree = get_local_id(2);
	int localEnsembleOff = get_local_id(1) * CHAINS_PER_ITEM;
	int localGidOff = get_local_id(0)* INSTANCES_PER_ITEM;

	int chainIndex = *pChainIndex;

	int nodesPerTree = pown(2.f, *pMaxLevel + 1) - 1;
	int maxLevel = *pMaxLevel;
	int numValues = *pNumValues;
	int ensembleSize = *pEnsembleSize;
	int chainSize = *pChainSize;
	int forestSize = *pForestSize;
	int numAttributes = numValues - chainSize;

	int itemVoteBuffer[CHAINS_PER_ITEM * INSTANCES_PER_ITEM];
	double itemResultBuffer[CHAINS_PER_ITEM * INSTANCES_PER_ITEM];

	initLocal(groupResults, groupVotes, blockSize, lid);

	for (int gid = gidOff; gid < INSTANCES_PER_ITEM + gidOff; ++gid)
	{
		for (int ensembleIndex = ensembleOff; ensembleIndex < CHAINS_PER_ITEM + ensembleOff; ++ensembleIndex)
		{
			int label = labelOrders[chainSize * ensembleIndex + chainIndex];
			int resultIndex = (gid * chainSize) + label;
			int lid = ensembleIndex * localBufferStride + localTree;

			double itemResult = 0.0;
			int itemVote = 0;

			for (int treeIndex = treeOff; treeIndex < TREES_PER_ITEM + treeOff; ++treeIndex)
			{
				double value = traverse(
					data,
					nodeValues,
					forestSize,
					maxLevel,
					nodesPerTree,
					attributeIndices,
					gid,
					numValues,
					treeIndex,
					chainIndex,
					chainSize,
					ensembleIndex,
					numAttributes,
					labelBuffer
				);

				itemResult += value;

				if (value != 0)
				{
					++itemVote;
				}
			}

			int localIndex = localBufferSize[1] * localBufferSize[2] * (localGidOff + gid - gidOff)
				+ localBufferSize[1] * (localEnsembleOff + ensembleIndex - ensembleOff)
				+ localTree;
			groupResultBuffer[localIndex] = itemResult;
			groupVoteBuffer[localIndex] = itemVote;
		}
	}

	reduceForests(groupResultBuffer, groupVoteBuffer, localBufferSize, localTree, ensembleOff, localGidOff,
		globalVoteBuffer, globalResultBuffer, globalBufferSize, localEnsembleOff, gidOff);
}