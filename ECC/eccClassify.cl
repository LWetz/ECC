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
                        int numChains,
                        int ensembleIndex
                ) 
{ 
    int startNodeIndex = (ensembleIndex * nodesPerTree * forestSize * numChains) 
        + (nodesPerTree * forestSize * chainIndex);
    int right = 0;
    int nodeIndex = 0;

    for(int level = 0; level < maxLevel; ++level)
    {
        int tmpNodeIndex = startNodeIndex + (treeIndex * nodesPerTree + nodeIndex);
        double nodeValue = nodeValues[tmpNodeIndex];
        int attributeIndex = attributeIndices[tmpNodeIndex];

        double value = data[gid * numValues + attributeIndex];

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

kernel void eccClassify(
                            //input (read-only)
                            global double* nodeValues,
                            global int* attributeIndices,
                            global int* labelOrders,
                            global int* pMaxLevel, 
                            global int* pForestSize,
                            global int* pNumChains, //equal to numLabels
                            global int* pNumEnsembles,
                            global double* data, 
                            global int* pNumValues,
                            //output (read-write)
                            global double* results,
                            //output (write-only)
                            global int* votes
                        ) 
{ 
    int gid = get_global_id(0);
    int nodesPerTree = pown(2.f, *pMaxLevel + 1) - 1;
    int maxLevel = *pMaxLevel;
    int numValues = *pNumValues;
    int numEnsembles = *pNumEnsembles;
    int numChains = *pNumChains;
    int forestSize = *pForestSize;
    int numAttributes = numValues - numChains;

    for(int ensembleIndex = 0; ensembleIndex < numEnsembles; ++ensembleIndex)
    {
        for(int chainIndex = 0; chainIndex < numChains; ++chainIndex)
        {

            int label = labelOrders[numChains * ensembleIndex + chainIndex];
            int resultIndex = (gid * numChains) + label;

            for(int treeIndex = 0; treeIndex < forestSize; ++treeIndex)
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
                    numChains,
                    ensembleIndex
                    );

                results[resultIndex] += value;
                
                if(value != 0)
                {
                    ++votes[resultIndex];
                }
            }

            if(results[resultIndex] > 0)
            {
                data[numValues * gid + numAttributes + label] = 1;
            }
            else
            {
                data[numValues * gid + numAttributes + label] = 0;
            }

        }
    }

}
