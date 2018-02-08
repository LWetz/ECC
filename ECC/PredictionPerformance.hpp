#pragma once
#include "ECCData.hpp"

class PredictionPerformance
{
private:
	double threshold;
	int numInstances;
	int numLabels;

	double loss(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet);
	double hammingLoss(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet);
	double accuracy(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet);
	double fmeasure(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet);
	double logloss(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet);
public:
	PredictionPerformance(int _numLabels, int _numInstances, double _threshold);
	Measurement calculatePerfomance(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet);
};
