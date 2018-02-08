#include "PredictionPerformance.hpp"

double PredictionPerformance::loss(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet)
{
	int matches = 0;
	for (int i = 0; i < numInstances; ++i)
	{
		auto trueVec = trueSet[i];
		auto predictedVec = predictedSet[i];

		bool match = true;
		for (int l = 0; l < numLabels && match; ++l)
		{
			match &= trueVec.getLabel(l) == predictedVec.getPrediction(l, threshold);
		}

		if (match) ++matches;
	}

	return 1.0 - ((double)matches) / numInstances;
}

double PredictionPerformance::hammingLoss(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet)
{
	int matches = 0;
	for (int i = 0; i < numInstances; ++i)
	{
		auto trueVec = trueSet[i];
		auto predictedVec = predictedSet[i];

		for (int l = 0; l <numLabels; ++l)
		{
			if (trueVec.getLabel(l) == predictedVec.getPrediction(l, threshold))
			{
				++matches;
			}
		}
	}

	return 1.0 - ((double)matches) / (numLabels * numInstances);
}

double PredictionPerformance::accuracy(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet)
{
	double accuracy = 0.0;

	for (int i = 0; i < numInstances; ++i)
	{
		auto trueVec = trueSet[i];
		auto predictedVec = predictedSet[i];

		double orSize = 0;
		double andSize = 0;

		for (int l = 0; l <numLabels; ++l)
		{
			bool trueVal = trueSet[i].getLabel(l);
			bool predictedVal = predictedSet[i].getPrediction(l, threshold);

			if (trueVal || predictedVal)
				orSize++;

			if (trueVal && predictedVal)
				andSize++;
		}

		accuracy += orSize > 0.0 ? andSize / orSize : 1.0;
	}

	return accuracy / numInstances;
}

double PredictionPerformance::fmeasure(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet)
{
	double fmeasure = 0.0;

	for (int l = 0; l < numLabels; ++l)
	{
		double tp = 0.0;
		double fp = 0.0;
		double fn = 0.0;

		for (int i = 0; i < numInstances; ++i)
		{
			bool trueVal = trueSet[i].getLabel(l);
			bool predictedVal = predictedSet[i].getPrediction(l, threshold);

			if (trueVal && predictedVal)
				tp++;

			if (!trueVal && predictedVal)
				fp++;

			if (trueVal && !predictedVal)
				fn++;
		}

		fmeasure += 2.0 * tp / (2.0 * tp + fn + fp);
	}

	return fmeasure / numLabels;
}

double PredictionPerformance::logloss(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet)
{
	double logloss = 0.0;
	double logN = log((double)numInstances);

	for (int i = 0; i < numInstances; ++i)
	{
		auto trueVec = trueSet[i];
		auto predictedVec = predictedSet[i];

		for (int l = 0; l <numLabels; ++l)
		{
			bool trueVal = trueSet[i].getLabel(l);
			double confidence = predictedSet[i].getConfidence(l);

			double ll = trueVal ? -log(confidence) : -log(1.0 - confidence);
			logloss += logN < ll ? logN : ll;
		}
	}

	return logloss / (numLabels * numInstances);
}

PredictionPerformance::PredictionPerformance(size_t _numLabels, size_t _numInstances, double _threshold) :
	numLabels(_numLabels), numInstances(_numInstances), threshold(_threshold)
{

}

Measurement PredictionPerformance::calculatePerfomance(std::vector<MultilabelInstance> trueSet, std::vector<MultilabelPrediction> predictedSet)
{
	Measurement measurement;

	measurement["evalLoss"] = loss(trueSet, predictedSet);
	measurement["evalHammingLoss"] = hammingLoss(trueSet, predictedSet);
	measurement["evalAccuracy"] = accuracy(trueSet, predictedSet);
	measurement["evalFMeasure"] = fmeasure(trueSet, predictedSet);
	measurement["evalLogLoss"] = logloss(trueSet, predictedSet);

	return measurement;
}
