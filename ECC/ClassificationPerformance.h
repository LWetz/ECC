#pragma once
#include <vector>

bool isSameResult(std::vector<double> valFirst, std::vector<int> voteFirst, std::vector<double> valSecond, std::vector<int> voteSecond)
{
	bool sameResult = true;
	for (int i = 0; i < valFirst.size() && sameResult; ++i)
	{
		if (abs(valFirst[i] - valSecond[i]) > 0.001)
		{
			sameResult = false;
		}
		if (voteFirst[i] - voteSecond[i] != 0)
		{
			sameResult = false;
		}
	}
	return sameResult;
}

//void loss()
//{
//	double hitsOld = 0, hitsNew = 0, hitsFixed = 0;
//	for (int i = 0; i < evalCopy.size(); ++i)
//	{
//		MultilabelInstance iOrig = evalCopy[i];
//		bool hitOld = true;
//		bool hitNew = true;
//		bool hitFixed = true;
//		for (int l = 0; l < numLabels && hitOld && hitNew && hitFixed; ++l)
//		{
//			double predNew = valNew[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			double predOld = valOld[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			double predFixed = valFixed[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			if (predOld != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				hitOld = false; 
//			}
//			if (predNew != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				hitNew = false;
//			}
//			if (predFixed != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				hitFixed = false;
//			}
//		}
//		hitsOld += hitOld;
//		hitsNew += hitNew;
//		hitsFixed += hitFixed;
//	}
//	double oldLoss = 1.0 - (hitsOld  / (double)evalCopy.size());
//	double newLoss = 1.0 - (hitsNew / (double)evalCopy.size());
//	double fixedLoss = 1.0 - (hitsFixed / (double)evalCopy.size());
//
//	std::cout << "0/1 Loss: " << oldLoss << " (OLD) | " << newLoss << " (NEW) | " << fixedLoss << " (FIXED)" << std::endl;
//	outfile << "0/1 Loss: " << oldLoss << " (OLD) | " << newLoss << " (NEW) | " << fixedLoss << " (FIXED)" << std::endl;
//}
//
//void hammingloss()
//{
//	double hitsOld = 0, hitsNew = 0, hitsFixed = 0;
//	for (int i = 0; i < evalCopy.size(); ++i)
//	{
//		MultilabelInstance iOrig = evalCopy[i];
//
//		for (int l = 0; l < numLabels; ++l)
//		{
//			double predNew = valNew[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			double predOld = valOld[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			double predFixed = valFixed[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			if (predOld != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				hitsOld += 1.0;
//			}
//			if (predNew != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				hitsNew += 1.0;
//			}
//			if (predFixed != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				hitsFixed += 1.0;
//			}
//		}
//	}
//	double oldLoss = 1.0 - (hitsOld / (double)(numLabels + evalCopy.size()));
//	double newLoss = 1.0 - (hitsNew / (double)(numLabels + evalCopy.size()));
//	double fixedLoss = 1.0 - (hitsFixed / (double)(numLabels + evalCopy.size()));
//
//	std::cout << "HammingLoss: " << oldLoss << " (OLD) | " << newLoss << " (NEW) | " << fixedLoss << " (FIXED)" << std::endl;
//	outfile << "HammingLoss: " << oldLoss << " (OLD) | " << newLoss << " (NEW) | " << fixedLoss << " (FIXED)" << std::endl;
//}
//
//void accuracy()
//{
//	double oldAccuracy = 0.0;
//	double newAccuracy = 0.0;
//	double fixedAccuracy = 0.0;
//	for (int i = 0; i < evalCopy.size(); ++i)
//	{
//		MultilabelInstance iOrig = evalCopy[i];
//
//		double unionOld = 0.0, unionNew = 0.0, unionFixed = 0.0;
//		double intersectionOld = 0.0, intersectionNew = 0.0, intersectionFixed = 0.0;
//
//		for (int l = 0; l < numLabels; ++l)
//		{
//			double predNew = valNew[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			double predOld = valOld[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			double predFixed = valFixed[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			if (predOld == 1.0 || iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				unionOld += 1.0;
//			}
//			if (predNew == 1.0 || iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				unionNew += 1.0;
//			}
//			if (predFixed == 1.0 || iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				unionFixed += 1.0;
//			}
//			if (predOld == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				intersectionOld += 1.0;
//			}
//			if (predNew == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				intersectionNew += 1.0;
//			}
//			if (predFixed == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				intersectionFixed += 1.0;
//			}
//		}
//		oldAccuracy += unionOld > 0.0 ? intersectionOld / unionOld : 0.0;
//		newAccuracy += unionNew > 0.0 ? intersectionNew / unionNew : 0.0;
//		fixedAccuracy += unionFixed > 0.0 ? intersectionFixed / unionFixed : 0.0;
//	}
//	oldAccuracy /= ((double)evalCopy.size());
//	newAccuracy /= ((double)evalCopy.size());
//	fixedAccuracy /= ((double)evalCopy.size());
//
//	std::cout << "Accuracy: " << oldAccuracy << " (OLD) | " << newAccuracy << " (NEW) | " << fixedAccuracy << " (FIXED)" << std::endl;
//	outfile << "Accuracy: " << oldAccuracy << " (OLD) | " << newAccuracy << " (NEW) | " << fixedAccuracy << " (FIXED)" << std::endl;
//}
//
//void fmeasure()
//{
//	double oldFMeasure = 0.0;
//	double newFMeasure = 0.0;
//	double fixedFMeasure = 0.0;
//
//	for (int l = 0; l < numLabels; ++l)
//	{
//		double tpOld = 0.0, tpNew = 0.0, tpFixed = 0.0;
//		double fpfnOld = 0.0, fpfnNew = 0.0, fpfnFixed = 0.0;
//
//		for (int i = 0; i < evalCopy.size(); ++i)
//		{
//			MultilabelInstance iOrig = evalCopy[i];
//
//			double predNew = valNew[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			double predOld = valOld[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			double predFixed = valFixed[i*numLabels + l] > 0 ? 1.0 : 0.0;
//			if (predOld == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				tpOld += 2.0;
//			}
//			if (predNew == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				tpNew += 2.0;
//			}
//			if (predFixed == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
//			{
//				tpFixed += 2.0;
//			}
//			if (predOld != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				fpfnOld += 1.0;
//			}
//			if (predNew != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				fpfnNew += 1.0;
//			}
//			if (predFixed != iOrig.getData()[l + iOrig.getNumAttribs()])
//			{
//				fpfnFixed += 1.0;
//			}
//		}
//		oldFMeasure += tpOld / (tpOld + fpfnOld);
//		newFMeasure += tpNew / (tpNew + fpfnNew);
//		fixedFMeasure += tpFixed / (tpFixed + fpfnFixed);
//	}
//	oldFMeasure /= (double)numLabels;
//	newFMeasure /= (double)numLabels;
//	fixedFMeasure /= (double)numLabels;
//
//	std::cout << "FMeasure: " << oldFMeasure << " (OLD) | " << newFMeasure << " (NEW) | " << fixedFMeasure << " (FIXED)" << std::endl;
//	outfile << "FMeasure: " << oldFMeasure << " (OLD) | " << newFMeasure << " (NEW) | " << fixedFMeasure << " (FIXED)" << std::endl;
//}
//
//#define MIN(a,b) ((a) > (b)) ? (a) : (b);
//
//void logloss()
//{
//	double oldLogLoss = 0.0;
//	double newLogLoss = 0.0;
//	double fixedLogLoss = 0.0;
//
//	double maximum = log(evalCopy.size());
//
//	for (int i = 0; i < evalCopy.size(); ++i)
//	{
//		for (int l = 0; l < numLabels; ++l)
//		{
//			MultilabelInstance iOrig = evalCopy[i];
//
//			double predNew = ((double)valNew[i*numLabels + l] / (double)voteNew[i*numLabels+l]) * 0.5 + 0.5;
//			double predOld = ((double)valOld[i*numLabels + l] / (double)voteOld[i*numLabels + l]) * 0.5 + 0.5;
//			double predFixed = ((double)valFixed[i*numLabels + l] / (double)voteFixed[i*numLabels + l]) * 0.5 + 0.5;
//
//			predNew = isnan(predNew) ? 0.0 : predNew;
//			predOld = isnan(predOld) ? 0.0 : predOld;
//			predFixed = isnan(predFixed) ? 0.0 : predFixed;
//
//			double real = iOrig.getData()[l + iOrig.getNumAttribs()];
//			if(predNew > 1e-08)
//				newLogLoss += MIN(maximum, -(log(predNew)*real + log(1.0 - predNew)*(1.0 - real)));
//			if (predOld > 1e-08)
//				oldLogLoss += MIN(maximum, -(log(predOld)*real + log(1.0 - predOld)*(1.0 - real)));
//			if (predFixed > 1e-08)
//				fixedLogLoss += MIN(maximum, -(log(predFixed)*real + log(1.0 - predFixed)*(1.0 - real)));
//		}
//	}
//	oldLogLoss /= (double)(numLabels + evalCopy.size());
//	newLogLoss /= (double)(numLabels + evalCopy.size());
//	fixedLogLoss /= (double)(numLabels + evalCopy.size());
//
//	std::cout << "LogLoss: " << oldLogLoss << " (OLD) | " << newLogLoss << " (NEW) | " << fixedLogLoss << " (FIXED)" << std::endl;
//	outfile << "LogLoss: " << oldLogLoss << " (OLD) | " << newLogLoss << " (NEW) | " << fixedLogLoss << " (FIXED)" << std::endl;
//}