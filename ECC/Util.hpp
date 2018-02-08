#pragma once
#include<random>
#include<vector>
#include<fstream>
#include<iostream>
#include<string>
#include<chrono>
#include<algorithm>
#include<map>

namespace Util
{
	class StopWatch
	{
	private:
		std::chrono::time_point<std::chrono::system_clock> startTime, endTime;

	public:
		void start();
		size_t stop();
	};

	int randomInt(int min, int max);
	int randomInt(int bound);
	int randomInt(int bound, const std::vector<int>& exclude);

	std::string loadFileToString(const std::string& filename);

	class Random
	{
	private:
		int64_t _seed;

		int64_t shr(int64_t lhs, uint8_t rhs);

	public:
		void setSeed(int64_t seed);

		int next(int bits);

		int nextInt(int bound);
	};

	extern Random RANDOM;
}

typedef std::map<std::string, double> Measurement;
typedef std::map<std::string, size_t> Configuration;