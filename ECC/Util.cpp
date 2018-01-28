#include "Util.hpp"

namespace Util
{
	void StopWatch::start()
	{
		startTime = std::chrono::system_clock::now();
	}

	size_t StopWatch::stop()
	{
		endTime = std::chrono::system_clock::now();
		auto elapsed = endTime - startTime;
		return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
	}

	int64_t Random::shr(int64_t lhs, uint8_t rhs)
	{
		return (lhs >> rhs) & ((1LL << (64 - rhs)) - 1);
	}
	void Random::setSeed(int64_t seed)
	{
		_seed = (seed ^ 0x5DEECE66DLL) & ((1LL << 48) - 1);
	}

	int Random::next(int bits)
	{
		_seed = (_seed * 0x5DEECE66DLL + 0xBLL) & ((1LL << 48) - 1);
		return (int)shr(_seed, 48 - bits);
	} 

	int Random::nextInt(int bound)
	{
		int r = next(31);
		int m = bound - 1;
		if ((bound & m) == 0)
		{
			r = (int)((bound * (int64_t)r) >> 31);
		}
		else
		{
			for (int u = r; u - (r = u % bound) + m < 0; u = next(31));
		}
		return r;
	}

	Random makeRand()
	{
		Random rnd;
		rnd.setSeed(std::chrono::system_clock::now().time_since_epoch().count());
		return rnd;
	}

	int randomInt(int min, int max)
	{
		int rnd = RANDOM.nextInt(max);

		while (rnd < min)
		{
			rnd = RANDOM.nextInt(max);
		}

		return rnd;

		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_int_distribution<int> dist(min, max - 1);

		return dist(mt);
	}

	int randomInt(int bound)
	{
		//return RANDOM.nextInt(bound);
		return randomInt(0, bound);
	}

	int randomInt(int bound, const std::vector<int>& exclude)
	{
		int rand = randomInt(bound);
		while (std::find(exclude.begin(), exclude.end(), rand) != exclude.end())
		{
			rand = randomInt(bound);
		}
		return rand;
	}

	std::string loadFileToString(const std::string& filename)
	{
		std::string str;
		std::ifstream in(filename.c_str());

		if (!in)
		{
			std::cout << "Couldn't open file \"" << filename << "\"" << std::endl;
			return "";
		}

		std::getline(in, str, std::string::traits_type::to_char_type(
			std::string::traits_type::eof()));

		return str;
	}
}

Util::Random Util::RANDOM = Util::makeRand();