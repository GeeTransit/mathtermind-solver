#include <bits/stdc++.h>

using namespace std;

struct Triplet {
	char a, b, c;
};

vector<Triplet> ok_triplets(
	const vector<Triplet>& pool,
	const vector<char>& nums,
	char matches
) {
	vector<Triplet> result;
	for (auto [a, b, c] : pool) {
		char count = 0;
		for (auto x : nums)
			count += (x == a) + (x == b) + (x == c);
		if (count == matches)
			result.push_back({a, b, c});
	}
	return result;
}

bool emongus(
	const vector<Triplet>& pool,
	const vector<char>& nums,
	char levels
);

int G = 0;

bool curse(
	const vector<Triplet>& pool,
	// vector<vector<char>> guesses,
	char levels
) {
	// G += 1;
	// if (G % 100000 == 0)
		// cout << G << ' ';
	if (pool.size() == 1)
		return true;
		// return [f'must be {" ".join(map(str, pool[0]))}']
	if (levels == 0)
		if (pool.size() == 1)
			return true;
			// return [f'must be {" ".join(map(str, pool[0]))}']
		else
			return false;

	array<char, 16> numsSet;
	numsSet.fill(false);
	for (auto [a, b, c] : pool) {
		numsSet[a] = true;
		numsSet[b] = true;
		numsSet[c] = true;
	}
	vector<char> nums;
	for (char i = 0; i < 16; i++)
		if (numsSet[i])
			nums.push_back(i);
	const char size = nums.size();
	for (char i = 0; i < size; i++) { char I = nums[i];
		for (char j = i+1; j < size; j++) { char J = nums[j];
			for (char k = j+1; k < size; k++) { char K = nums[k];
				for (char l = k+1; l < size; l++) { char L = nums[l];
					if (emongus(pool, {I, J, K, L}, levels)) return true;
	}}}}
	for (char i = 0; i < size; i++) { char I = nums[i];
		for (char j = i+1; j < size; j++) { char J = nums[j];
			for (char k = j+1; k < size; k++) { char K = nums[k];
				if (emongus(pool, {I, J, K}, levels)) return true;
	}}}
	for (char i = 0; i < size; i++) { char I = nums[i];
		for (char j = i+1; j < size; j++) { char J = nums[j];
			if (emongus(pool, {I, J}, levels)) return true;
	}}
	for (char i = 0; i < size; i++) { char I = nums[i];
		if (emongus(pool, {I}, levels)) return true;
	}
	return false;
}

bool emongus(
	const vector<Triplet>& pool,
	const vector<char>& nums,
	char levels
) {
	// path = {}
	bool ok = false;
	for (char matches = 0; matches <= nums.size(); matches++) {
		// new_guesses = (guesses, (nums, matches))
		const vector<Triplet>& new_pool = ok_triplets(pool, nums, matches);
		if (new_pool.size() == pool.size())
			continue;
		if (new_pool.size() == 0)
			continue;
		if (new_pool.size() == 1) {
			ok = true;
			// path[matches] = [f'must be {" ".join(map(str, new_pool[0]))}']
			continue;
		}
		bool p = curse(
			new_pool,
			// new_guesses,
			levels - 1
		);
		if (!p) {
			ok = false;
			break;
		}
		ok = true;
		// path[matches] = p
	}
	if (ok)
		return true;
		// return " ".join(map(str, nums)), path
	return false;
}

int main() {
	vector<Triplet> pool;
	for (char i = 1; i < 16; i++)
		for (char j = i+1; j < 16; j++)
			for (char k = j+1; k < 16; k++)
				pool.push_back({i, j, k});
	auto A = ok_triplets(pool, {1, 2, 3, 4}, 1);
	auto B = ok_triplets(A, {5, 6, 7, 8}, 1);
	auto C = ok_triplets(B, {9, 10, 11, 12}, 1);
	// D = ok_triplets(pool=C, nums=(1, 5), matches=2)

	cout << "levels";
	int x; cin >> x;
	cout << "calculating";
	if (curse(C, x))
		cout << "ye";
	else
		cout << "na";
}
