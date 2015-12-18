/*
 * slave.h
 * Copyright 2015 John Lawson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#ifndef _PTMPI_SLAVE_H_
#define _PTMPI_SLAVE_H_

#include <mpi.h>
#include <vector>
#include <set>
#include <unordered_set>

#include "boost/container/flat_set.hpp"

#include "ptope/polytope_check.h"

#include "codec.h"

namespace ptmpi {
class Slave {
typedef ptope::PolytopeCandidate PC;
typedef arma::vec Vec;
static constexpr int max_depth = 5;
struct VecLess {
	static constexpr double error = 1e-10;
	/** true if lhs < rhs, false otherwise. */
	bool operator()( const Vec & lhs, const Vec & rhs ) const {
		bool result = true;
		for(arma::uword i = 0; result && i < lhs.size(); ++i) {
			result = ( (lhs(i) - rhs(i)) < error);
		}
		return result;
	}
};
struct PCCache {
	PCCache () : _cache(max_depth + 1) {}
	PC & get(int d) {
		return _cache[d];
	}
	std::vector<PC> _cache;
};
struct ChkCache {
	ChkCache () : _cache(max_depth + 1) {}
	ptope::PolytopeCheck & get(int d) {
		return _cache[d];
	}
	std::vector<ptope::PolytopeCheck> _cache;
};
//typedef std::set<Vec, VecLess> VecSet;
typedef boost::container::flat_set<Vec, VecLess> VecSet;
typedef VecSet::const_iterator VIter;
public:
	Slave(int size, int rank);
	void run();

private:
	MPI::Status _status;
	Codec _codec;
	ptope::PolytopeCandidate _pt;
	std::vector<PC> _l3;
	VecSet _vectors;
	ptope::PolytopeCheck _chk;
	std::ofstream _l3_out;
	std::ofstream _lo_out;
	int _depth;
	PCCache _cache;
	ChkCache _chk_cache;

	bool receive();
	void send_result(const int result);
	int do_work();
	void add_till_polytope(const PC & p, VIter begin, const VIter & end,
			int depth);
};
}
#endif

