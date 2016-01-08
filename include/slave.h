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
#include "ptope/unique_matrix_check.h"

#include "codec.h"

namespace ptmpi {
class Slave {
typedef ptope::PolytopeCandidate PC;
typedef arma::vec Vec;
static constexpr int max_depth = 2;
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
template<class T>
struct Cache {
	Cache () : _cache(max_depth + 1) {}
	T & get(int d) {
		return _cache[d];
	}
	std::vector<T> _cache;
};
typedef Cache<PC> PCCache;
typedef Cache<ptope::PolytopeCheck> ChkCache;
typedef Cache<ptope::UniqueMPtrCheck> UMCCache;
//typedef std::set<Vec, VecLess> VecSet;
typedef boost::container::flat_set<Vec, VecLess> VecSet;
//typedef VecSet::const_iterator VIter;
typedef std::vector<std::vector<std::size_t>> CompatibilitySet;
typedef std::vector<std::size_t>::const_iterator CompatibleIter;

public:
	Slave(int size, int rank);
	void run();

private:
	MPI::Status _status;
	Codec _codec;
	ptope::PolytopeCandidate _pt;
	VecSet _vectors;
	CompatibilitySet _compatible;
	ptope::PolytopeCheck _chk;
	std::ofstream _l3_out;
	std::ofstream _lo_out;
	PCCache _pc_cache;
	ChkCache _chk_cache;
	UMCCache _unique_cache;

	/** Get next work unit from master. */
	bool
	receive();
	/** Ask master for more work. */
	void
	send_result(const int result);
	/** Compute all polytopes form the most recently received thing. */
	int
	do_work();
	/** Add vertices until the polytope is a polytope (or times out). */
	void
	add_till_polytope(std::size_t index);
	void
	add_till_polytope(const PC & p, CompatibleIter begin,
			const CompatibleIter & end, int depth);
	/** Check whether the two vectors meet at 'nice' angle */
	bool
	valid_angle(const arma::vec & a, const arma::vec & b) const;
	/** Construct the set showing compatibility of the possible vectors to add to
	 * polytopes. */
	void
	check_compatibility();
};
}
#endif

