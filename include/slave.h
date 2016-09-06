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

#include "boost/container/flat_set.hpp"

#include "ptope/angles.h"
#include "ptope/compatibility_info.h"
#include "ptope/polytope_check.h"
#include "ptope/unique_matrix_check.h"
#include "ptope/vector_set.h"

#include "codec.h"

namespace ptmpi {
class Slave {
typedef ptope::PolytopeCandidate PC;
typedef arma::vec Vec;
static constexpr int max_depth = 3;
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
typedef std::vector<std::size_t> IndexVec;

public:
	Slave(unsigned int total_dimension, std::ofstream && l3_filename,
			std::ofstream && lo_filename);
	void run(const bool only_compute_l3 = false);

private:
	ptope::VectorSet _vectors;
	MPI::Status _status;
	Codec _codec;
	ptope::PolytopeCandidate _pt;
	ptope::CompatibilityInfo _compatible;
	ptope::PolytopeCheck _chk;
	std::ofstream _l3_out;
	std::ofstream _lo_out;
	PCCache _pc_cache;
	ChkCache _chk_cache;
	IndexVec _added;

	/** Get next work unit from master. */
	bool
	receive();
	/** Ask master for more work. */
	void
	send_result(const int result);
	/** Compute all polytopes form the most recently received thing. */
	int
	do_work(const bool only_compute_l3);
	/** Add vertices until the polytope is a polytope (or times out). */
	void
	add_till_polytope(std::size_t index);
	void
	add_till_polytope(const PC & p, std::size_t index_to_add, int depth,
			IndexVec & added);
};
}
#endif

