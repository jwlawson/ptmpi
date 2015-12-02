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

#include "ptope/polytope_check.h"

#include "codec.h"

namespace ptmpi {
class Slave {
typedef ptope::PolytopeCandidate PC;
typedef arma::vec Vec;
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
public:
	Slave(int size, int rank);
	void run();

private:
	MPI::Status _status;
	Codec _codec;
	ptope::PolytopeCandidate _pt;
	std::vector<PC> _l3;
	std::set<Vec, VecLess> _vectors;
	ptope::PolytopeCheck _chk;
	std::ofstream _l3_out;
	std::ofstream _lo_out;
	int _depth;

	bool receive();
	void send_result(const int result);
	int do_work();
	typedef std::set<Vec, VecLess>::const_iterator VIter;
	void add_till_polytope(const PC & p, VIter begin, const VIter & end,
			int depth);
};
}
#endif

