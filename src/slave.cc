/*
 * slave.cc
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
#include "slave.h"

#include <string>

#include "ptope/angle_check.h"
#include "ptope/calc.h"
#include "ptope/combined_check.h"
#include "ptope/duplicate_column_check.h"
#include "ptope/filtered_iterator.h"
#include "ptope/parabolic_check.h"
#include "ptope/polytope_extender.h"
#include "ptope/polytope_rebaser.h"
#include "ptope/stacked_iterator.h"
#include "ptope/elliptic_factory.h"

#include "codec.h"
#include "mpi_tags.h"
#include "compatibility_util.h"

namespace ptmpi {
namespace {
typedef ptope::StackedIterator<ptope::PolytopeRebaser, ptope::PolytopeExtender,
					ptope::PolytopeCandidate> PCtoL3;
typedef ptope::CombinedCheck3<ptope::AngleCheck, true, ptope::UniquePCCheck, true,
				ptope::DuplicateColumnCheck, false> Check;
typedef ptope::FilteredIterator<PCtoL3, ptope::PolytopeCandidate, Check, true> L3F;

unsigned long no_computed = 0;
std::chrono::duration<double> time_waited(0);
std::chrono::duration<double> max_wait(0);
std::size_t max_l3(0);
}
Slave::Slave(unsigned int total_dimension, std::ofstream && l3_os, std::ofstream && lo_os)
	: _vectors(total_dimension)
	, _l3_out(std::move(l3_os))
	, _lo_out(std::move(lo_os))
	, _added(max_depth)
{}

void
Slave::run(const bool only_compute_l3) {
	while(receive()) {
		int result = do_work(only_compute_l3);
		send_result(result);
	}
	std::cerr << "worker " << MPI::COMM_WORLD.Get_rank() << ": Average wait "
		<< (time_waited.count() / no_computed) << ", max " << max_wait.count()
		<< " with largest L3: " << max_l3
		<< std::cerr.widen('\n');
}
bool
Slave::receive() {
	static arma::podarray<double> __gram_array_cache;
	static arma::podarray<double> __vect_array_cache;
	int g_size = 0;
	MPI::COMM_WORLD.Recv(&g_size, 1, MPI::INT, MASTER, MPI::ANY_TAG, _status);
	if(_status.Get_tag() == END_TAG) {
		return false;
	}
	MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, MASTER, OK_TAG);
	__gram_array_cache.set_min_size(g_size + 2);
	MPI::COMM_WORLD.Recv(__gram_array_cache.memptr(), g_size, MPI::DOUBLE, MASTER, GRAM_TAG);
	MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, MASTER, OK_TAG);
	int v_size = 0;
	MPI::COMM_WORLD.Recv(&v_size, 1, MPI::INT, MASTER, MPI::ANY_TAG, _status);
	MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, MASTER, OK_TAG);
	__vect_array_cache.set_min_size(v_size + 2);
	MPI::COMM_WORLD.Recv(__vect_array_cache.memptr(), v_size, MPI::DOUBLE, MASTER, VECTOR_TAG);
	MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, MASTER, OK_TAG);

	_pt = _codec.decode(__gram_array_cache.memptr(), g_size, __vect_array_cache.memptr(), v_size);
	return true;
}
void
Slave::send_result(const int res) {
	auto start = std::chrono::system_clock::now();
	MPI::COMM_WORLD.Send(&res, 1, MPI::INT, MASTER, RESULT_TAG);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	time_waited += diff;
	if(diff > max_wait) max_wait = diff;
	++no_computed;
}
int
Slave::do_work(const bool only_compute_l3) {
	PCtoL3 l3_iter(_pt);
	L3F l3(std::move(l3_iter));
	const arma::uword last_vec_ind = _pt.vector_family().size();
	while(l3.has_next()) {
		auto & n = l3.next();
		if(_chk(n)) {
			n.save(_l3_out);
		} else if(!only_compute_l3) {
			_vectors.add( n.vector_family().get_ptr(last_vec_ind) );
		}
	}
	if(only_compute_l3) return 0;
	check_compatibility();
	// Need to check against _vectors.size rather than _compatibility.size as the
	// latter is not updated if there are fewer vectors than on a preious run.
	for(std::size_t i = 0, max = _vectors.size(); i < max; ++i) {
		add_till_polytope(i);
	}
	if(_vectors.size() > max_l3) {
		max_l3 = _vectors.size();
	}
	_vectors.clear();
	return 0;
}
void
Slave::add_till_polytope(std::size_t index) {
	if(_compatible[index].empty()) return;
	auto & next_pc = _pc_cache.get(0);
	const auto vec_to_add = _vectors.at( index );
	_pt.extend_by_vector(next_pc, vec_to_add);
	_added[0] = index;
	for(auto start = _compatible[index].begin(), end = _compatible[index].end();
			start != end;
			++start) {
		add_till_polytope(next_pc, start, end, 1, _added);
	}
}
void
Slave::add_till_polytope(const PC & p, CompatibleIter begin,
		const CompatibleIter & end, int depth, IndexVec & added) {
	auto & next_pc = _pc_cache.get(depth);
	std::size_t index_to_add = *begin;
	/* Check that the new index is compatible with all added vectors. */
	for(auto iter = added.begin(), max = iter + depth; iter != max; ++iter) {
		std::size_t & a = *iter;
		if(!std::binary_search(_compatible[a].begin(), _compatible[a].end(),
					index_to_add)) {
			return;
		}
	}
	const auto vec_to_add = _vectors.at( index_to_add );
	p.extend_by_vector(next_pc, vec_to_add);
	if(_chk_cache.get(depth)(next_pc)) {
		next_pc.save(_lo_out);
	} else if(depth != max_depth) {
		added[depth] = index_to_add;
		for(++begin; begin != end; ++begin) {
			add_till_polytope(next_pc, begin, end, depth + 1, added);
		}
	}
}
void
Slave::check_compatibility() {
	static ptmpi::CompatibilityUtil c_util;
	c_util.compute_compatibilities( _vectors, _compatible );
}
}

