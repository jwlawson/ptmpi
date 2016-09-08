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
	: _vectors(total_dimension, 9500)
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
	//static ptope::BloomPCCheck unique_check;
	PCtoL3 l3_iter(_pt);
	L3F l3(std::move(l3_iter));
	const arma::uword last_vec_ind = _pt.vector_family().size();
	while(l3.has_next()) {
		auto & n = l3.next();
		//if ( unique_check(n) ) {
		if(_polytope_check(n)) {
			n.save(_l3_out);
		} else {
			_vectors.add( n.vector_family().get_ptr(last_vec_ind) );
		}
		//}
	}
	if(_vectors.size() > max_l3) { max_l3 = _vectors.size(); }
	if( !only_compute_l3 ) { 
		_compatible.from( _vectors );
		// Don't actually need to check the last one because of how it will have been
		// checked in all others, so the only thing to check would be just adding the
		// last vector itself, which was already checked in above loop.
		for(std::size_t i = 0, max = _vectors.size() - 1; i < max; ++i) {
			add_till_polytope(i);
		}
	}
	_vectors.clear();
	return 0;
}
void
Slave::add_till_polytope(std::size_t index) {
	std::size_t next_ind = _compatible.next_compatible_to( index, 0 );
	if( next_ind == index ) { return; }
	auto & next_pc = _pc_cache.get(0);
	auto const& vec_to_add = _vectors.at( index );
	_pt.extend_by_vector(next_pc, vec_to_add);
	_added[0] = index;
	while ( next_ind != index ) {
		add_till_polytope( next_pc, next_ind, 1, _added);
		next_ind = _compatible.next_compatible_to( index, next_ind );
	}
}
void
Slave::add_till_polytope(const PC & p, std::size_t index_to_add,
		 int depth, IndexVec & added) {
	auto & next_pc = _pc_cache.get(depth);
	/* Check that the new index is compatible with all added vectors. */
	for(auto iter = added.cbegin(), max = iter + depth; iter != max; ++iter) {
		std::size_t const& a = *iter;
		if( !_compatible.are_compatible( index_to_add , a ) ) { return; }
	}
	auto const& vec_to_add = _vectors.at( index_to_add );
	p.extend_by_vector(next_pc, vec_to_add);
	if(_polytope_check(next_pc)) {
		next_pc.save(_lo_out);
	} else if(depth != max_depth) {
		added[depth] = index_to_add;
		std::size_t next_ind = _compatible.next_compatible_to( index_to_add , 0 );
		while ( next_ind != index_to_add ) {
			add_till_polytope( next_pc, next_ind, depth + 1, added );
			next_ind = _compatible.next_compatible_to( index_to_add , next_ind );
		}
	}
}
}

