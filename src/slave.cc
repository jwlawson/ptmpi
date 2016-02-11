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
typedef ptope::CombinedCheck2<ptope::AngleCheck, true,
				ptope::DuplicateColumnCheck, false> Check;
typedef ptope::FilteredIterator<PCtoL3, ptope::PolytopeCandidate, Check, true> L3F;

/* Argh global(ish) vars... */
Check __valid_chk;
unsigned long no_computed = 0;
std::chrono::duration<double> time_waited(0);
std::chrono::duration<double> max_wait(0);
std::size_t max_l3(0);
ptope::ParabolicCheck __para_chk;
ptope::BloomPCCheck __unique_chk;
}
Slave::Slave(std::ofstream && l3_os, std::ofstream && lo_os)
	: _l3_out(std::move(l3_os)),
		_lo_out(std::move(lo_os)),
		_added(max_depth),
		_angles(ptope::Angles::get().inner_products().begin(),
				ptope::Angles::get().inner_products().end()) {}
void
Slave::run() {
	while(receive()) {
		int result = do_work();
		send_result(result);
	}
	std::cerr << "worker " << MPI::COMM_WORLD.Get_rank() << ": Average wait "
		<< (time_waited.count() / no_computed) << ", max " << max_wait.count()
		<< " with largest L3: " << max_l3
		<< std::endl;
}
bool
Slave::receive() {
	int g_size;
	MPI::COMM_WORLD.Recv(&g_size, 1, MPI::INT, MASTER, MPI::ANY_TAG, _status);
	if(_status.Get_tag() == END_TAG) {
		return false;
	}
	MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, MASTER, OK_TAG);
	double * g_arr = new double[g_size];
	MPI::COMM_WORLD.Recv(g_arr, g_size, MPI::DOUBLE, MASTER, GRAM_TAG);
	MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, MASTER, OK_TAG);
	int v_size;
	MPI::COMM_WORLD.Recv(&v_size, 1, MPI::INT, MASTER, MPI::ANY_TAG, _status);
	MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, MASTER, OK_TAG);
	double * v_arr = new double[v_size];
	MPI::COMM_WORLD.Recv(v_arr, v_size, MPI::DOUBLE, MASTER, VECTOR_TAG);
	MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, MASTER, OK_TAG);

	_pt = _codec.decode(g_arr, g_size, v_arr, v_size);
	delete [] g_arr;
	delete [] v_arr;
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
Slave::do_work() {
	PCtoL3 l3_iter(_pt);
	L3F l3(std::move(l3_iter));
	const arma::uword last_vec_ind = _pt.vector_family().size();
	while(l3.has_next()) {
		auto & n = l3.next();
		if(__unique_chk(n) && !__para_chk(n)) {
			if(_chk(n)) {
					n.save(_l3_out);
			} else {
				Vec * p = new Vec(n.vector_family().get(last_vec_ind));
				bool success = _vectors.insert(p).second;
				if(!success) {
					delete p;
				}
			}
		}
	}
	check_compatibility();
	// Need to check against _vectors.size rather than _compatibility.size as the
	// latter is not updated if there are fewer vectors than on a preious run.
	for(std::size_t i = 0, max = _vectors.size(); i < max; ++i) {
		add_till_polytope(i);
	}
	if(_vectors.size() > max_l3) {
		max_l3 = _vectors.size();
	}
	for(Vec * p : _vectors) {
		delete p;
	}
	_vectors.clear();
	return 0;
}
void
Slave::add_till_polytope(std::size_t index) {
	if(_compatible[index].empty()) return;
	auto & next_pc = _pc_cache.get(0);
	const auto vec_iter = _vectors.begin() + index;
	const auto & vec_to_add = **(vec_iter);
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
	const auto & vec_add = **(_vectors.begin() + index_to_add);
	p.extend_by_vector(next_pc, vec_add);
	//if(!__para_chk(next_pc)) {
		if(_chk_cache.get(depth)(next_pc)) {
				next_pc.save(_lo_out);
		} else if(depth != max_depth) {
			added[depth] = index_to_add;
			for(++begin; begin != end; ++begin) {
				add_till_polytope(next_pc, begin, end, depth + 1, added);
			}
		}
	//}
}
namespace {
constexpr double error = 1e-10;
#ifdef __PTMPI_SLAVE_ANGLE_VEC
ptope::comparator::DoubleLess __d_less;
#endif
}
bool
Slave::valid_angle(const arma::vec & a, const arma::vec & b) const {
	const double val = ptope::calc::mink_inner_prod(a, b);
	bool dotted = val < -1;
	bool angle = !dotted && val < error &&
#ifdef __PTMPI_SLAVE_ANGLE_VEC
		std::binary_search(_angles.begin(), _angles.end(), val, __d_less);
#else
		_angles.find(val) != _angles.end();
#endif
	return angle;
}
void
Slave::check_compatibility() {
	_compatible.resize(_vectors.size());
	for(auto & vec : _compatible) vec.clear();
	std::size_t i = 0;
	for(auto start = _vectors.begin(), end = _vectors.end();
			start != end;
			++i, ++start) {
		std::size_t j = i + 1;
		for(auto vit = start + 1;
				vit != end;
				++j, ++vit) {
			if(valid_angle(**start, **vit)) {
				_compatible[i].push_back(j);
			}
		}
	}
}
}

