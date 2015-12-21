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

#include "ptope/angles.h"
#include "ptope/angle_check.h"
#include "ptope/combined_check.h"
#include "ptope/duplicate_column_check.h"
#include "ptope/filtered_iterator.h"
#include "ptope/polytope_extender.h"
#include "ptope/polytope_rebaser.h"
#include "ptope/stacked_iterator.h"
#include "ptope/unique_matrix_check.h"

#include "codec.h"
#include "mpi_tags.h"

namespace ptmpi {
namespace {
typedef ptope::StackedIterator<ptope::PolytopeRebaser, ptope::PolytopeExtender,
					ptope::PolytopeCandidate> PCtoL3;
typedef ptope::CombinedCheck2<ptope::DuplicateColumnCheck, false,
				ptope::AngleCheck, true> Check;
typedef ptope::FilteredIterator<PCtoL3, ptope::PolytopeCandidate, Check, true> L3F;

std::string l3_filename(int size, int rank) {
	std::string result("/extra/var/users/njcz19/ptope/l3.");
	result.append(std::to_string(size));
	result.append(".poly.");
	result.append(std::to_string(rank));
	return result;
}
std::string lo_filename(int size, int rank) {
	std::string result("/extra/var/users/njcz19/ptope/lo.");
	result.append(std::to_string(size));
	result.append(".poly.");
	result.append(std::to_string(rank));
	return result;
}
/* Argh global(ish) vars... */
Check valid_chk;
unsigned long no_computed = 0;
std::chrono::duration<double> time_waited(0);
std::chrono::duration<double> max_wait(0);
std::size_t max_l3(0);
ptope::UniqueMatrixCheck __unique_chk;
}
Slave::Slave(int size, int rank)
	: _l3_out(l3_filename(size, rank)),
		_lo_out(lo_filename(size, rank)) {}
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
	_l3.clear();
	_vectors.clear();
	PCtoL3 l3_iter(_pt);
	L3F l3(std::move(l3_iter));
	const arma::uword last_vec_ind = _pt.vector_family().size();
	while(l3.has_next()) {
		auto n = l3.next();
		if(__unique_chk(n)) {
			if(_chk(n)) {
				if(_chk.used_all()) {
					_l3_out << n << _l3_out.widen('\n');
				}
			} else {
				_vectors.emplace(n.vector_family().get(last_vec_ind));
				_l3.emplace_back(std::move(n));
			}
		}
	}
	for(auto & pc : _l3) {
		// Recursively add vectors till get polytopes
		add_till_polytope(pc, _vectors.begin(), _vectors.end(), 0);
	}
	if(_l3.size() > max_l3) {
		max_l3 = _l3.size();
	}
	return 0;
}
void
Slave::add_till_polytope(const PC & p, VIter begin, const VIter & end,
		int depth) {
	/* Not really interested in polytopes with huge numbers of faces. */
	if(depth > max_depth) return;
	auto & next = _cache.get(depth);
	p.extend_by_vector(next, *begin);
	if(valid_chk(next)) {
		if(_chk_cache.get(depth)(next)) {
			if(_chk_cache.get(depth).used_all()) {
				_lo_out << next << _lo_out.widen('\n');
			}
		} else {
			for(; begin != end; ++begin) {
				add_till_polytope(next, begin, end, depth + 1);
			}
		}
	}
}
namespace {
constexpr double error = 1e-14;
struct DLess {
	bool
	operator()(const double & lhs, const double & rhs) {
		return (lhs < rhs - error);
	}
} __d_less;
double mink_inner_prod(const arma::vec & a, const arma::vec & b) {
	double sq = 0;
	arma::uword max = a.size() - 1;
	for(arma::uword i = 0; i < max; ++i) {
		sq += a(i) * b(i);
	}
	sq -= a(max) * b(max);
	return sq;
}
}
bool
Slave::valid_angle(const arma::vec & a, const arma::vec & b) const {
	auto & angles = ptope::Angles::get().inner_products();
	const double val = mink_inner_prod(a, b);
	return std::binary_search(angles.begin(), angles.end(), val, __d_less);
}
}

