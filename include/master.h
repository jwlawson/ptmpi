/*
 * master.h
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
#ifndef _PTMPI_MASTER_H_
#define _PTMPI_MASTER_H_

#include <mpi.h>

#include "ptope/polytope_candidate.h"

#include "codec.h"
#include "mpi_tags.h"

namespace ptmpi {
template <class It>
class Master {
	typedef ptope::PolytopeCandidate PolytopeCandidate;
public:
	Master(It && iter)
		: _iter(iter),
			_num_proc(MPI::COMM_WORLD.Get_size()) {}
	/**
	 * Go through the iterator and pass polytopes to worker threads.
	 */
	void run();

private:
	It _iter;
	int _num_proc;
	MPI::Status _status;
	Codec _codec;
	std::chrono::duration<double> _time_waited{0};
	unsigned long _no_computed = 0;
	/**
	 * Send the polytope to the specified worker thread.
	 */
	void
	send_matrix(const ptope::PolytopeCandidate & matrix, const int worker);
	/**
	 * Wait for a result from a worker. Once a result is obtained it is passed to
	 * handle_result, before another polytope is sent to the worker.
	 */
	int
	receive_result();
	/**
	 * Send shutdown signal to all worker threads.
	 */
	void
	send_shutdown();
};
template <class It>
void
Master<It>::run() {
	/* 
	 * Keep track of how many tasks were originally submitted. It could happen
	 * that fewer tasks are generated and sent than there are cores, so don't
	 * want to be waiting for tasks to return which were never submitted.
	 */
	int submitted = 1;
	PolytopeCandidate next;
	/* Send initial matrices to workers. */
	for(int i = 1; i < _num_proc && _iter.has_next(); ++i) {
		next = _iter.next();
		send_matrix(next, i);
		submitted++;
	}
	while(_iter.has_next()) {
		/* Might as well compute the next polytope while waiting. */
		next = _iter.next();
		receive_result();
		int worker = _status.Get_source();
		send_matrix(next, worker);
	}
	/* Wait for remaining tasks. */
	for(int i = 1; i < submitted; ++i) {
		receive_result();
	}
	send_shutdown();
	std::cerr << "master: Average wait " << (_time_waited.count() / _no_computed)
		<< std::endl;
}
template <class It>
void
Master<It>::send_matrix(const PolytopeCandidate & matrix, const int worker) {
	int g_size = _codec.size_gram(matrix);
	MPI::COMM_WORLD.Send(&g_size, 1, MPI::INT, worker, SIZE_TAG);
	MPI::COMM_WORLD.Recv(NULL, 0, MPI::BYTE, worker, OK_TAG);
	const double * gram = _codec.encode_gram(matrix);
	MPI::COMM_WORLD.Send(gram, g_size, MPI::DOUBLE, worker, GRAM_TAG);
	MPI::COMM_WORLD.Recv(NULL, 0, MPI::BYTE, worker, OK_TAG);
	int v_size = _codec.size_vectors(matrix);
	MPI::COMM_WORLD.Send(&v_size, 1, MPI::INT, worker, SIZE_TAG);
	MPI::COMM_WORLD.Recv(NULL, 0, MPI::BYTE, worker, OK_TAG);
	const double * vecs = _codec.encode_vectors(matrix);
	MPI::COMM_WORLD.Send(vecs, v_size, MPI::DOUBLE, worker, VECTOR_TAG);
	MPI::COMM_WORLD.Recv(NULL, 0, MPI::BYTE, worker, OK_TAG);
	/* Note: do not need to delete pointers as they are managed by the
	 * PolytopeCandidate instance. */
}
template <class It>
int
Master<It>::receive_result() {
	int result;
	auto start = std::chrono::system_clock::now();
	MPI::COMM_WORLD.Recv(&result, 1, MPI::INT, MPI::ANY_SOURCE, RESULT_TAG,
			_status);
	auto end = std::chrono::system_clock::now();
	_time_waited += (end - start);
	++_no_computed;
	return result;
}
template <class It>
void
Master<It>::send_shutdown() {
	int number = MPI::COMM_WORLD.Get_size();
	for(int i = 1; i < number; ++i) {
		MPI::COMM_WORLD.Send(NULL, 0, MPI::BYTE, i, END_TAG);
	}
}
} 
#endif

