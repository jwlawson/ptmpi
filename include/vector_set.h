/*
 * vector_set.h
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
/**
 * A set of vectors. As with other sets, this ensures that each vector in the
 * set is unique. The vectors are not sorted in any way, but are stored in a
 * consistent ordering given by how they were added.
 */
#pragma once
#ifndef _PTMPI_VECTOR_SET_H_
#define _PTMPI_VECTOR_SET_H_

#include <boost/container/flat_set.hpp>

#include "armadillo"

namespace ptmpi {
class VectorSet {
	struct VecPtrComparator {
		VecPtrComparator( arma::uword const& size );
		// true if lhs < rhs, false otherwise.
		bool
		operator()( double const * const lhs, double const * const rhs ) const;
	private:
		arma::uword const m_size;
	};
public:
	//TODO iterators

	VectorSet( arma::uword const& dimension );

	/**
	 * Clear the set.
	 *
	 * No memory is released.
	 */
	void
	clear();
	/**
	 * Get the number of vectors in the set.
	 */
	std::size_t
	size();
	/**
	 * Check whether the provided vector is already in the set.
	 * Return: true if present in set
	 * Complexity: logarithmic
	 */
	bool
	contains( double const * vec_ptr ) const;
	bool
	contains( arma::vec const& vec ) const;
	/**
	 * Insert the provided vector into the set.
	 * Return: true if inserted, false if already present
	 * Complexity: Logarithmic search plus linear insertion
	 */
	bool
	add( double const * vec_ptr );
	bool
	add( arma::vec const& vec);
	/**
	 * Get a (strict) vector from the set at the specified index.
	 * This vector uses the memory in the set, so cannot be resized and should not
	 * be changed.
	 *
	 * Also, if the vector store is resized, this vector will be invalidated.
	 */
	arma::vec
	at( arma::uword const& index );
	double *
	ptr_at( arma::uword const& index );
	double const *
	ptr_at( arma::uword const& index ) const;
	/**
	 * Get a pointer to the underlying matrix of vectors.
	 */
	double *
	memptr();
	double const *
	memptr() const;

private:
	arma::uword const m_dimension;
	boost::container::flat_set<double const *, VecPtrComparator> m_ordered_pointers;
	arma::mat m_vector_store;
	double const * m_current_data_ptr;

	bool
	priv_insert_without_resize( double const * vec_ptr );
	void
	priv_resize_extend();
};


inline
void
VectorSet::clear() {
	m_ordered_pointers.clear();
}
inline
std::size_t
VectorSet::size() {
	return m_ordered_pointers.size();
}
inline
double *
VectorSet::memptr() {
	return m_vector_store.memptr();
}
inline
double const *
VectorSet::memptr() const {
	return m_current_data_ptr;
}
inline
arma::vec
VectorSet::at( arma::uword const& index ) {
	arma::vec result( ptr_at(index) , m_dimension , false , true );
	return result;
}
inline
double *
VectorSet::ptr_at( arma::uword const& index ) {
	return memptr() + index * m_dimension;
}
inline
double const *
VectorSet::ptr_at( arma::uword const& index ) const {
	return memptr() + index * m_dimension;
}
}
#endif

