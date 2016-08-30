/*
 * compatibility_util.cc
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
#include "compatibility_util.h"

namespace ptmpi {
const char CompatibilityUtil::N = 'N';
const char CompatibilityUtil::T = 'T';
const char CompatibilityUtil::U = 'U';
const double CompatibilityUtil::one = 1.0;
const double CompatibilityUtil::minus_one = -1.0;
const arma::blas_int CompatibilityUtil::int_one = 1;
void
CompatibilityUtil::compute_compatibilities(ptope::VectorSet const& vectors,
		Slave::CompatibilitySet& output) {

	priv_products( vectors );
	priv_compatibility_from_products( vectors , output );
}
std::size_t
CompatibilityUtil::priv_min_product_size( ptope::VectorSet const& vectors ) {
	std::size_t num = vectors.size();
	return ( num * ( num + 1 ) ) / 2;
}
std::pair<std::size_t, std::size_t>
CompatibilityUtil::priv_rfp_index_to_ij( std::size_t const index ,
		arma::uword const nrows ) {
	std::size_t rfp_ncols = ( nrows + 1 ) / 2;
	std::size_t rfp_nrows = nrows + 1 - ( nrows % 2 );
	std::size_t rfp_col = index / rfp_nrows;
	std::size_t row_cutoff = nrows / 2 + 1 + rfp_col;
	std::size_t rfp_row = index % rfp_nrows;
	std::size_t row;
	std::size_t col;
	if( rfp_row < row_cutoff ) {
		row = rfp_row;
		col = rfp_col + rfp_ncols - ( nrows % 2 );
	} else {
		col = rfp_ncols - rfp_col - 1 - ( nrows % 2 );
		row = rfp_nrows - rfp_row - 1;
	}
	return { row, col };
}
void
CompatibilityUtil::priv_products( ptope::VectorSet const& vectors ) {
	priv_products_prepare( vectors );
	priv_products_compute( vectors );
}
void
CompatibilityUtil::priv_products_prepare( ptope::VectorSet const& vectors ) {
	m_inner_products.set_min_size( priv_min_product_size(vectors) );
	m_inner_products.zeros();
}
void
CompatibilityUtil::priv_products_compute( ptope::VectorSet const& vectors ) {
	std::size_t total_dim = vectors.dimension();
	std::size_t real_dim = total_dim - 1;
	double const * vector_ptr = vectors.memptr();
	double const * hyp_part_ptr = vector_ptr + real_dim;
	arma::blas_int n = vectors.size();
	arma::blas_int k = real_dim;
	arma::blas_int lda = total_dim;
	double * out_ptr = m_inner_products.memptr();

	/* Compute hyperbolic part of product */
	arma_fortran(arma_dsfrk)(&N, &U, &T, &n, &int_one, &one, hyp_part_ptr, &lda,
			&one, out_ptr);
	/* Compute inner product of euclidean part and subtract the hyperbolic
	 * product. */
	arma_fortran(arma_dsfrk)(&N, &U, &T, &n, &k, &one, vector_ptr, &lda,
			&minus_one, out_ptr);
}
void
CompatibilityUtil::priv_compatibility_from_products( ptope::VectorSet const& vectors ,
		CompatibilitySet& comp ) {
	std::size_t max_index = priv_min_product_size( vectors );
	std::size_t num_vecs = vectors.size();
	comp.resize(num_vecs);
	for(auto & vec : comp) { vec.clear(); }
	std::size_t row, col;
	for(std::size_t index = 0; index < max_index; ++index) {
		if( m_angle_check( m_inner_products[index] ) ) {
			std::tie(row, col) = priv_rfp_index_to_ij( index , num_vecs );
			comp[col].push_back( row );
		}
	}
}
}

