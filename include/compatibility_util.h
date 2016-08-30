/*
 * compatibility_util.h
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
#ifndef _PTMPI_COMPATIBILITY_UTIL_H_
#define _PTMPI_COMPATIBILITY_UTIL_H_

#include "slave.h"
#include "ptope/angle_check.h"

namespace ptmpi {
class CompatibilityUtil {
	static const char N;
	static const char T;
	static const char U;
	static const double one;
	static const double minus_one;
	static const arma::blas_int int_one;
	typedef Slave::CompatibilitySet CompatibilitySet;
public:
	void
	compute_compatibilities(ptope::VectorSet const& vectors, CompatibilitySet& output);

private:
	/** Workspace for computing inner products. */
	arma::podarray<double> m_inner_products;
	/** Check for valid angles. */
	ptope::AngleCheck m_angle_check;

	/** Get the smallest size needed to hold all inner products of vectors. */
	std::size_t
	priv_min_product_size( ptope::VectorSet const& vectors );
	/** Get the row, col pair for an rfp index. */
	std::pair<std::size_t, std::size_t>
	priv_rfp_index_to_ij( std::size_t const index , arma::uword const nrows );
	/** Find the inner products of all vectors in the vector set. */
	void
	priv_products( ptope::VectorSet const& vectors);
	/** Prepare to compute products.
	 * Make sure the workspace is large enough etc.
	 */
	void
	priv_products_prepare( ptope::VectorSet const& vectos );
	/** Compute the inner products of the vectors. */
	void
	priv_products_compute( ptope::VectorSet const& vectors );
	/** Use computed inner products to determine compatibilities. */
	void
	priv_compatibility_from_products( ptope::VectorSet const& vectors ,
			CompatibilitySet& output );
};
#ifdef ARMA_USE_BLAS
#ifndef ARMA_BLAS_CAPITALS
	#define arma_dsyr dsyr
	#define arma_dsfrk dsfrk
#else
	#define arma_dsyr DSYR
	#define arma_dsfrk DSFRK
#endif
extern "C" {
	void arma_fortran(arma_dsyr)(
			const char * uplo,
			const arma::blas_int * n,
			const double* alpha,
			double* x,
			const arma::blas_int* incx,
			double* A,
			const arma::blas_int* lda);
	/**
	 * Level 3 BLAS like routine for C in RFP Format.
	 * 
	 * DSFRK performs one of the symmetric rank--k operations
	 *
	 *		C := alpha*A*A**T + beta*C,
	 *
	 * or
	 *
	 *		C := alpha*A**T*A + beta*C,
	 *
	 * where alpha and beta are real scalars, C is an n--by--n symmetric
	 * matrix and A is an n--by--k matrix in the first case and a k--by--n
	 * matrix in the second case.
	 */
	void arma_fortran(arma_dsfrk)(
		const char * TRANSR,
		const char * UPLO,
		const char * TRANS,
		const arma::blas_int * N,
		const arma::blas_int * K,
		const double * ALPHA,
		const double * A,
		const arma::blas_int * LDA,
		const double * BETA,
		double * C
	);
}
#endif
}
#endif

