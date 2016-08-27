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

namespace ptmpi {
class CompatibilityUtil {
static const char U;
static const char T;
static const double one;
static const double minus_one;
static const arma::blas_int incx;
public:
void
compute_inner_products(Slave::VecSet const& vecset, arma::mat & output,
		const std::size_t real_dim);

private:
arma::mat eucl;
arma::vec hyp;
};
#ifdef ARMA_USE_BLAS
#ifndef ARMA_BLAS_CAPITALS
	#define arma_dsyr dsyr
#else
	#define arma_dsyr DSYR
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
}
#endif
}
#endif

