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
const char CompatibilityUtil::U = 'U';
const char CompatibilityUtil::T = 'T';
const double CompatibilityUtil::one = 1.0;
const double CompatibilityUtil::minus_one = -1.0;
const arma::blas_int CompatibilityUtil::incx = 1;
void
CompatibilityUtil::compute_inner_products(Slave::VecSet const& vecset,
		arma::mat & output, const std::size_t real_dim) {
	const std::size_t num_vecs = vecset.size();
	//TODO Oversize the matrices, to prevent continuous resizing
	if(eucl.n_rows < real_dim || eucl.n_cols < num_vecs) {
		eucl.set_size(real_dim, num_vecs + 100);
	}
	if(hyp.n_rows < num_vecs) {
		hyp.set_size(num_vecs + 100);
	}
	if(output.n_rows < num_vecs || output.n_cols < num_vecs) {
		output.set_size(num_vecs + 100, num_vecs + 100);
	}
	auto v_iter = vecset.begin();
	for(std::size_t v_ind = 0; v_ind < num_vecs; ++v_ind, ++v_iter) {
		arma::vec * vec = *v_iter;
		eucl.col(v_ind) = vec->head_rows(real_dim);
		hyp(v_ind) = vec->at(real_dim);
	}
	arma::blas_int nv = static_cast<arma::blas_int>(num_vecs);
	arma::blas_int k = static_cast<arma::blas_int>(real_dim);
	arma::blas_int lda = output.n_rows;
	arma::blas_int ldeucl = eucl.n_rows;
	double* outptr = output.memptr();
	double* eucptr = eucl.memptr();
	arma_fortran(arma_dsyr)(&U, &nv, &one, hyp.memptr(), &incx, outptr, &lda);
	arma::arma_fortran(arma_dsyrk)(&U, &T, &nv, &k, &one, eucptr, &ldeucl, &minus_one, outptr, &lda);
}
}

