/*
 * codec.cc
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
#include "codec.h"

namespace ptmpi {
int
Codec::size_gram(const PolytopeCandidate & p){
	int result = p.gram().n_cols;
	result *= result;
	return result;
}
const double *
Codec::encode_gram(const PolytopeCandidate & p){
	return p.gram().memptr();
}
int
Codec::size_vectors(const PolytopeCandidate & p){
	int x = p.vector_family().dimension();
	int y = p.vector_family().size();
	return x * y;
}
const double *
Codec::encode_vectors(const PolytopeCandidate & p){
	return p.vector_family().underlying_matrix().memptr();
}
ptope::PolytopeCandidate
Codec::decode(const double * gram, int g_size, const double * vectors, int v_size){
	int gram_size = std::sqrt(g_size);
	int vector_height = v_size / gram_size;
	ptope::PolytopeCandidate result(gram, gram_size, vectors, vector_height,
			gram_size);
	return result;
}
}

