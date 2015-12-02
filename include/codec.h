/*
 * codec.h
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
#ifndef _PTMPI_CODEC_H_
#define _PTMPI_CODEC_H_

#include "ptope/polytope_candidate.h"

namespace ptmpi {
class Codec {
typedef ptope::PolytopeCandidate PolytopeCandidate;
public:
	/**
	 * Get the size of the encoded array for the gram matrix.
	 */
	int
	size_gram(const PolytopeCandidate & p);
	/**
	 * Get a double array representing the gram matrix.
	 */
	const double *
	encode_gram(const PolytopeCandidate & p);
	/**
	 * Get the size of the encoded array for the vector family.
	 */
	int
	size_vectors(const PolytopeCandidate & p);
	/**
	 * Get a double array representing the vector family.
	 */
	const double *
	encode_vectors(const PolytopeCandidate & p);
	/**
	 * Decode the passed arrays to a PolytopCandidate.
	 */
	PolytopeCandidate
	decode(const double * gram, int g_size, const double * vectors, int v_size);
};
}
#endif

