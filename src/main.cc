/*
 * main.cc
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
#include "master.h"
#include "slave.h"

#include <unistd.h>
#include <fstream>
#include <string>

#include "ptope/angles.h"
#include "ptope/angle_check.h"
#include "ptope/combined_check.h"
#include "ptope/construct_iterator.h"
#include "ptope/duplicate_column_check.h"
#include "ptope/elliptic_generator.h"
#include "ptope/filtered_iterator.h"
#include "ptope/number_dotted_check.h"
#include "ptope/parabolic_check.h"
#include "ptope/polytope_extender.h"
#include "ptope/stacked_iterator.h"
#include "ptope/unique_matrix_check.h"

#include "mpi_tags.h"

/* These templates provide the nested iterators which handle, create and filter the stream of
 * matrices which might or might not be polytopes. */
namespace iter {
typedef ptope::CombinedCheck3<ptope::AngleCheck, true,
	ptope::DuplicateColumnCheck, false, ptope::UniquePCCheck,
	true> Check1;
typedef ptope::CombinedCheck2<Check1, true, ptope::ParabolicCheck, false> Check;

namespace matrix {
typedef ptope::PolytopeExtender L0toL1;
typedef ptope::FilteredIterator<L0toL1, ptope::PolytopeCandidate, Check, true> L1F;
typedef ptope::FilteredPrintIterator<L1F, ptope::PolytopeCandidate, ptope::PolytopeCheck, false> L1NoP;

typedef ptope::StackedIterator<L1NoP, ptope::PolytopeExtender, ptope::PolytopeCandidate> L1toL2;
typedef ptope::FilteredIterator<L1toL2, ptope::PolytopeCandidate, Check, true> L2F;
typedef ptope::FilteredPrintIterator<L2F, ptope::PolytopeCandidate, ptope::PolytopeCheck, false> L2NoP;
}
namespace generated {
typedef ptope::ConstructIterator<ptope::EllipticGenerator, ptope::PolytopeCandidate> EtoL0;
typedef ptope::StackedIterator<EtoL0, ptope::PolytopeExtender, ptope::PolytopeCandidate> L0toL1;
typedef ptope::FilteredIterator<L0toL1, ptope::PolytopeCandidate, Check, true> L1F;
typedef ptope::FilteredPrintIterator<L1F, ptope::PolytopeCandidate, ptope::PolytopeCheck, false> L1NoP;

typedef ptope::StackedIterator<L1NoP, ptope::PolytopeExtender, ptope::PolytopeCandidate> L1toL2;
typedef ptope::FilteredIterator<L1toL2, ptope::PolytopeCandidate, Check, true> L2F;
typedef ptope::FilteredPrintIterator<L2F, ptope::PolytopeCandidate, ptope::PolytopeCheck, false> L2NoP;
}
}
void
usage(int rank) {
	if(rank == MASTER) {
		std::cout
			<< "ptmpi [-s size] [-abde] [-f directory] [-p prefix] [-x suffix] [-3]" << std::endl
			<< " -s Specify the dimension of the space to search in" << std::endl
			<< " -a, -b, -d, -e" << std::endl
			<< "    Specify the initial elliptic subdiagram to start with (by Dynkin type)," << std::endl
			<< "    defaults to using all possible elliptic subdiagrams." << std::endl
			<< " -f Specify directory to store results" << std::endl
			<< " -p Specify result file prefix" << std::endl
			<< " -x Specify result file suffix (will be appended by mpi rank)" << std::endl
			<< " -3 Only compute up to L3, and don't attempt to extend the L3 polytopes" << std::endl;
	}
}
enum Start {
	All, A, B, D, E
};
std::string
filename(const std::string & dir, const std::string & prefix, const int f,
		const int size, const std::string & suffix) {
	std::string result(dir);
	result.reserve(dir.size() + prefix.size() + suffix.size() + 8);
	result.append("/").append(prefix).append(std::to_string(f)).append(".")
		.append(std::to_string(size)).append(suffix);

	const int rank = MPI::COMM_WORLD.Get_rank();
	if(rank != MASTER) result.append(".").append(std::to_string(rank));

	return result;
}
iter::generated::L2NoP
generated_master_iter(const int size, std::ofstream & l1_os,
		std::ofstream & l2_os) {
	using namespace iter::generated;
	ptope::EllipticGenerator e(size);
	EtoL0 l0(e);
	L0toL1 l1(std::move(l0));
	L1F l1f(std::move(l1));
	L1NoP l1np(std::move(l1f), l1_os);
	L1toL2 l2(std::move(l1np));
	L2F l2f(std::move(l2));
	L2NoP l2np(std::move(l2f), l2_os);
	return l2np;
}
iter::matrix::L2NoP
matrix_master_iter(const arma::mat & m, std::ofstream & l1_os,
		std::ofstream & l2_os) {
	using namespace iter::matrix;
	L0toL1 l1(m);
	L1F l1f(std::move(l1));
	L1NoP l1np(std::move(l1f), l1_os);
	L1toL2 l2(std::move(l1np));
	L2F l2f(std::move(l2));
	L2NoP l2np(std::move(l2f), l2_os);
	return l2np;
}
template<class Iterator>
void
start_master(Iterator && it) {
	ptmpi::Master<Iterator> master(std::move(it));
	master.run();
}
/* TODO input checking */
int
main(int argc, char* argv[]) {

	MPI::Init(argc, argv);
	int rank = MPI::COMM_WORLD.Get_rank();

	int opt;
	int size = 0;
	Start initial = All;
	std::string dir = "/extra/var/users/njcz19/ptope";
	std::string prefix = "l";
	std::string suffix = ".poly";
	bool only_l3 = false;

	while ((opt = getopt (argc, argv, "s:abdef:p:x:3")) != -1){
		switch (opt) {
			case 's':
				size = std::atoi(optarg);
				break;
			case 'a':
				initial = A;
				break;
			case 'b':
				initial = B;
				break;
			case 'd':
				initial = D;
				break;
			case 'E':
				initial = E;
				break;
			case 'f':
				dir = optarg;
				break;
			case 'p':
				prefix = optarg;
				break;
			case 'x':
				suffix = optarg;
				break;
			case '3':
				only_l3 = true;
				break;
			case '?':
				usage(rank);
				return 1;
			default:
				usage(rank);
				return 2;
		}
	}

	if(size > 1) {
		ptope::Angles::get().set_angles({2, 3, 4, 5, 8, 10});
		if(rank == MASTER) {
			std::string l1_f = filename(dir, prefix, 1, size, suffix);
			std::ofstream l1_os(l1_f);
			if(!l1_os.is_open()) {
				std::cerr << "Error opening file " << l1_f << std::endl;
				return -1;
			}
			std::string l2_f = filename(dir, prefix, 2, size, suffix);
			std::ofstream l2_os(l2_f);
			if(!l2_os.is_open()) {
				std::cerr << "Error opening file " << l2_f << std::endl;
				return -1;
			}
			switch(initial) {
				case A:
					start_master(matrix_master_iter(ptope::elliptic_factory::type_a(size),
								l1_os, l2_os));
					break;
				case B:
					start_master(matrix_master_iter(ptope::elliptic_factory::type_b(size),
								l1_os, l2_os));
					break;
				case D:
					start_master(matrix_master_iter(ptope::elliptic_factory::type_d(size),
								l1_os, l2_os));
					break;
				case E:
					start_master(matrix_master_iter(ptope::elliptic_factory::type_e(size),
								l1_os, l2_os));
					break;
				case All:
				default:
					start_master(generated_master_iter(size, l1_os, l2_os));
					break;
			}
		} else {
			std::string l3_f = filename(dir, prefix, 3, size, suffix);
			std::ofstream l3_os(l3_f);
			if(!l3_os.is_open()) {
				std::cerr << "Error opening file " << l3_f << std::endl;
				return -1;
			}
			std::string lo_f = filename(dir, prefix, 4, size, suffix);
			std::ofstream lo_os(lo_f);
			if(!lo_os.is_open()) {
				std::cerr << "Error opening file " << lo_f << std::endl;
				return -1;
			}
			ptmpi::Slave slave(size + 1, std::move(l3_os), std::move(lo_os));
			slave.run(only_l3);
		}

	} else {
		usage(rank);
	}

	MPI::Finalize();
	return 0;
}

