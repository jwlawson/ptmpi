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

#include "ptope/angle_check.h"
#include "ptope/combined_check.h"
#include "ptope/construct_iterator.h"
#include "ptope/duplicate_column_check.h"
#include "ptope/elliptic_generator.h"
#include "ptope/filtered_iterator.h"
#include "ptope/number_dotted_check.h"
#include "ptope/polytope_extender.h"
#include "ptope/stacked_iterator.h"

#include "mpi_tags.h"

void usage(int rank) {
	if(rank == MASTER) {
		std::cout << "ptmpi [-s size]" << std::endl;
	}
}
int main(int argc, char* argv[]) {

	MPI::Init(argc, argv);
	int rank = MPI::COMM_WORLD.Get_rank();

	int opt;
	int size = 0;

	while ((opt = getopt (argc, argv, "s:")) != -1){
		switch (opt) {
			case 's':
				size = std::atoi(optarg);
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
		const std::vector<uint> angles = {2, 3, 4, 5, 6, 7, 8};
		ptope::PolytopeExtender::set_default_angles(angles);
		if(rank == MASTER) {
			typedef ptope::PolytopeCandidate PolytopeCandidate;
			struct ValidCheck {
				bool operator()(const PolytopeCandidate & p) {
					return p.valid();
				}
			};
			typedef ptope::CombinedCheck3<ValidCheck, true, ptope::DuplicateColumnCheck, false, ptope::NumberDottedCheck<0>, true> Check;

			typedef ptope::ConstructIterator<ptope::EllipticGenerator, PolytopeCandidate> EtoL0;
			typedef ptope::StackedIterator<EtoL0, ptope::PolytopeExtender, PolytopeCandidate> L0toL1;
			typedef ptope::FilteredIterator<L0toL1, PolytopeCandidate, Check, true> L1F;
			typedef ptope::FilteredIterator<L1F, PolytopeCandidate, ptope::AngleCheck, true> L1Filtered;
			typedef ptope::FilteredPrintIterator<L1Filtered, PolytopeCandidate, ptope::PolytopeCheck, false> L1NoP;

			typedef ptope::StackedIterator<L1NoP, ptope::PolytopeExtender, PolytopeCandidate> L1toL2;
			typedef ptope::FilteredIterator<L1toL2, PolytopeCandidate, Check, true> L2F;
			typedef ptope::FilteredIterator<L2F, PolytopeCandidate, ptope::AngleCheck, true> L2Filtered;
			typedef ptope::FilteredPrintIterator<L2Filtered, PolytopeCandidate, ptope::PolytopeCheck, false> L2NoP;

			ptope::EllipticGenerator e(size);
			EtoL0 l0(e);
			L0toL1 l1(std::move(l0));
			L1F l1fa(std::move(l1));
			L1Filtered l1f(std::move(l1fa), angles);
			std::string ol1("/extra/var/users/njcz19/ptope/l1.");
			ol1.append(std::to_string(size));
			ol1.append(".poly");
			std::ofstream outl1(ol1);
			L1NoP l1np(std::move(l1f), outl1);
			L1toL2 l2(std::move(l1np));
			L2F l2fa(std::move(l2));
			L2Filtered l2f(std::move(l2fa), angles);
			std::string ol2("/extra/var/users/njcz19/ptope/l2.");
			ol2.append(std::to_string(size));
			ol2.append(".poly");
			std::ofstream outl2(ol2);
			L2NoP l2np(std::move(l2f), outl2);

			ptmpi::Master<L2NoP> master(std::move(l2np));
			master.run();
		} else {
			ptmpi::Slave slave(size, rank);
			slave.run();
		}

	} else {
		usage(rank);
	}

	MPI::Finalize();
	return 0;
}

		