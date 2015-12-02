MAIN = ptmpi

CXX = mpic++
COMPILER = $(shell $(CXX) -showme:command)

# OPT is used when compiling object files
# B_OPT when compiling the final executable binary
ifeq ($(COMPILER),icpc)
# Intel optimizations are different to gcc ones
# When '-O3' is used I have noticed that there are frequently memory corruption
# issues.
CXXFLAGS = -std=c++11 -xhost
OPT = -O3 -ipo -no-prec-div
B_OPT = -O3 -ipo -no-prec-div
else
# Using cygwin -std=gnu++11 should be used rather than -std=c++11
CXXFLAGS = -Wall -std=gnu++11 -march=native
OPT = -g -O3
B_OPT = -g -O3
endif
CXXFLAGS += -DARMA_DONT_USE_WRAPPER

# Specify base directory
BASE_DIR = .

# Specify source directory
SRC_DIR = $(BASE_DIR)/src
INC_DIR = $(BASE_DIR)/include

# define the output directory for .o
OBJ_DIR = $(BASE_DIR)/build

# define any directories containing header files other than /usr/include
# e.g. I/home/include
INCLUDES = -I$(HOME)/include -I$(BASE_DIR)/include -I$(BASE_DIR)/lib/include

# define library paths in addition to /usr/lib
# e.g. -L/home/lib
LFLAGS = -L$(HOME)/lib -L$(BASE_DIR)/lib

# define any libraries to link into executable:
LIBS = -lptope -lopenblas -llapack

# find the C source files
SRCS = $(wildcard $(SRC_DIR)/*.cc)

# define the C object files
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .cc of all words in the macro SRCS
# with the .o suffix
_OBJS = $(SRCS:.cc=.o)

# Puts objs in obj_dir
OBJS = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%,$(_OBJS))

.PHONY: clean

all:   $(MAIN)

$(MAIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(B_OPT) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

# Should be made more general
install:	$(MAIN)
	cp $(MAIN) $(HOME)/bin/

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) $(OPT) $(INCLUDES) -c $<  -o $@

$(OBJS): | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	$(RM) *.o *~ $(MAIN) $(OBJ_DIR)/*.o

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
