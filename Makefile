MAIN = ptmpi

CXX = mpic++
COMPILER = $(shell $(CXX) -showme:command)

# OPT is used when compiling object files
# B_OPT when compiling the final executable binary
ifeq ($(COMPILER),icpc)
# Intel optimizations are different to gcc ones
CXXFLAGS += -std=c++11 -xhost
OPT = -O3 -ipo
else
# Using cygwin -std=gnu++11 should be used rather than -std=c++11
CXXFLAGS += -Wall -Wextra -std=gnu++11 -march=native\
	-fno-signed-zeros\
	-fno-math-errno\
	-fno-rounding-math\
	-fno-signaling-nans\
	-fno-trapping-math\
	-ffinite-math-only\
	-Wno-misleading-indentation
CXXFLAGS += -flto -fuse-linker-plugin
OPT = -O3
endif
CXXFLAGS += -DARMA_DONT_USE_WRAPPER -DARMA_NO_DEBUG -DNDEBUG
B_OPT += $(OPT)

# Specify base directory
BASE_DIR = .

# Specify source directory
SRC_DIR = $(BASE_DIR)/src
INC_DIR = $(BASE_DIR)/include

# define the output directory for .o
OBJ_DIR = $(BASE_DIR)/build

# define any directories containing header files other than /usr/include
# e.g. I/home/include
INCLUDES = -I$(BASE_DIR)/include -I$(BASE_DIR)/lib/include -I$(HOME)/include

# define library paths in addition to /usr/lib
# e.g. -L/home/lib
LFLAGS = -L$(BASE_DIR) -L$(BASE_DIR)/lib -L$(HOME)/lib

# define any libraries to link into executable:
LIBS = -Wl,-Bstatic -lptope -Wl,-Bdynamic -lopenblas -llapack -lboost_system
#LIBS = -lptope -lopenblas -llapack -lboost_system

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

install:	$(MAIN)
	cp $(MAIN) $(HOME)/bin/

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
