#*************************************************************************#
#*									 *#
#*		Makefile for the C5.0 system				 *#
#*		----------------------------				 *#
#*									 *#
#*************************************************************************#

# This file began as the Makefile of C5.0 Release 2.07 GPL Edition,
# copyright 2010 Rulequest Research Pty Ltd, distributed under the GNU
# General Public License, version 3 or later (see gpl.txt).
# Modifications Copyright 2026 Geoffrey Mainland.
# SPDX-License-Identifier: GPL-3.0-or-later
# Modified 2026 by Geoffrey Mainland: C++17 build, test, and
# reference-comparison targets.


CC	= gcc
#	-ffloat-store affects only x87 arithmetic. 64-bit targets evaluate at
#	the declared precision, so only 32-bit builds keep it.
POINTER_SIZE := $(shell echo __SIZEOF_POINTER__ | g++ -E -P -x c++ - 2>/dev/null)
CXX	= g++ $(if $(filter 8,$(POINTER_SIZE)),,-ffloat-store)
CXXSTD = -std=c++17
CXXTHREAD = -pthread
CXXFLAGS = $(CXXSTD) $(CXXTHREAD) -g -Wall -DVerbOpt -O0
CPPFLAGS = -Iinclude
LFLAGS = $(S)

.PHONY: all test reference-comparison


#	Definitions of file sets
#	New file ordering suggested by gprof

SRC_DIR = src

sources =\
	$(SRC_DIR)/global.cpp\
	$(SRC_DIR)/c50_api.cpp\
	$(SRC_DIR)/c50_input.cpp\
	$(SRC_DIR)/c50_output.cpp\
	$(SRC_DIR)/c50_inspection.cpp\
	$(SRC_DIR)/c50_locale.cpp\
	$(SRC_DIR)/c50_model.cpp\
	$(SRC_DIR)/c50.cpp\
	$(SRC_DIR)/construct.cpp\
	$(SRC_DIR)/formtree.cpp\
	$(SRC_DIR)/info.cpp\
	$(SRC_DIR)/discr.cpp\
	$(SRC_DIR)/contin.cpp\
	$(SRC_DIR)/subset.cpp\
	$(SRC_DIR)/prune.cpp\
	$(SRC_DIR)/p-thresh.cpp\
	$(SRC_DIR)/trees.cpp\
	$(SRC_DIR)/siftrules.cpp\
	$(SRC_DIR)/ruletree.cpp\
	$(SRC_DIR)/rules.cpp\
	$(SRC_DIR)/getdata.cpp\
	$(SRC_DIR)/implicitatt.cpp\
	$(SRC_DIR)/mcost.cpp\
	$(SRC_DIR)/confmat.cpp\
	$(SRC_DIR)/sort.cpp\
	$(SRC_DIR)/update.cpp\
	$(SRC_DIR)/attwinnow.cpp\
	$(SRC_DIR)/classify.cpp\
	$(SRC_DIR)/formrules.cpp\
	$(SRC_DIR)/getnames.cpp\
	$(SRC_DIR)/modelfiles.cpp\
	$(SRC_DIR)/utility.cpp\
	$(SRC_DIR)/xval.cpp

prediction_sources =\
	$(filter-out $(SRC_DIR)/c50.cpp,$(sources))\
	tests/prediction_probe.cpp

objects = $(sources:.cpp=.o)

headers =\
	include/c50/c50.hpp\
	$(SRC_DIR)/c50_api_internal.h\
	$(SRC_DIR)/c50_index.h\
	$(SRC_DIR)/c50_input.h\
	$(SRC_DIR)/c50_locale.h\
	$(SRC_DIR)/c50_output.h\
	$(SRC_DIR)/c50_rng.h\
	$(SRC_DIR)/c50_sort.h\
	$(SRC_DIR)/defns.i\
	$(SRC_DIR)/extern.i\
	$(SRC_DIR)/text.i

all: c5.0 report


test: c5.0 report prediction-probe
	./tests/test_makefile_headers.sh
	./tests/test_cli.sh
	./tests/test_report.sh
	./tests/test_predictions.sh


# Compare c5.0 with Release 2.07, which the script builds from `main`. This
# builds and trains generated data, so `test` does not run it.
reference-comparison: c5.0
	./tests/reference/run.sh ./c5.0


report: $(SRC_DIR)/report.cpp Makefile
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LFLAGS) -o $@ $(SRC_DIR)/report.cpp -lm


prediction-probe:\
	$(prediction_sources) $(headers) Makefile
	cat $(SRC_DIR)/defns.i $(prediction_sources)\
		| egrep -v 'defns.i|extern.i' >$(SRC_DIR)/predictiongt.cpp
	$(CXX) $(CXXSTD) $(CXXTHREAD) $(CPPFLAGS) $(LFLAGS) -O3 -o $@ $(SRC_DIR)/predictiongt.cpp -lm
	rm $(SRC_DIR)/predictiongt.cpp


# debug version (including verbosity option)

c5.0dbg:\
	$(objects) $(headers) Makefile
	$(CXX) $(CXXSTD) $(CXXTHREAD) $(CPPFLAGS) -g -o c5.0dbg $(objects) -lm


# production version

c5.0:\
	$(sources) $(headers) Makefile
	cat $(SRC_DIR)/defns.i $(sources)\
		| egrep -v 'defns.i|extern.i' >$(SRC_DIR)/c50gt.cpp
	$(CXX) $(CXXSTD) $(CXXTHREAD) $(CPPFLAGS) $(LFLAGS) -O3 -o c5.0 $(SRC_DIR)/c50gt.cpp -lm
	strip c5.0
	rm $(SRC_DIR)/c50gt.cpp


$(objects):	Makefile $(headers)


$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<
