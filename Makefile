#*************************************************************************#
#*									 *#
#*		Makefile for the C5.0 system				 *#
#*		----------------------------				 *#
#*									 *#
#*************************************************************************#


CC	= gcc
CXX	= g++ -ffloat-store
CXXSTD = -std=c++17 -Iinclude
CXXFLAGS = $(CXXSTD) -g -Wall -DVerbOpt -O0
LFLAGS = $(S)

.PHONY: all test


#	Definitions of file sets
#	New file ordering suggested by gprof

SRC_DIR = src

sources =\
	$(SRC_DIR)/global.cpp\
	$(SRC_DIR)/c50_api.cpp\
	$(SRC_DIR)/c50_input.cpp\
	$(SRC_DIR)/c50_output.cpp\
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
	$(SRC_DIR)/c50_api_internal.h\
	$(SRC_DIR)/c50_input.h\
	$(SRC_DIR)/c50_output.h\
	$(SRC_DIR)/c50_rng.h\
	$(SRC_DIR)/defns.i\
	$(SRC_DIR)/extern.i\
	$(SRC_DIR)/text.i

all: c5.0 report


test: c5.0 report prediction-probe
	./tests/test_cli.sh
	./tests/test_report.sh
	./tests/test_predictions.sh


report: $(SRC_DIR)/report.c Makefile
	$(CC) $(LFLAGS) -o $@ $(SRC_DIR)/report.c -lm


prediction-probe:\
	$(prediction_sources) $(headers) Makefile
	cat $(SRC_DIR)/defns.i $(prediction_sources)\
		| egrep -v 'defns.i|extern.i' >$(SRC_DIR)/predictiongt.cpp
	$(CXX) $(CXXSTD) $(LFLAGS) -O3 -o $@ $(SRC_DIR)/predictiongt.cpp -lm
	rm $(SRC_DIR)/predictiongt.cpp


# debug version (including verbosity option)

c5.0dbg:\
	$(objects) $(headers) Makefile
	$(CXX) $(CXXSTD) -g -o c5.0dbg $(objects) -lm


# production version

c5.0:\
	$(sources) $(headers) Makefile
	cat $(SRC_DIR)/defns.i $(sources)\
		| egrep -v 'defns.i|extern.i' >$(SRC_DIR)/c50gt.cpp
	$(CXX) $(CXXSTD) $(LFLAGS) -O3 -o c5.0 $(SRC_DIR)/c50gt.cpp -lm
	strip c5.0
	rm $(SRC_DIR)/c50gt.cpp


$(objects):	Makefile $(headers)


$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
