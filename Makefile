#*************************************************************************#
#*									 *#
#*		Makefile for the C5.0 system				 *#
#*		----------------------------				 *#
#*									 *#
#*************************************************************************#


CC	= gcc -ffloat-store
CFLAGS = -g -Wall -DVerbOpt -O0
CPPFLAGS = -Iinclude
LFLAGS = $(S)

.PHONY: all test


#	Definitions of file sets
#	New file ordering suggested by gprof

SRC_DIR = src

sources =\
	$(SRC_DIR)/global.c\
	$(SRC_DIR)/c50_api.c\
	$(SRC_DIR)/c50_input.c\
	$(SRC_DIR)/c50_output.c\
	$(SRC_DIR)/c50_model.c\
	$(SRC_DIR)/c50.c\
	$(SRC_DIR)/construct.c\
	$(SRC_DIR)/formtree.c\
	$(SRC_DIR)/info.c\
	$(SRC_DIR)/discr.c\
	$(SRC_DIR)/contin.c\
	$(SRC_DIR)/subset.c\
	$(SRC_DIR)/prune.c\
	$(SRC_DIR)/p-thresh.c\
	$(SRC_DIR)/trees.c\
	$(SRC_DIR)/siftrules.c\
	$(SRC_DIR)/ruletree.c\
	$(SRC_DIR)/rules.c\
	$(SRC_DIR)/getdata.c\
	$(SRC_DIR)/implicitatt.c\
	$(SRC_DIR)/mcost.c\
	$(SRC_DIR)/confmat.c\
	$(SRC_DIR)/sort.c\
	$(SRC_DIR)/update.c\
	$(SRC_DIR)/attwinnow.c\
	$(SRC_DIR)/classify.c\
	$(SRC_DIR)/formrules.c\
	$(SRC_DIR)/getnames.c\
	$(SRC_DIR)/modelfiles.c\
	$(SRC_DIR)/utility.c\
	$(SRC_DIR)/xval.c

prediction_sources =\
	$(filter-out $(SRC_DIR)/c50.c,$(sources))\
	tests/prediction_probe.c

objects =\
	 $(SRC_DIR)/c50.o $(SRC_DIR)/c50_api.o $(SRC_DIR)/global.o\
	 $(SRC_DIR)/construct.o $(SRC_DIR)/formtree.o $(SRC_DIR)/info.o\
	 $(SRC_DIR)/discr.o $(SRC_DIR)/contin.o $(SRC_DIR)/subset.o\
	 $(SRC_DIR)/prune.o $(SRC_DIR)/p-thresh.o $(SRC_DIR)/trees.o\
	 $(SRC_DIR)/formrules.o $(SRC_DIR)/siftrules.o $(SRC_DIR)/ruletree.o\
	 $(SRC_DIR)/rules.o $(SRC_DIR)/xval.o $(SRC_DIR)/getnames.o\
	 $(SRC_DIR)/getdata.o $(SRC_DIR)/implicitatt.o $(SRC_DIR)/mcost.o\
	 $(SRC_DIR)/classify.o $(SRC_DIR)/confmat.o $(SRC_DIR)/sort.o\
	 $(SRC_DIR)/update.o $(SRC_DIR)/utility.o $(SRC_DIR)/modelfiles.o\
	 $(SRC_DIR)/attwinnow.o

headers =\
	$(SRC_DIR)/c50_api_internal.h\
	$(SRC_DIR)/c50_input.h\
	$(SRC_DIR)/c50_output.h\
	$(SRC_DIR)/defns.i\
	$(SRC_DIR)/extern.i\
	$(SRC_DIR)/text.i

all: c5.0 report


test: c5.0 report prediction-probe
	./tests/test_cli.sh
	./tests/test_report.sh
	./tests/test_predictions.sh


report: $(SRC_DIR)/report.c Makefile
	$(CC) $(CPPFLAGS) $(LFLAGS) -o $@ $(SRC_DIR)/report.c -lm


prediction-probe:\
	$(prediction_sources) $(headers) Makefile
	cat $(SRC_DIR)/defns.i $(prediction_sources)\
		| egrep -v 'defns.i|extern.i' >$(SRC_DIR)/predictiongt.c
	$(CC) $(CPPFLAGS) $(LFLAGS) -O3 -o $@ $(SRC_DIR)/predictiongt.c -lm
	rm $(SRC_DIR)/predictiongt.c


# debug version (including verbosity option)

c5.0dbg:\
	$(objects) $(headers) Makefile
	$(CC) $(CPPFLAGS) -g -o c5.0dbg $(objects) -lm


# production version

c5.0:\
	$(sources) $(headers) Makefile
	cat $(SRC_DIR)/defns.i $(sources)\
		| egrep -v 'defns.i|extern.i' >$(SRC_DIR)/c50gt.c
	$(CC) $(CPPFLAGS) $(LFLAGS) -O3 -o c5.0 $(SRC_DIR)/c50gt.c -lm
	strip c5.0
	rm $(SRC_DIR)/c50gt.c


$(objects):	Makefile $(headers)


$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<
