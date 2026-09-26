#*************************************************************************#
#*									 *#
#*		Makefile for the C5.0 system				 *#
#*		----------------------------				 *#
#*									 *#
#*************************************************************************#


CC	= gcc -ffloat-store
CFLAGS = -g -Wall -DVerbOpt -O0
LFLAGS = $(S)


#	Definitions of file sets
#	New file ordering suggested by gprof

SRC_DIR = src

sources =\
	$(SRC_DIR)/global.c\
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

objects =\
	 $(SRC_DIR)/c50.o $(SRC_DIR)/global.o\
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
	$(SRC_DIR)/defns.i\
	$(SRC_DIR)/extern.i\
	$(SRC_DIR)/text.i

all:
	$(MAKE) c5.0
	$(CC) $(LFLAGS) -o report $(SRC_DIR)/report.c -lm


# debug version (including verbosity option)

c5.0dbg:\
	$(objects) $(headers) Makefile
	$(CC) -g -o c5.0dbg $(objects) -lm


# production version

c5.0:\
	$(sources) $(headers) Makefile
	cat $(SRC_DIR)/defns.i $(sources)\
		| egrep -v 'defns.i|extern.i' >$(SRC_DIR)/c50gt.c
	$(CC) $(LFLAGS) -O3 -o c5.0 $(SRC_DIR)/c50gt.c -lm
	strip c5.0
	rm $(SRC_DIR)/c50gt.c


$(objects):	Makefile $(headers)


$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<
