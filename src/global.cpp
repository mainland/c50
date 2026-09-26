/*************************************************************************/
/*									 */
/*  Copyright 2010 Rulequest Research Pty Ltd.				 */
/*									 */
/*  This file is part of C5.0 GPL Edition, a single-threaded version	 */
/*  of C5.0 release 2.07.						 */
/*									 */
/*  C5.0 GPL Edition is free software: you can redistribute it and/or	 */
/*  modify it under the terms of the GNU General Public License as	 */
/*  published by the Free Software Foundation, either version 3 of the	 */
/*  License, or (at your option) any later version.			 */
/*									 */
/*  C5.0 GPL Edition is distributed in the hope that it will be useful,	 */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of	 */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU	 */
/*  General Public License for more details.				 */
/*									 */
/*  You should have received a copy of the GNU General Public License	 */
/*  (gpl.txt) along with C5.0 GPL Edition.  If not, see 		 */
/*									 */
/*      <http://www.gnu.org/licenses/>.					 */
/*									 */
/*************************************************************************/



/*************************************************************************/
/*									 */
/*		General data for C5.0					 */
/*		---------------------					 */
/*									 */
/*************************************************************************/

#include "defns.i"

/*************************************************************************/
/*									 */
/*		Parameters etc						 */
/*									 */
/*************************************************************************/

/*************************************************************************/
/*									 */
/*		Attributes and data					 */
/*									 */
/*************************************************************************/

int		LineNo=0,	/* input line number */
		ErrMsgs=0,	/* errors found */
		AttExIn=0,	/* attribute exclusions/inclusions */
		TSBase=0;	/* base day for time stamps */

String		FileStem="undefined";

/*************************************************************************/
/*									 */
/*		Trees							 */
/*									 */
/*************************************************************************/

float		SampleFrac=1;	/* fraction used when sampling */

double		**DFreq=0;	/* DFreq[a][c*x] = Freq[][] for attribute a */

float		*Gain=0,	/* Gain[a] = info gain by split on att a */
		*Info=0,	/* Info[a] = max info from split on att a */
		*EstMaxGR=0;	/* EstMaxGR[a] = est max GR from folit on a */

ContValue	*Bar=0;		/* Bar[a]  = best threshold for contin att a */

double		GlobalBaseInfo,	/* base information before split */
		**Bell=0;	/* table of Bell numbers for subsets */

Byte		*Tested=0;	/* Tested[a] = att a already tested */

Set		**Subset=0;	/* Subset[a][s] = subset s for att a */
int		*Subsets=0;	/* Subsets[a] = no. subsets for att a */

EnvRec		GEnv;		/* environment block */

/*************************************************************************/
/*									 */
/*		Rules							 */
/*									 */
/*************************************************************************/

Byte		**Fires=Nil,	/* Fires[r][*] = cases covered by rule r */
		*CBuffer=Nil;	/* buffer for compressing lists */

int		*CovBy=Nil,	/* entry numbers for Fires inverse */
		*List=Nil;	/* temporary list of cases or rules */

float		AttTestBits,	/* average bits to encode tested attribute */
		*BranchBits=0;	/* ditto attribute value */
int		*AttValues=0,	/* number of attribute values in the data */
		*PossibleCuts=0;/* number of thresholds for an attribute */

double		*LogCaseNo=0,	/* LogCaseNo[i] = log2(i) */
		*LogFact=0;	/* LogFact[i] = log2(i!) */

/*************************************************************************/
/*									 */
/*		Misc							 */
/*									 */
/*************************************************************************/

int		KRInit=0,	/* KRandom initializer for Context->options.sample_fraction */
		Now=0;		/* current stage */

FILE		*TRf=0;		/* file pointer for tree and rule i/o */
char		Fn[500];	/* file name */

FILE  		*Of=0;		/* output file */
