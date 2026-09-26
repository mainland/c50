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

/*************************************************************************/
/*									 */
/*		Rules							 */
/*									 */
/*************************************************************************/

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
