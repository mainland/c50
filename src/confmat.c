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
/*								  	 */
/*	Routine for printing confusion matrices and attribute usage	 */
/*	-----------------------------------------------------------	 */
/*								  	 */
/*************************************************************************/

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"


void PrintConfusionMatrix(c50_context *Context, CaseNo *ConfusionMat)
/*   --------------------  */
{
    int		Row, Col, Entry, EntryWidth=10000;

    /*  For more than 20 classes, use summary instead  */

    if ( Context->schema.max_class > 20 )
    {
	PrintErrorBreakdown(Context, ConfusionMat);
	return;
    }

    /*  Find maximum entry width in chars  */

    ForEach(Row, 1, Context->schema.max_class)
    {
	ForEach(Col, 1, Context->schema.max_class)
	{
	    EntryWidth = Max(EntryWidth, ConfusionMat[Row*(Context->schema.max_class+1) + Col]);
	}
    }

    EntryWidth = floor(Log(EntryWidth + 100.0) / Log(10.0)) + 2;

    /*  Print the heading, then each row  */

    fprintf(Context->io.output, "\n\n\t");
    ForEach(Col, 1, Context->schema.max_class)
    {
	fprintf(Context->io.output, "%*s(%c)", EntryWidth-3, " ", 'a' + Col-1);
    }

    fprintf(Context->io.output, "    <-" T_classified_as "\n\t");
    ForEach(Col, 1, Context->schema.max_class)
    {
	fprintf(Context->io.output, "%*.*s", EntryWidth, EntryWidth-2, "----------");
    }
    fprintf(Context->io.output, "\n");

    ForEach(Row, 1, Context->schema.max_class)
    {
	fprintf(Context->io.output, "\t");
	ForEach(Col, 1, Context->schema.max_class)
	{
	    if ( (Entry = ConfusionMat[Row*(Context->schema.max_class+1) + Col]) )
	    {
		fprintf(Context->io.output, " %*d", EntryWidth-1, Entry);
	    }
	    else
	    {
		fprintf(Context->io.output, "%*s", EntryWidth, " ");
	    }
	}
	fprintf(Context->io.output, "    (%c): " T_class " %s\n", 'a' + Row-1, Context->schema.class_names[Row]);
    }
}



void PrintErrorBreakdown(c50_context *Context, CaseNo *ConfusionMat)
/*   -------------------  */
{
    CaseNo	*TruePos, *FalsePos, *FalseNeg, Entry;
    int		Row, Col, EntryWidth=100000, ClassWidth=5;
    size_t	NameWidth;

    TruePos  = AllocZero(Context->schema.max_class+1, CaseNo);
    FalsePos = AllocZero(Context->schema.max_class+1, CaseNo);
    FalseNeg = AllocZero(Context->schema.max_class+1, CaseNo);

    ForEach(Row, 1, Context->schema.max_class)
    {
	ForEach(Col, 1, Context->schema.max_class)
	{
	    Entry = ConfusionMat[Row*(Context->schema.max_class+1) + Col];

	    if ( Col == Row )
	    {
		TruePos[Row] += Entry;
	    }
	    else
	    {
		FalseNeg[Row] += Entry;
		FalsePos[Col] += Entry;
	    }
	}

	EntryWidth = Max(EntryWidth, TruePos[Row] + FalseNeg[Row]);
	NameWidth = strlen(Context->schema.class_names[Row]);
	if ( NameWidth > INT_MAX ) Error(Context, LONGNAME, "", "");
	ClassWidth = Max(ClassWidth, (int) NameWidth);
    }

    EntryWidth = floor(Log(EntryWidth + 100.0) / Log(10.0)) + 2;

    /*  Print heading (tricky spacing) */

    fprintf(Context->io.output, "\n\n\t  %-*s %*s %*s %*s\n\t  %*s %*s %*s %*s\n",
		ClassWidth, "Class",
		EntryWidth, "Cases",
		EntryWidth, "False",
		EntryWidth, "False",
		ClassWidth, "",
		EntryWidth, "",
		EntryWidth, "Pos",
		EntryWidth, "Neg");
    fprintf(Context->io.output, "\t  %-*s %*s %*s %*s\n",
		ClassWidth, "-----",
		EntryWidth, "-----",
		EntryWidth, "-----",
		EntryWidth, "-----");

    ForEach(Row, 1, Context->schema.max_class)
    {
	fprintf(Context->io.output, "\t  %-*s %*d %*d %*d\n",
		ClassWidth, Context->schema.class_names[Row],
		EntryWidth, TruePos[Row] + FalseNeg[Row],
		EntryWidth, FalsePos[Row],
		EntryWidth, FalseNeg[Row]);
    }

    Free(TruePos);
    Free(FalsePos);
    Free(FalseNeg);
}



void PrintUsageInfo(c50_context *Context, CaseNo *Usage)
/*   --------------  */
{
    Attribute	Att, Best;
    float	Tests;
    Boolean	First=true;

    Tests = Max(1, Context->cases.max_case+1);

    while ( true )
    {
	Best = 0;

	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    if ( Usage[Att] > Usage[Best] ) Best = Att;
	}

	if ( ! Best || Usage[Best] < 0.01 * Tests ) break;

	if ( First )
	{
	    fprintf(Context->io.output, T_Usage);
	    First = false;
	}

	fprintf(Context->io.output, "\t%7d%%  %s\n",
	    (int) ((100 * Usage[Best]) / Tests + 0.5), Context->schema.attribute_names[Best]);

	Usage[Best] = 0;
    }
}
