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
/* Modifications Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Modified 2026 by Geoffrey Mainland. */



/*************************************************************************/
/*									 */
/*	Program to produce average results from an xval			 */
/*	-----------------------------------------------			 */
/*									 */
/*************************************************************************/

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <array>
#include <charconv>
#include <cmath>
#include <iostream>
#include <new>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using Metrics = std::array<float, 3>;

void PrintSummary(const std::vector<Metrics> &Val, const char *Title);
float	SE(float sum, float sumsq, int no);

int Boost=0, Composite=0, Costs=0, Rules;

#define	SIZE	0
#define	ERRP	1
#define	COST	2

static bool ParseInteger(const char *Text, int &Value)
{
    const char *End = Text + strlen(Text);
    const auto Result = std::from_chars(Text, End, Value);
    return Result.ec == std::errc() && Result.ptr == End;
}

static bool ParseResult(const std::string &Line, bool BoostLine,
                        bool CompositeLine, bool HasCost, int &Size,
                        int &Errs, float &Percentage, float &Cost)
{
    if ( BoostLine && CompositeLine ) return false;
    std::string_view Data(Line);
    if ( CompositeLine )
    {
        if ( Data.size() < 18 ) return false;
        Data.remove_prefix(18);
    }

    std::istringstream Input{std::string(Data)};
    if ( BoostLine )
    {
        std::string Keyword;
        if ( !(Input >> Keyword) || Keyword != "boost" ) return false;
    }
    else if ( !(Input >> Size) )
    {
        return false;
    }

    char Open, Percent, Close;
    if ( !(Input >> Errs >> Open >> Percentage >> Percent >> Close) ||
         Open != '(' || Percent != '%' || Close != ')' ||
         ! std::isfinite(Percentage) )
    {
        return false;
    }
    if ( HasCost && (!(Input >> Cost) || ! std::isfinite(Cost)) ) return false;

    Input >> std::ws;
    if ( Input.eof() ) return true;

    std::string Marker;
    if ( !(Input >> Marker) || Marker != "<<" ) return false;
    Input >> std::ws;
    return Input.eof();
}


int main(int argc, char *argv[])
/*  ----  */
{
    try
    {
    const char	*p;
    std::string Line;
    int		Cases, Folds, Repeats, f, r, i, N,
		Size=0, Errs=0, OK=0;
    float	FX, Tests, Cost=0;

    if ( argc != 5 ||
	 ! ParseInteger(argv[1], Cases) ||
	 ! ParseInteger(argv[2], Folds) ||
	 ! ParseInteger(argv[3], Repeats) ||
	 ! ParseInteger(argv[4], Rules) ||
	 Cases < 1 || Folds < 2 || Folds > Cases || Repeats < 1 ||
	 ( Rules != 0 && Rules != 1 ) )
    {
	fprintf(stderr,
		"Usage: report <cases> <folds> <repeats> <rules>\n");
	return 1;
    }

    /*  Assemble all data  */

    std::vector<std::vector<Metrics>> Raw(
        static_cast<size_t>(Repeats),
        std::vector<Metrics>(static_cast<size_t>(Folds)));
    std::vector<Metrics> Average;
    if ( Repeats > 1 ) Average.resize(static_cast<size_t>(Repeats));

    /*  Determine input type from the first line  */

    if ( ! std::getline(std::cin, Line) )
    {
	fprintf(stderr, "Expecting %d lines\n", Folds * Repeats);
	return 1;
    }

    /*  Count the numbers on the line  */

    N = 0;
    for ( p = Line.c_str() ; *p ; )
    {
	if ( isdigit(static_cast<unsigned char>(*p)) )
	{
	    N++;
	    while ( isdigit(static_cast<unsigned char>(*p)) || *p == '.' ) p++;
	}
	else
	{
	    p++;
	}
    }

    if ( Line.compare(0, 5, "boost") == 0 )
    {
	Boost = 1;
	Costs = ( N == 3 );
    }
    else
    if ( Line.compare(0, 9, "composite") == 0 )
    {
	Composite = 1;
	Rules = 0;
	Costs = ( N == 4 );
    }
    else
    {
	Costs = ( N == 4 );
    }

    for ( r = 0 ; r < Repeats ; r++ )
    {
	for ( f = 0 ; f < Folds ; f++ )
	{
	    if ( r + f != 0 && ! std::getline(std::cin, Line) )
	    {
		printf("\nExpecting %d lines\n", Folds * Repeats);
		return 1;
	    }

	    Tests = Cases / Folds + ( f >= Folds - Cases % Folds);
            const bool BoostLine = Line.compare(0, 5, "boost") == 0;
            if ( BoostLine ) Boost = 1;
            OK = ParseResult(Line, BoostLine, Composite, Costs, Size, Errs,
                             FX, Cost);

            if ( ! OK )
	    {
		printf("\nCannot parse line\n\t%s\n", Line.c_str());
		return 1;
	    }

	    Raw[r][f][SIZE] = Size;
	    Raw[r][f][ERRP] = (100.0 * Errs) / Tests;
	    Raw[r][f][COST] = Cost;

	    if ( ! Average.empty() )
	    {
		for ( i = 0 ; i < 3 ; i++ )
		{
		    Average[r][i] += Raw[r][f][i];
		}
	    }
	}

	if ( ! Average.empty() )
	{
	    for ( i = 0 ; i < 3 ; i++ )
	    {
		Average[r][i] /= Folds;
	    }
	}
    }

    /*  Check that amount of data is correct  */

    if ( std::getline(std::cin, Line) )
    {
	printf("\nExpecting %d lines\n", Folds * Repeats * 2);
	return 1;
    }

    if ( ! Average.empty() )
    {
	PrintSummary(Average, "XVal");
    }
    else
    {
	PrintSummary(Raw[0], "Fold");
    }
    return 0;
    }
    catch ( const std::bad_alloc & )
    {
	fprintf(stderr, "Out of memory\n");
	return 1;
    }
    catch ( const std::length_error & )
    {
	fprintf(stderr, "Out of memory\n");
	return 1;
    }
}


const char
     *StdP[]  = {	"    Decision Tree   ",
			"  ----------------  ",
			"    Size    Errors  " },

     *StdPC[] = {	"        Decision Tree      ",
			"  -----------------------  ",
			"    Size    Errors   Cost  " },

     *Extra[] = {	"        Rules     ",
			"  ----------------",
			"      No    Errors" },

     *ExtraC[]= {	"           Rules         ",
			"  -----------------------",
			"      No    Errors   Cost" };

/*  The mean and standard error of one column of fold results.  Release
    2.07 sums the values and their squares in float.  That arithmetic is kept
    for every column whose mean and standard error are finite, so the printed
    text of those summaries does not change.  The squares overflow float for
    a value above about 1.8e19, the sum overflows near the limit of float,
    and the difference of two nearly equal sums can be negative.  Release
    2.07 prints each of those as infinity or nan.  A column that float cannot
    summarize is summarized in double from its deviations about the mean,
    which neither overflows nor cancels.  */

struct FoldSummary
{
    double	mean;
    double	se;
};

static FoldSummary Summarise(const std::vector<float> &Values)
/*                 ---------  */
{
    const int	No = static_cast<int>(Values.size());
    float	Sum = 0, SumSq = 0;

    for ( float V : Values )
    {
	Sum += V;
	SumSq += V * V;
    }

    const FoldSummary Float = { Sum / No, SE(Sum, SumSq, No) };

    if ( No < 2 || (std::isfinite(Float.mean) && std::isfinite(Float.se)) )
    {
	return Float;
    }

    double Mean = 0, Deviations = 0;

    for ( float V : Values ) Mean += V;
    Mean /= No;
    for ( float V : Values ) Deviations += (V - Mean) * (V - Mean);

    return { Mean, sqrt(Deviations / (No - 1) / No) };
}


void PrintSummary(const std::vector<Metrics> &Val, const char *Title)
/*   ------------  */
{
    const int No = static_cast<int>(Val.size());
    int i, j;
    FoldSummary Stats[3];

    for ( i = 0 ; i < 3 ; i++ )
    {
	std::vector<float> Column;
	for ( j = 0 ; j < No ; j++ ) Column.push_back(Val[j][i]);
	Stats[i] = Summarise(Column);
    }

    for ( i = 0 ; i <= 2 ; i++ )
    {
	switch ( i )
	{
	    case 0:
		printf("\n\t%s  ", Title);
		break;

	    case 1:
		printf("\t----  ");
		break;

	    case 2:
		printf("\t      ");
	}

	printf("%s\n", ( Composite ?
			 ( Costs ? ExtraC[i] : Extra[i] ) :
			 Rules ?
			 ( Costs ? ExtraC[i] : Extra[i] ) :
			 ( Costs ? StdPC[i] : StdP[i] ) ));
    }
    printf("\n");

    for ( j = 0 ; j < No ; j++ )
    {
	printf("\t%3d   ", j+1);

	if ( Boost )
	{
	    printf("       *");
	}
	else
	{
	    printf("%8.1f", Val[j][SIZE]);
	}

	printf("     %4.1f%%  ", Val[j][ERRP]);

	if ( Costs )
	{
	    printf("%5.2f  ", Val[j][COST]);
	}

	printf("\n");
    }

    printf("\n\tMean  ");

    if ( Boost )
    {
	printf("        ");
    }
    else
    {
	printf("%8.1f", Stats[SIZE].mean);
    }

    printf("     %4.1f%%  ", Stats[ERRP].mean);

    if ( Costs )
    {
	printf("%5.2f  ", Stats[COST].mean);
    }

    printf("\n\tSE    ");

    if ( Boost )
    {
	printf("        ");
    }
    else
    {
	printf("%8.1f", Stats[SIZE].se);
    }

    printf("     %4.1f%%  ", Stats[ERRP].se);

    if ( Costs )
    {
	printf("%5.2f  ", Stats[COST].se);
    }

    printf("\n");
}



float SE(float sum, float sumsq, int no)
/*    --  */
{
    float mean;

    mean = sum / no;

    return sqrt( (double) (((sumsq - no * mean * mean) / (no - 1)) / no) );
}
