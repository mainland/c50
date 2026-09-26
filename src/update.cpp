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
/*		Routines that provide information on progress		 */
/*              ---------------------------------------------		 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"


/*************************************************************************/
/*									 */
/*	There are several stages (see messages in Progress() below)	 */
/*	Record stage and open update file if necessary			 */
/*									 */
/*************************************************************************/


void NotifyStage(c50_context *Context, int Stage)
/*   -----------  */
{
    Context->progress.stage = Stage;
    if ( Stage == 1 )
    {
	if ( ! (Context->progress.update_file = GetFile(".tmp", "w")) )
	{
	    Error(NOFILE, "", E_ForWrite);
	}
    }
}



/*************************************************************************/
/*									 */
/*	Print progress message.  This routine is called in two ways:	 */
/*	  *  negative Delta = measure of total effort required for stage */
/*	  *  positive Delta = increment since last call			 */
/*									 */
/*************************************************************************/


void Progress(c50_context *Context, float Delta)
/*   --------  */
{
    int p;
    int tell = (Context->progress.stage >= FORMTREE &&
		Context->progress.stage <= SIFTRULES);
    const char *message;
    const char done[] = ">>>>>>>>>>>>>>>>>>>>";
    const char todo[] = "....................";

    switch ( Context->progress.stage )
    {
	case READDATA:     message = "Reading training data      "; break;
	case WINNOWATTS:   message = "Winnowing attributes       "; break;
	case FORMTREE:     message = "Constructing decision tree "; break;
	case SIMPLIFYTREE: message = "Simplifying decision tree  "; break;
	case FORMRULES:    message = "Forming rules              "; break;
	case SIFTRULES:    message = "Selecting final rules      "; break;
	case EVALTRAIN:    message = "Evaluating on training data"; break;
	case READTEST:     message = "Reading test data          "; break;
	case EVALTEST:     message = "Evaluating on test data    "; break;
	case CLEANUP:      message = "Cleaning up                "; break;
	case ALLOCTABLES:  message = "Allocating tables          "; break;
	case RESULTS:      message = "Preparing results          "; break;
	default:           message = ""; break;
    }

    if ( Context->progress.last_stage == Context->progress.stage && ! tell )
    {
	return;
    }

    Context->progress.last_stage = Context->progress.stage;

    if ( Delta <= -1 )
    {
	Context->progress.total = -Delta;
	Context->progress.current = 0;
	Context->progress.twentieth = -1;
    }
    else
    {
	Context->progress.current =
	    Min(Context->progress.total, Context->progress.current + Delta);
    }

    if ( (p = rint((20.0 * Context->progress.current) /
		   Context->progress.total)) != Context->progress.twentieth )
    {
	Context->progress.twentieth = p;
assert(p >= 0 && p <= 20);
	fprintf(Context->progress.update_file, "%s", message);
	if ( tell )
	{
	    fprintf(Context->progress.update_file, "  %s%s  (%d %s)",
			&done[20 - Context->progress.twentieth],
			&todo[Context->progress.twentieth],
			(int) (Context->progress.current+0.5),
			( Context->progress.stage == SIFTRULES ?
			    "refinements" : "cases covered" ));
	}
	fprintf(Context->progress.update_file, "\n");
	fflush(Context->progress.update_file);
    }
}
