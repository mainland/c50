/*************************************************************************/
/*									 */
/*  Copyright 2010 Rulequest Research Pty Ltd.				 */
/*  Modifications Copyright 2026 Geoffrey Mainland.			 */
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
/*	Print header for all C5.0 programs				 */
/*	----------------------------------				 */
/*									 */
/*************************************************************************/

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"
#include <stdint.h>


#define  NAME T_C50


void PrintHeader(String Title)
/*   -----------  */
{
    char	TitleLine[80];
    time_t	clock;
    int		Underline;

    clock = time(0);
    sprintf(TitleLine, "%s%s [%s]", NAME, Title, TX_Release(RELEASE));
    fprintf(Of, "\n%s  \t%s", TitleLine, ctime(&clock));

    Underline = CharWidth(TitleLine);
    while ( Underline-- ) putc('-', Of);
    putc('\n', Of);
}



/*************************************************************************/
/*									 */
/*	This is a specialised form of the getopt utility.		 */
/*									 */
/*************************************************************************/


String	OptArg, Option;


char ProcessOption(int Argc, char *Argv[], char *Options)
/*   -------------  */
{
    int		i;
    static int	OptNo=1;

    if ( OptNo >= Argc ) return '\00';

    if ( *(Option = Argv[OptNo++]) != '-' ) return '?';

    for ( i = 0 ; Options[i] ; i++ )
    {
	if ( Options[i] == Option[1] )
	{
	    OptArg = (char *) ( Options[i+1] != '+' ? Nil :
				Option[2] ? Option+2 :
				OptNo < Argc ? Argv[OptNo++] : "0" );
	    return Option[1];
	}
    }

    return '?';
}



/*************************************************************************/
/*									 */
/*	Protected memory allocation routines				 */
/*									 */
/*************************************************************************/



void *Pmalloc(size_t Bytes)
/*    -------  */
{
    void *p=Nil;

    if ( ! Bytes || (p = (void *) malloc(Bytes)) )
    {
	return p;
    }

    Error(NOMEM, "", "");

    return Nil;
}



void *Prealloc(void *Present, size_t Bytes)
/*    --------  */
{
    void *p=Nil;

    if ( ! Bytes ) return Nil;

    if ( ! Present ) return Pmalloc(Bytes);

    if ( (p = (void *) realloc(Present, Bytes)) )
    {
	return p;
    }

    Error(NOMEM, "", "");

    return Nil;
}



void *Pcalloc(size_t Number, unsigned int Size)
/*    -------  */
{
    void *p=Nil;

    if ( ! Number || (p = (void *) calloc(Number, Size)) )
    {
	return p;
    }

    Error(NOMEM, "", "");

    return Nil;
}



void FreeVector(void **V, int First, int Last)
/*   ----------  */
{
    if ( V )
    {
	while ( First <= Last )
	{
	    FreeUnlessNil(V[First]);
	    First++;
	}

	Free(V);
    }
}



/*************************************************************************/
/*									 */
/*	Special memory allocation routines for case memory		 */
/*									 */
/*************************************************************************/

typedef struct _datablockrec	*DataBlock;

typedef	struct _datablockrec
	{
	  DataRec	Head;		/* first address */
	  int		Allocated;	/* number of cases in this block */
	  DataBlock	Prev;		/* previous data block */
	}
	DataBlockRec;

DataRec NewCase(c50_context *Context)
/*      -------  */
{
    DataBlock	Prev;

    if ( ! Context->cases.memory_blocks ||
	 Context->cases.memory_blocks->Allocated == Context->cases.block_size )
    {
	Context->cases.block_size =
	    Min(8192, 262144 / (Context->schema.max_attribute+2) + 1);

	Prev = Context->cases.memory_blocks;
	Context->cases.memory_blocks = AllocZero(1, DataBlockRec);
	Context->cases.memory_blocks->Head =
	    Alloc(Context->cases.block_size *
		  (Context->schema.max_attribute+2), AttValue);
	Context->cases.memory_blocks->Prev = Prev;
    }

    return Context->cases.memory_blocks->Head +
	(Context->cases.memory_blocks->Allocated++) *
	(Context->schema.max_attribute+2) + 1;
}



void FreeCases(c50_context *Context)
/*   ---------  */
{
    DataBlock	Prev;

    while ( Context->cases.memory_blocks )
    {
	Prev = Context->cases.memory_blocks->Prev;
	Free(Context->cases.memory_blocks->Head);
	Free(Context->cases.memory_blocks);
	Context->cases.memory_blocks = Prev;
    }
}



void FreeLastCase(c50_context *Context, DataRec Case)
/*   ------------  */
{
    (void) Case;
    Context->cases.memory_blocks->Allocated--;
}



/*************************************************************************/
/*									 */
/*	Generate uniform random numbers					 */
/*									 */
/*************************************************************************/


#define	Modify(F,S)	if ( (F -= S) < 0 ) F += 1.0

double KRandom(KRState *State)
/*     -------  */
{
    double		V1, V2;
    int			i, j;

    /*  Initialisation  */

    if ( State->first == State->second )
    {
	State->first = 0;
	State->second = 31;

	V1 = 1.0;
	V2 = 0.314159285;

	ForEach(i, 1, 55)
	{
	    State->values[ j = (i * 21) % 55 ] = V1;
	    V1 = V2 - V1;
	    if ( V1 < 0 ) V1 += 1.0;
	    V2 = State->values[j];
	}

	ForEach(j, 0, 5)
	{
	    ForEach(i, 0, 54)
	    {
		Modify(State->values[i], State->values[(i+30) % 55]);
	    }
	}
    }

    State->first = (State->first + 1) % 55;
    State->second = (State->second + 1) % 55;
    Modify(State->values[State->first], State->values[State->second]);

    return State->values[State->first];
}



void ResetKR(KRState *State, int Seed)
/*   -------  */
{
    State->first = State->second = 0;

    Seed += 1000;
    while ( Seed-- )
    {
	KRandom(State);
    }
}



/*************************************************************************/
/*									 */
/*	Error messages							 */
/*									 */
/*************************************************************************/


void C50Exit(int Status)
/*   -------  */
{
    if ( c50_abort_active_operation(Status) ) return;
    exit(Status);
}



void ErrorContext(c50_context *Context, int ErrNo, String S1, String S2)
/*   -----  */
{
    Boolean	Quit=false, WarningOnly=false;
    char	Buffer[10000], *Msg=Buffer;


    if ( Of ) fprintf(Of, "\n");

    if ( ErrNo == NOFILE || ErrNo == NOMEM || ErrNo == MODELFILE )
    {
	sprintf(Msg, "*** ");
    }
    else
    {
	sprintf(Msg, TX_Line(LineNo, Fn));
    }
    Msg += strlen(Buffer);

    switch ( ErrNo )
    {
	case NOFILE:
	    sprintf(Msg, E_NOFILE(Fn, S2));
	    Quit = true;
	    break;

	case BADCLASSTHRESH:
	    sprintf(Msg, E_BADCLASSTHRESH, S1);
	    break;

	case LEQCLASSTHRESH:
	    sprintf(Msg, E_LEQCLASSTHRESH, S1);
	    break;

	case BADATTNAME:
	    sprintf(Msg, E_BADATTNAME, S1);
	    break;

	case EOFINATT:
	    sprintf(Msg, E_EOFINATT, S1);
	    break;

	case SINGLEATTVAL:
	    sprintf(Msg, E_SINGLEATTVAL(S1, S2));
	    break;

	case DUPATTNAME:
	    sprintf(Msg, E_DUPATTNAME, S1);
	    break;

	case CWTATTERR:
	    sprintf(Msg, E_CWTATTERR);
	    break;

	case BADATTVAL:
	    sprintf(Msg, E_BADATTVAL(S2, S1));
	    break;

	case BADNUMBER:
	    sprintf(Msg, E_BADNUMBER(S1));
	    break;

	case BADCLASS:
	    sprintf(Msg, E_BADCLASS, S2);
	    break;

	case BADCOSTCLASS:
	    sprintf(Msg, E_BADCOSTCLASS, S1);
	    Quit = true;
	    break;

	case BADCOST:
	    sprintf(Msg, E_BADCOST, S1);
	    Quit = true;
	    break;

	case NOMEM:
	    sprintf(Msg, E_NOMEM);
	    Quit = true;
	    break;

	case TOOMANYVALS:
	    sprintf(Msg, E_TOOMANYVALS(S1, (int) (intptr_t) S2));
	    break;

	case BADDISCRETE:
	    sprintf(Msg, E_BADDISCRETE, S1);
	    break;

	case NOTARGET:
	    sprintf(Msg, E_NOTARGET, S1);
	    Quit = true;
	    break;

	case BADCTARGET:
	    sprintf(Msg, E_BADCTARGET, S1);
	    Quit = true;
	    break;

	case BADDTARGET:
	    sprintf(Msg, E_BADDTARGET, S1);
	    Quit = true;
	    break;

	case LONGNAME:
	    sprintf(Msg, E_LONGNAME);
	    Quit = true;
	    break;

	case HITEOF:
	    sprintf(Msg, E_HITEOF);
	    break;

	case MISSNAME:
	    sprintf(Msg, E_MISSNAME, S2);
	    break;

	case BADTSTMP:
	    sprintf(Msg, E_BADTSTMP(S2, S1));
	    break;

	case BADDATE:
	    sprintf(Msg, E_BADDATE(S2, S1));
	    break;

	case BADTIME:
	    sprintf(Msg, E_BADTIME(S2, S1));
	    break;

	case UNKNOWNATT:
	    sprintf(Msg, E_UNKNOWNATT, S1);
	    break;

	case BADDEF1:
	    sprintf(Msg, E_BADDEF1(Context->schema.attribute_names[Context->schema.max_attribute], S1, S2));
	    break;

	case BADDEF2:
	    sprintf(Msg, E_BADDEF2(Context->schema.attribute_names[Context->schema.max_attribute], S1, S2));
	    break;

	case SAMEATT:
	    sprintf(Msg, E_SAMEATT(Context->schema.attribute_names[Context->schema.max_attribute], S1));
	    WarningOnly = true;
	    break;

	case BADDEF3:
	    sprintf(Msg, E_BADDEF3, Context->schema.attribute_names[Context->schema.max_attribute]);
	    break;

	case BADDEF4:
	    sprintf(Msg, E_BADDEF4, Context->schema.attribute_names[Context->schema.max_attribute]);
	    WarningOnly = true;
	    break;

	case MODELFILE:
	    sprintf(Msg, EX_MODELFILE(Fn));
	    sprintf(Msg, "    (%s `%s')\n", S1, S2);
	    Quit = true;
	    break;
    }

    if ( Of ) fputs(Buffer, Of);
	
    if ( ! WarningOnly )
    {
	ErrMsgs++;
	c50_record_error(ErrNo == NOMEM ? C50_STATUS_OUT_OF_MEMORY :
			 ErrNo == NOFILE ? C50_STATUS_IO_ERROR :
			 C50_STATUS_PARSE_ERROR,
			 Buffer);
    }

    if ( ErrMsgs == 10 )
    {
	if ( Of ) fprintf(Of,  T_ErrorLimit);
	Context->cases.max_case--;
	Quit = true;
    }

    if ( Quit )
    {
	Goodbye(1);
    }
}



void Error(int ErrNo, String S1, String S2)
/*   -----  */
{
    ErrorContext(Nil, ErrNo, S1, S2);
}



/*************************************************************************/
/*                                                                       */
/*      Generate the label for a case                                    */
/*                                                                       */
/*************************************************************************/

char	LabelBuffer[1000];


String CaseLabel(c50_context *Context, CaseNo N)
/*     ---------  */
{
    String      p;

    if ( Context->schema.label_attribute &&
	 (p = Context->ignored_values +
	      SVal(Context->cases.records[N], Context->schema.label_attribute)) )
	;
    else
    {
	sprintf(LabelBuffer, "#%d", N+1);
	p = LabelBuffer;
    }

    return p;
}



/*************************************************************************/
/*									 */
/*	Open file with given extension for read/write			 */
/*									 */
/*************************************************************************/


FILE *GetFile(String Extension, String RW)
/*    --------  */
{
    strcpy(Fn, FileStem);
    strcat(Fn, Extension);
    return fopen(Fn, RW);
}



/*************************************************************************/
/*									 */
/*	Determine total elapsed time so far.				 */
/*									 */
/*************************************************************************/


#include <sys/time.h>

double  ExecTime()
/*      --------  */
{
    struct timeval	TV;
    struct timezone	TZ={0,0};

    gettimeofday(&TV, &TZ);
    return TV.tv_sec + TV.tv_usec / 1000000.0;
}



/*************************************************************************/
/*									 */
/*	Determine precision of floating value				 */
/*									 */
/*************************************************************************/


int Denominator(ContValue Val)
/*  -----------  */
{
    double	RoundErr, Accuracy;
    int		Mult;

    Accuracy = fabs(Val) * 1E-6;	/* approximate */
    Val = modf(Val, &RoundErr);

    for ( Mult = 100000 ; Mult >= 1 ; Mult /= 10 )
    {
	RoundErr = fabs(rint(Val * Mult) / Mult - Val);
	if ( RoundErr > 2 * Accuracy )
	{
	    return Mult * 10;
	}
    }

    return 1;
}



/*************************************************************************/
/*									 */
/*	Routines to process date (Algorithm due to Gauss?)		 */
/*									 */
/*************************************************************************/


int GetInt(String S, int N)
/*  ------  */
{
    int	Result=0;

    while ( N-- )
    {
	if ( ! isdigit(*S) ) return 0;

	Result = Result * 10 + (*S++ - '0');
    }

    return Result;
}


int DateToDay(String DS)	/*  Day 1 is 0000/03/01  */
/*  ---------  */
{
    int Year, Month, Day;

    if ( strlen(DS) != 10 ) return 0;

    Year  = GetInt(DS, 4);
    Month = GetInt(DS+5, 2);
    Day   = GetInt(DS+8, 2);

    if ( ! ( ( DS[4] == '/' && DS[7] == '/' ) ||
	     ( DS[4] == '-' && DS[7] == '-' ) ) ||
	 Year < 0 || Month < 1 || Day < 1 ||
	 Month > 12 ||
	 Day > 31 ||
	 ( Day > 30 &&
	   ( Month == 4 || Month == 6 || Month == 9 || Month == 11 ) ) ||
	 ( Month == 2 &&
	    ( Day > 29 ||
	      ( Day > 28 && ( Year % 4 != 0 ||
			      ( Year % 100 == 0 && Year % 400 != 0 ) ) ) ) ) )
    {
	return 0;
    }

    if ( (Month -= 2) <= 0 )
    {
	Month += 12;
	Year -= 1;
    }

    return Year * 365 + Year / 4 - Year / 100 + Year / 400
	   + 367 * Month / 12
	   + Day - 30;
}



void DayToDate(int Day, String Date)
/*   ---------  */
{
    int Year, Month, OrigDay=Day;

    if ( Day <= 0 )
    {
	strcpy(Date, "?");
	return;
    }

    Year = (Day - 1) / 365.2425L;  /*  Year = completed years  */
    Day -= Year * 365 + Year / 4 - Year / 100 + Year / 400;

    if ( Day < 1 )
    {
	Year--;
	Day = OrigDay - (Year * 365 + Year / 4 - Year / 100 + Year / 400);
    }
    else
    if ( Day > 366 ||
	 ( Day == 366 &&
	   ( (Year+1) % 4 != 0 ||
	     ( (Year+1) % 100 == 0 && (Year+1) % 400 != 0 ) ) ) )
    {
	Year++;
	Day = OrigDay - (Year * 365 + Year / 4 - Year / 100 + Year / 400);
    }

    Month = (Day + 30) * 12 / 367;
    Day -= 367 * Month / 12 - 30;
    if ( Day < 1 )
    {
	Month = 11;
	Day = 31;
    }

    Month += 2;
    if ( Month > 12 )
    {
	Month -= 12;
	Year++;
    }

    sprintf(Date, "%d/%d%d/%d%d", Year, Month/10, Month % 10, Day/10, Day % 10);
}



/*************************************************************************/
/*									 */
/*	Routines to process clock time and timestamps			 */
/*									 */
/*************************************************************************/


int TimeToSecs(String TS)
/*  ----------  */
{
    int Hour, Mins, Secs;

    if ( strlen(TS) != 8 ) return -1;

    Hour = GetInt(TS, 2);
    Mins = GetInt(TS+3, 2);
    Secs = GetInt(TS+6, 2);

    if ( TS[2] != ':' || TS[5] != ':' ||
	 Hour >= 24 || Mins >= 60 || Secs >= 60 )
    {
	return -1;
    }

    return Hour * 3600 + Mins * 60 + Secs;
}



void SecsToTime(int Secs, String Time)
/*   ----------  */
{
    int Hour, Mins;

    Hour = Secs / 3600;
    Mins = (Secs % 3600) / 60;
    Secs = Secs % 60;

    sprintf(Time, "%d%d:%d%d:%d%d",
		  Hour / 10, Hour % 10,
		  Mins / 10, Mins % 10,
		  Secs / 10, Secs % 10);
}



void SetTSBase(int y)
/*   ---------  */
{
    y -= 15;
    TSBase = y * 365 + y / 4 - y / 100 + y / 400 + (367 * 4) / 12 + 1 - 30;
}



int TStampToMins(String TS)
/*  ------------  */
{
    int		Day, Sec, i;

    /*  Check for reasonable length and space between date and time  */

    if ( strlen(TS) < 19 || ! Space(TS[10]) ) return (1 << 30);

    /*  Read date part  */

    TS[10] = '\00';
    Day = DateToDay(TS);
    TS[10] = ' ';

    /*  Skip one or more spaces  */

    for ( i = 11 ; TS[i] && Space(TS[i]) ; i++ )
	;

    /*  Read time part  */

    Sec = TimeToSecs(TS+i);

    /*  Return a long time in the future if there is an error  */

    return ( Day < 1 || Sec < 0 ? (1 << 30) :
	     (Day - TSBase) * 1440 + (Sec + 30) / 60 );
}



/*************************************************************************/
/*									 */
/*	Convert a continuous value to a string.		DS must be	 */
/*	large enough to hold any value (e.g. a date, time, ...)		 */
/*									 */
/*************************************************************************/


void CValToStr(c50_context *Context, ContValue CV, Attribute Att, String DS)
/*   ---------  */
{
    int		Mins;

    if ( TStampVal(Att) )
    {
	DayToDate(floor(CV / 1440) + TSBase, DS);
	DS[10] = ' ';
	Mins = rint(CV) - floor(CV / 1440) * 1440;
	SecsToTime(Mins * 60, DS+11);
    }
    else
    if ( DateVal(Att) )
    {
	DayToDate(CV, DS);
    }
    else
    if ( TimeVal(Att) )
    {
	SecsToTime(CV, DS);
    }
    else
    {
	sprintf(DS, "%.*g", PREC, CV);
    }
}



/*************************************************************************/
/*									 */
/*	Check parameter value						 */
/*									 */
/*************************************************************************/


void Check(float Val, float Low, float High)
/*   -----  */
{
    if ( Val < Low || Val > High )
    {
	fprintf(Of, TX_IllegalValue(Val, Low, High));
	C50Exit(1);
    }
}





/*************************************************************************/
/*									 */
/*	Deallocate all dynamic storage					 */
/*									 */
/*************************************************************************/


void Cleanup(c50_context *Context)
/*   -------  */
{
    int		t, r;

    extern DataRec	*Blocked;
    extern Tree		*SubDef;
    extern int		SubSpace;
    extern FILE		*Uf;

    NotifyStage(CLEANUP);

    CheckClose(Uf);					Uf = Nil;
    CheckClose(TRf);					TRf = Nil;

    /*  Boost voting (construct.c)  */

    FreeUnlessNil(Context->training.boost_vote_block);				Context->training.boost_vote_block = Nil;

    /*  Stuff from attribute winnowing  */

    FreeUnlessNil(Context->cases.saved_records);				Context->cases.saved_records = Nil;
    FreeUnlessNil(Context->training.attribute_importance);
    Context->training.attribute_importance = Nil;
    FreeUnlessNil(Context->training.split_attributes);
    Context->training.split_attributes = Nil;
    FreeUnlessNil(Context->training.used_attributes);
    Context->training.used_attributes = Nil;

    if ( Context->options.rules )
    {
	FreeFormRuleData();
	FreeSiftRuleData(Context);
    }

    /*  May have interrupted a winnowing tree  */

    if ( Context->options.winnow && Context->trees.winnow )
    {
	FreeTree(Context->trees.winnow);				Context->trees.winnow = Nil;
    }

    FreeUnlessNil(Blocked);				Blocked = Nil;

    FreeData(Context);

    if ( Context->costs.matrix )
    {
	FreeVector((void **) Context->costs.matrix, 1, Context->schema.max_class);	Context->costs.matrix = Nil;
	FreeUnlessNil(Context->costs.weight_multipliers);			Context->costs.weight_multipliers = Nil;
    }

    ForEach(t, 0, Context->trees.max_tree)
    {
	FreeClassifier(Context, t);
    }

    if ( Context->options.rules )
    {
	/*  May be incomplete ruleset in Context->rules.rules[]  */

	if ( Context->rules.rules )
	{
	    ForEach(r, 1, Context->rules.count)
	    {
		FreeRule(Context->rules.rules[r]);
	    }
	    Free(Context->rules.rules);					Context->rules.rules = Nil;
	}						

	FreeUnlessNil(Context->rules.sets);				Context->rules.sets = Nil;
	FreeUnlessNil(LogCaseNo);			LogCaseNo = Nil;
	FreeUnlessNil(LogFact);				LogFact = Nil;
    }

    FreeTreeData(Context);

    FreeUnlessNil(Context->evaluation.utility_errors);				Context->evaluation.utility_errors = Nil;
    FreeUnlessNil(Context->evaluation.utility_bands);				Context->evaluation.utility_bands = Nil;
    FreeUnlessNil(Context->evaluation.utility_costs);				Context->evaluation.utility_costs = Nil;

    FreeUnlessNil(Context->cases.some_missing);				Context->cases.some_missing = Nil;
    FreeUnlessNil(Context->cases.some_not_applicable);				Context->cases.some_not_applicable = Nil;

    FreeNames(Context);

    FreeUnlessNil(SubDef);				SubDef = Nil;
							SubSpace = 0;
    Context->cases.max_case = -1;

    NotifyStage(0);
}



#ifdef UTF8
///////////////////////////////////////////////////////////////////////////
//									 //
//	Routines for Unicode/UTF-8 processing				 //
//	-------------------------------------				 //
//									 //
///////////////////////////////////////////////////////////////////////////

#include <wchar.h>



/*************************************************************************/
/*									 */
/*	Determine the total character width of a UTF-8 string		 */
/*									 */
/*************************************************************************/


int UTF8CharWidth(unsigned char *U)
/*  -------------  */
{
    int		CWidth=0, Mask, This;
    wchar_t	Unicode;

    while ( *U )
    {
	Unicode = *U;

	if ( *U < 0x7F )
	{
	    /*  ASCII character  */

	    CWidth++;
	    U++;
	}
	else
	{
	    /*  Discard header bits  */

	    Mask = 0x80;
	    while ( Unicode & Mask )
	    {
		Unicode ^= Mask;
		Mask = Mask >> 1;
	    }

	    while ( ((*(++U)) & 0xc0) == 0x80 )
	    {
		Unicode = (Unicode << 6) | (*U & 0x3f);
	    }

	    if ( (This = wcwidth(Unicode)) > 0 ) CWidth += This;
	}
    }

    return CWidth;
}



////////////////////////////////////////////////////////////////////////////////
//	Public domain code to determine the width of a Unicode character      //
////////////////////////////////////////////////////////////////////////////////


/*
 * This is an implementation of wcwidth() and wcswidth() as defined in
 * "The Single UNIX Specification, Version 2, The Open Group, 1997"
 * <http://www.UNIX-systems.org/online.html>
 *
 * Markus Kuhn -- 2000-02-08 -- public domain
 */

//#include <wchar.h>

/* These functions define the column width of an ISO 10646 character
 * as follows:
 *
 *    - The null character (U+0000) has a column width of 0.
 *
 *    - Other C0/C1 control characters and DEL will lead to a return
 *      value of -1.
 *
 *    - Non-spacing and enclosing combining characters (general
 *      category code Mn or Me in the Unicode database) have a
 *      column width of 0.
 *
 *    - Spacing characters in the East Asian Wide (W) or East Asian
 *      FullWidth (F) category as defined in Unicode Technical
 *      Report #11 have a column width of 2.
 *
 *    - All remaining characters (including all printable
 *      ISO 8859-1 and WGL4 characters, Unicode control characters,
 *      etc.) have a column width of 1.
 *
 * This implementation assumes that wchar_t characters are encoded
 * in ISO 10646.
 */

int wcwidth(wchar_t ucs)
{
  /* sorted list of non-overlapping intervals of non-spacing characters */
  static const struct interval {
    unsigned short first;
    unsigned short last;
  } combining[] = {
    { 0x0300, 0x034E }, { 0x0360, 0x0362 }, { 0x0483, 0x0486 },
    { 0x0488, 0x0489 }, { 0x0591, 0x05A1 }, { 0x05A3, 0x05B9 },
    { 0x05BB, 0x05BD }, { 0x05BF, 0x05BF }, { 0x05C1, 0x05C2 },
    { 0x05C4, 0x05C4 }, { 0x064B, 0x0655 }, { 0x0670, 0x0670 },
    { 0x06D6, 0x06E4 }, { 0x06E7, 0x06E8 }, { 0x06EA, 0x06ED },
    { 0x0711, 0x0711 }, { 0x0730, 0x074A }, { 0x07A6, 0x07B0 },
    { 0x0901, 0x0902 }, { 0x093C, 0x093C }, { 0x0941, 0x0948 },
    { 0x094D, 0x094D }, { 0x0951, 0x0954 }, { 0x0962, 0x0963 },
    { 0x0981, 0x0981 }, { 0x09BC, 0x09BC }, { 0x09C1, 0x09C4 },
    { 0x09CD, 0x09CD }, { 0x09E2, 0x09E3 }, { 0x0A02, 0x0A02 },
    { 0x0A3C, 0x0A3C }, { 0x0A41, 0x0A42 }, { 0x0A47, 0x0A48 },
    { 0x0A4B, 0x0A4D }, { 0x0A70, 0x0A71 }, { 0x0A81, 0x0A82 },
    { 0x0ABC, 0x0ABC }, { 0x0AC1, 0x0AC5 }, { 0x0AC7, 0x0AC8 },
    { 0x0ACD, 0x0ACD }, { 0x0B01, 0x0B01 }, { 0x0B3C, 0x0B3C },
    { 0x0B3F, 0x0B3F }, { 0x0B41, 0x0B43 }, { 0x0B4D, 0x0B4D },
    { 0x0B56, 0x0B56 }, { 0x0B82, 0x0B82 }, { 0x0BC0, 0x0BC0 },
    { 0x0BCD, 0x0BCD }, { 0x0C3E, 0x0C40 }, { 0x0C46, 0x0C48 },
    { 0x0C4A, 0x0C4D }, { 0x0C55, 0x0C56 }, { 0x0CBF, 0x0CBF },
    { 0x0CC6, 0x0CC6 }, { 0x0CCC, 0x0CCD }, { 0x0D41, 0x0D43 },
    { 0x0D4D, 0x0D4D }, { 0x0DCA, 0x0DCA }, { 0x0DD2, 0x0DD4 },
    { 0x0DD6, 0x0DD6 }, { 0x0E31, 0x0E31 }, { 0x0E34, 0x0E3A },
    { 0x0E47, 0x0E4E }, { 0x0EB1, 0x0EB1 }, { 0x0EB4, 0x0EB9 },
    { 0x0EBB, 0x0EBC }, { 0x0EC8, 0x0ECD }, { 0x0F18, 0x0F19 },
    { 0x0F35, 0x0F35 }, { 0x0F37, 0x0F37 }, { 0x0F39, 0x0F39 },
    { 0x0F71, 0x0F7E }, { 0x0F80, 0x0F84 }, { 0x0F86, 0x0F87 },
    { 0x0F90, 0x0F97 }, { 0x0F99, 0x0FBC }, { 0x0FC6, 0x0FC6 },
    { 0x102D, 0x1030 }, { 0x1032, 0x1032 }, { 0x1036, 0x1037 },
    { 0x1039, 0x1039 }, { 0x1058, 0x1059 }, { 0x17B7, 0x17BD },
    { 0x17C6, 0x17C6 }, { 0x17C9, 0x17D3 }, { 0x18A9, 0x18A9 },
    { 0x20D0, 0x20E3 }, { 0x302A, 0x302F }, { 0x3099, 0x309A },
    { 0xFB1E, 0xFB1E }, { 0xFE20, 0xFE23 }
  };
  int min = 0;
  int max = sizeof(combining) / sizeof(struct interval) - 1;
  int mid;

  /* test for 8-bit control characters */
  if (ucs == 0)
    return 0;
  if (ucs < 32 || (ucs >= 0x7f && ucs < 0xa0))
    return -1;

  /* first quick check for Latin-1 etc. characters */
  if (ucs < combining[0].first)
    return 1;

  /* binary search in table of non-spacing characters */
  while (max >= min) {
    mid = (min + max) / 2;
    if (combining[mid].last < ucs)
      min = mid + 1;
    else if (combining[mid].first > ucs)
      max = mid - 1;
    else if (combining[mid].first <= ucs && combining[mid].last >= ucs)
      return 0;
  }

  /* if we arrive here, ucs is not a combining or C0/C1 control character */

  /* fast test for majority of non-wide scripts */
  if (ucs < 0x1100)
    return 1;

  return 1 +
    ((ucs >= 0x1100 && ucs <= 0x115f) || /* Hangul Jamo */
     (ucs >= 0x2e80 && ucs <= 0xa4cf && (ucs & ~0x0011) != 0x300a &&
      ucs != 0x303f) ||                  /* CJK ... Yi */
     (ucs >= 0xac00 && ucs <= 0xd7a3) || /* Hangul Syllables */
     (ucs >= 0xf900 && ucs <= 0xfaff) || /* CJK Compatibility Ideographs */
     (ucs >= 0xfe30 && ucs <= 0xfe6f) || /* CJK Compatibility Forms */
     (ucs >= 0xff00 && ucs <= 0xff5f) || /* Fullwidth Forms */
     (ucs >= 0xffe0 && ucs <= 0xffe6));
}


int wcswidth(const wchar_t *pwcs, size_t n)
{
  int w, width = 0;

  for (;*pwcs && n-- > 0; pwcs++)
    if ((w = wcwidth(*pwcs)) < 0)
      return -1;
    else
      width += w;

  return width;
}
#endif
