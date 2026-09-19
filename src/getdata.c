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
/*	Get cases from data file					 */
/*	------------------------					 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"
#include <stdint.h>


#define Inc 2048

#define XError(a,b,c)	if (! Context->suppress_error_messages) Error(a,b,c)


/*************************************************************************/
/*									 */
/*	Read raw cases from file with given extension.			 */
/*									 */
/*	On completion, cases are stored in array Case in the form	 */
/*	of vectors of attribute values, and MaxCase is set to the	 */
/*	number of data cases.						 */
/*									 */
/*************************************************************************/


void GetData(c50_context *Context, FILE *Df, Boolean Train,
	     Boolean AllowUnknownClass)
/*   -------  */
{
    c50_input Input;

    c50_input_init_file(&Input, Df);
    GetDataInput(Context, &Input, Train, AllowUnknownClass);
    fclose(Df);
}



void GetDataInput(c50_context *Context, c50_input *Input, Boolean Train,
		  Boolean AllowUnknownClass)
/*   ------------  */
{
    DataRec	DVec;
    CaseNo	CaseSpace, WantTrain, LeftTrain, WantTest, LeftTest;
    Boolean	FirstIgnore=true, SelectTrain;

    LineNo = 0;
    Context->suppress_error_messages = SAMPLE && ! Train;

    /*  Don't reset case count if appending data for xval  */

    if ( Train || ! Case )
    {
	MaxCase = CaseSpace = 0;
	Context->max_label = 0;
	Case = Alloc(1, DataRec);	/* for error reporting */
    }
    else
    {
	CaseSpace = MaxCase + 1;
	MaxCase++;
    }

    if ( SAMPLE )
    {
	if ( Train )
	{
	    Context->sample_from = CountDataInput(Input);
	    ResetKR(&Context->random, KRInit);	/* initialise KRandom() */
	}
	else
	{
	    ResetKR(&Context->random, KRInit);	/* restore  KRandom() */
	}

	WantTrain = Context->sample_from * SAMPLE + 0.5;
	LeftTrain = Context->sample_from;

	WantTest  = ( SAMPLE < 0.5 ? WantTrain :
		      Context->sample_from - WantTrain );
	LeftTest  = Context->sample_from - WantTrain;
    }

    while ( (DVec = GetDataRecInput(Context, Input, Train)) )
    {
	/*  Check whether to include if we are sampling */

	if ( SAMPLE )
	{
	    SelectTrain =
		KRandom(&Context->random) < WantTrain / (float) LeftTrain--;

	    /*  Include if
		 * Select and this is the training set
		 * ! Select and this is the test set and sub-select
	 	NB: Must use different random number generator for
		sub-selection since cannot disturb random number sequence  */

	    if ( SelectTrain )
	    {
		WantTrain--;
	    }

	    if ( SelectTrain != Train ||
		 ( ! Train && AltRandom >= WantTest / (float) LeftTest-- ) )
	    {
		FreeLastCase(DVec);
		continue;
	    }

	    if ( ! Train )
	    {
		WantTest--;
	    }
	}

	/*  Make sure there is room for another case  */

	if ( MaxCase >= CaseSpace )
	{
	    CaseSpace += Inc;
	    Realloc(Case, CaseSpace+1, DataRec);
	}

	/*  Ignore cases with unknown class  */

	if ( AllowUnknownClass || (Class(DVec) & 077777777) > 0 )
	{
	    Case[MaxCase] = DVec;
	    MaxCase++;
	}
	else
	{
	    if ( FirstIgnore && Of )
	    {
		fprintf(Of, T_IgnoreBadClass);
		FirstIgnore = false;
	    }

	    FreeLastCase(DVec);
	}
    }

    MaxCase--;

}



/*************************************************************************/
/*									 */
/*	Read a raw case from file Df.					 */
/*									 */
/*	For each attribute, read the attribute value from the file.	 */
/*	If it is a discrete valued attribute, find the associated no.	 */
/*	of this attribute value (if the value is unknown this is 0).	 */
/*									 */
/*	Returns the DataRec of the case (i.e. the array of attribute	 */
/*	values).							 */
/*									 */
/*************************************************************************/


DataRec GetDataRec(c50_context *Context, FILE *Df, Boolean Train)
/*      ----------  */
{
    c50_input Input;

    c50_input_init_file(&Input, Df);
    return GetDataRecInput(Context, &Input, Train);
}



DataRec GetDataRecInput(c50_context *Context, c50_input *Input, Boolean Train)
/*      ---------------  */
{
    Attribute	Att;
    char	Name[1000], *EndName;
    int		Dv, Chars;
    DataRec	DVec;
    ContValue	Cv;
    Boolean	FirstValue=true;


    if ( ReadNameInput(Context, Input, Name, 1000, '\00') )
    {
	Case[MaxCase] = DVec = NewCase(Context);
	ForEach(Att, 1, Context->schema.max_attribute)
	{
	    if ( Context->schema.attribute_definitions[Att] )
	    {
		DVec[Att] = EvaluateDef(Context, Context->schema.attribute_definitions[Att], DVec);

		if ( Continuous(Att) )
		{
		    CheckValue(Context, DVec, Att);
		}

		if ( SomeMiss )
		{
		    SomeMiss[Att] |= Unknown(DVec, Att);
		    SomeNA[Att]   |= NotApplic(Context, DVec, Att);
		}

		continue;
	    }

	    /*  Get the attribute value if don't already have it  */

	    if ( ! FirstValue && ! ReadNameInput(Context, Input, Name, 1000, '\00') )
	    {
		XError(HITEOF, Context->schema.attribute_names[Att], "");
		FreeLastCase(DVec);
		return Nil;
	    }
	    FirstValue = false;

	    if ( Exclude(Att) )
	    {
		if ( Att == Context->schema.label_attribute )
		{
		    /*  Record the value as a string  */

		    SVal(DVec,Att) = StoreIVal(Context, Name);
		}
	    }
	    else
	    if ( ! strcmp(Name, "?") )
	    {
		/*  Set marker to indicate missing value  */

		DVal(DVec, Att) = UNKNOWN;
		if ( SomeMiss ) SomeMiss[Att] = true;
	    }
	    else
	    if ( Att != Context->schema.class_attribute && ! strcmp(Name, "N/A") )
	    {
		/*  Set marker to indicate not applicable  */

		DVal(DVec, Att) = NA;
		if ( SomeNA ) SomeNA[Att] = true;
	    }
	    else
	    if ( Discrete(Att) )
	    {
		/*  Discrete attribute  */

		Dv = Which(Name, Context->schema.attribute_value_names[Att], 1, Context->schema.max_attribute_value[Att]);
		if ( ! Dv )
		{
		    if ( StatBit(Att, DISCRETE) )
		    {
			if ( Train || XVAL )
			{
			    /*  Add value to list  */

			    if ( Context->schema.max_attribute_value[Att] >=
				 (intptr_t) Context->schema.attribute_value_names[Att][0] )
			    {
				XError(TOOMANYVALS, Context->schema.attribute_names[Att],
					 (char *) Context->schema.attribute_value_names[Att][0] - 1);
				Dv = Context->schema.max_attribute_value[Att];
			    }
			    else
			    {
				Dv = ++Context->schema.max_attribute_value[Att];
				Context->schema.attribute_value_names[Att][Dv]   = strdup(Name);
				Context->schema.attribute_value_names[Att][Dv+1] = "<other>"; /* no free */
			    }
			    if ( Dv > Context->schema.max_discrete_value )
			    {
				Context->schema.max_discrete_value = Dv;
			    }
			}
			else
			{
			    /*  Set value to "<other>"  */

			    Dv = Context->schema.max_attribute_value[Att] + 1;
			}
		    }
		    else
		    {
			XError(BADATTVAL, Context->schema.attribute_names[Att], Name);
			Dv = UNKNOWN;
		    }
		}
		DVal(DVec, Att) = Dv;
	    }
	    else
	    {
		/*  Continuous value  */

		if ( TStampVal(Att) )
		{
		    CVal(DVec, Att) = Cv = TStampToMins(Name);
		    if ( Cv >= 1E9 )	/* long time in future */
		    {
			XError(BADTSTMP, Context->schema.attribute_names[Att], Name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		if ( DateVal(Att) )
		{
		    CVal(DVec, Att) = Cv = DateToDay(Name);
		    if ( Cv < 1 )
		    {
			XError(BADDATE, Context->schema.attribute_names[Att], Name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		if ( TimeVal(Att) )
		{
		    CVal(DVec, Att) = Cv = TimeToSecs(Name);
		    if ( Cv < 0 )
		    {
			XError(BADTIME, Context->schema.attribute_names[Att], Name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		{
		    CVal(DVec, Att) = strtod(Name, &EndName);
		    if ( EndName == Name || *EndName != '\0' )
		    {
			XError(BADATTVAL, Context->schema.attribute_names[Att], Name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}

		CheckValue(Context, DVec, Att);
	    }
	}

	if ( Context->schema.class_attribute )
	{
	    if ( Discrete(Context->schema.class_attribute) )
	    {
		Class(DVec) = XDVal(DVec, Context->schema.class_attribute);
	    }
	    else
	    if ( Unknown(DVec, Context->schema.class_attribute) || NotApplic(Context, DVec, Context->schema.class_attribute) )
	    {
		Class(DVec) = 0;
	    }
	    else
	    {
		/*  Find appropriate segment using class thresholds  */

		Cv = CVal(DVec, Context->schema.class_attribute);

		for ( Dv = 1 ; Dv < Context->schema.max_class && Cv > Context->schema.class_thresholds[Dv] ; Dv++ )
		    ;

		Class(DVec) = Dv;
	    }
	}
	else
	{
	    if ( ! ReadNameInput(Context, Input, Name, 1000, '\00') )
	    {
		XError(HITEOF, Fn, "");
		FreeLastCase(DVec);
		return Nil;
	    }

	    if ( (Class(DVec) = Dv = Which(Name, Context->schema.class_names, 1, Context->schema.max_class)) == 0 )
	    {
		if ( strcmp(Name, "?") ) XError(BADCLASS, "", Name);
	    }
	}

    if ( Context->schema.label_attribute &&
	     (Chars = strlen(Context->ignored_values +
			     SVal(DVec, Context->schema.label_attribute))) >
		 Context->max_label )
    {
	Context->max_label = Chars;
	}
	return DVec;
    }
    else
    {
	return Nil;
    }
}



/*************************************************************************/
/*                                                                       */
/*      Count cases in data file					 */
/*                                                                       */
/*************************************************************************/


CaseNo CountData(FILE *Df)
/*     ---------  */
{
    c50_input Input;

    c50_input_init_file(&Input, Df);
    return CountDataInput(&Input);
}



CaseNo CountDataInput(c50_input *Input)
/*     --------------  */
{
    char        Last=',';
    int         Count=0, Next;

    while ( true )
    {
	if ( (Next = c50_input_getc(Input)) == EOF )
	{
	    if ( Last != ',' ) Count++;
	    c50_input_rewind(Input);
	    return Count;
	}

	if ( Next == '|' )
	{
	    while ( (Next = c50_input_getc(Input)) != '\n' && Next != EOF )
		;
	    if ( Next == EOF )
	    {
		if ( Last != ',' ) Count++;
		c50_input_rewind(Input);
		return Count;
	    }
	}

	if ( Next == '\n' )
	{
	    if ( Last != ',' ) Count++;
	    Last = ',';
	}
	else
	if ( Next == '\\' )
	{
	    /*  Skip escaped character  */

	    c50_input_getc(Input);
	}
	else
	if ( Next != '\t' && Next != ' ' )
	{
	    Last = Next;
	}
    }
}



/*************************************************************************/
/*									 */
/*	Store a label or ignored value in IValStore			 */
/*									 */
/*************************************************************************/


int StoreIVal(c50_context *Context, String S)
/*  ---------  */
{
    int		StartIx, Length;

    if ( (Length=strlen(S) + 1) + Context->ignored_values_offset >
	 Context->ignored_values_size )
    {
	if ( Context->ignored_values )
	{
	    Context->ignored_values_size += 32768;
	    Realloc(Context->ignored_values, Context->ignored_values_size,
		    char);
	}
	else
	{
	    Context->ignored_values_size   = 32768;
	    Context->ignored_values_offset = 0;
	    Context->ignored_values = Alloc(Context->ignored_values_size,
					    char);
	}
    }

    StartIx = Context->ignored_values_offset;
    strcpy(Context->ignored_values + StartIx, S);
    Context->ignored_values_offset += Length;

    return StartIx;
}



/*************************************************************************/
/*									 */
/*	Free case space							 */
/*									 */
/*************************************************************************/


void FreeData(c50_context *Context)
/*   --------  */
{
    FreeCases();

    FreeUnlessNil(Context->ignored_values);
    Context->ignored_values = Nil;
    Context->ignored_values_size = Context->ignored_values_offset = 0;

    Free(Case);						Case = Nil;

    MaxCase = -1;
}



/*************************************************************************/
/*									 */
/*	Check for bad continuous value					 */
/*									 */
/*************************************************************************/


void CheckValue(c50_context *Context, DataRec DVec, Attribute Att)
/*   ----------  */
{
    ContValue	Cv;

    Cv = CVal(DVec, Att);
    if ( ! finite(Cv) )
    {
	Error(BADNUMBER, Context->schema.attribute_names[Att], "");

	CVal(DVec, Att) = UNKNOWN;
    }
}
