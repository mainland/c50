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
/*	Routines for saving and reading model files			 */
/*	-------------------------------------------			 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

static const char PropertyNames[] =
    "null\0att\0class\0cut\0conds\0elts\0entries\0forks\0freq\0id\0"
    "type\0low\0mid\0high\0result\0rules\0val\0lift\0cover\0ok\0"
    "default\0costs\0sample\0init\0";

#define ERRORP		0
#define ATTP		1
#define CLASSP		2
#define CUTP		3
#define	CONDSP		4
#define ELTSP		5
#define ENTRIESP	6
#define FORKSP		7
#define FREQP		8
#define IDP		9
#define TYPEP		10
#define LOWP		11
#define MIDP		12
#define HIGHP		13
#define RESULTP		14
#define RULESP		15
#define VALP		16
#define LIFTP		17
#define COVERP		18
#define OKP		19
#define DEFAULTP	20
#define COSTSP		21
#define SAMPLEP		22
#define INITP		23


static int WhichProperty(const char *Name)
{
    const char *Property;
    int Number;

    for ( Number = 0, Property = PropertyNames ; *Property ; Number++ )
    {
	if ( ! strcmp(Name, Property) ) return Number;
	Property += strlen(Property) + 1;
    }

    return 0;
}


/*************************************************************************/
/*									 */
/*	Check whether file is open.  If it is not, open it and		 */
/*	read/write sampling information and discrete names		 */
/*									 */
/*************************************************************************/


void CheckFile(c50_context *Context, String Extension, Boolean Write)
/*   ---------  */
{
    if ( ! Context->io.model_file || ! Context->last_model_extension ||
	 strcmp(Context->last_model_extension, Extension) )
    {
	Context->last_model_extension = Extension;

	if ( Context->io.model_file )
	{
	    fprintf(Context->io.model_file, "\n");
	    fclose(Context->io.model_file);
	}

	if ( Write )
	{
	    WriteFilePrefix(Context, Extension);
	}
	else
	{
	    ReadFilePrefix(Context, Extension);
	}
    }
}



/*************************************************************************/
/*									 */
/*	Write information on system, sampling				 */
/*									 */
/*************************************************************************/


void WriteFilePrefix(c50_context *Context, String Extension)
/*   ---------------  */
{
    time_t	clock;
    struct tm	*now;

    if ( ! (Context->io.model_file = GetFile(Context, Extension, "w")) )
    {
	Error(Context, NOFILE, Context->io.file_name, E_ForWrite);
    }

    clock = time(0);
    now = localtime(&clock);
    now->tm_mon++;
    fprintf(Context->io.model_file, "id=\"See5/C5.0 %s %d-%d%d-%d%d\"\n",
	    RELEASE,
	    now->tm_year + 1900,
	    now->tm_mon / 10, now->tm_mon % 10,
	    now->tm_mday / 10, now->tm_mday % 10);

    if ( Context->costs.matrix )
    {
	fprintf(Context->io.model_file, "costs=\"1\"\n");
    }

    if ( Context->options.sample_fraction > 0 )
    {
	fprintf(Context->io.model_file, "sample=\"%g\" init=\"%d\"\n", Context->options.sample_fraction, Context->io.random_initial_seed);
    }

    SaveDiscreteNames(Context);

    fprintf(Context->io.model_file, "entries=\"%d\"\n", Context->options.trials);
}



/*************************************************************************/
/*									 */
/*	Read header information						 */
/*									 */
/*************************************************************************/


void ReadFilePrefix(c50_context *Context, String Extension)
/*   --------------  */
{
    if ( ! (Context->io.model_file = GetFile(Context, Extension, "r")) ) Error(Context, NOFILE, Context->io.file_name, "");

    c50_input_init_file(&Context->classifier_input, Context->io.model_file);
    StreamIn(&Context->classifier_input, (char *) &Context->options.trials, sizeof(int));
    if ( memcmp((char *) &Context->options.trials, "id=", 3) != 0 )
    {
	printf("\nCannot read old format classifiers\n");
	C50Exit(Context, 1);
    }
    else
    {
	c50_input_rewind(&Context->classifier_input);
	ReadHeader(Context, &Context->classifier_input);
    }
}



/*************************************************************************/
/*									 */
/*	Save attribute values read with "discrete N"			 */
/*									 */
/*************************************************************************/


void SaveDiscreteNames(c50_context *Context)
/*   -----------------  */
{
    Attribute	Att;
    DiscrValue	v;

    ForEach(Att, 1, Context->schema.max_attribute)
    {
	if ( ! StatBit(Att, DISCRETE) || Context->schema.max_attribute_value[Att] < 2 ) continue;

	AsciiOut(Context, "att=", Context->schema.attribute_names[Att]);
	AsciiOut(Context, " elts=", Context->schema.attribute_value_names[Att][2]); 	/* skip N/A */

	ForEach(v, 3, Context->schema.max_attribute_value[Att])
	{
	    AsciiOut(Context, ",", Context->schema.attribute_value_names[Att][v]);
	}
	fprintf(Context->io.model_file, "\n");
    }
}



/*************************************************************************/
/*									 */
/*	Save entire decision tree T in file with extension Extension	 */
/*									 */
/*************************************************************************/


void SaveTree(c50_context *Context, Tree T, String Extension)
/*   --------  */
{
    CheckFile(Context, Extension, true);

    OutTree(Context, T);
}



void OutTree(c50_context *Context, Tree T)
/*   -------  */
{
    DiscrValue	v, vv;
    ClassNo	c;
    Boolean	First;

    fprintf(Context->io.model_file, "type=\"%d\"", T->NodeType);
    AsciiOut(Context, " class=", Context->schema.class_names[T->Leaf]);
    if ( T->Cases > 0 )
    {
	fprintf(Context->io.model_file, " freq=\"%g", T->ClassDist[1]);
	ForEach(c, 2, Context->schema.max_class)
	{
	    fprintf(Context->io.model_file, ",%g", T->ClassDist[c]);
	}
	fprintf(Context->io.model_file, "\"");
    }

    if ( T->NodeType )
    {
	AsciiOut(Context, " att=", Context->schema.attribute_names[T->Tested]);
	fprintf(Context->io.model_file, " forks=\"%d\"", T->Forks);

	switch ( T->NodeType )
	{
	    case BrDiscr:
		break;

	    case BrThresh:
		fprintf(Context->io.model_file, " cut=\"%.*g\"", PREC+1, T->Cut);
		if ( T->Upper > T->Cut )
		{
		    fprintf(Context->io.model_file, " low=\"%.*g\" mid=\"%.*g\" high=\"%.*g\"",
				 PREC, T->Lower, PREC, T->Mid, PREC, T->Upper);
		}
		break;

	    case BrSubset:
		ForEach(v, 1, T->Forks)
		{
		    First=true;
		    ForEach(vv, 1, Context->schema.max_attribute_value[T->Tested])
		    {
			if ( In(vv, T->Subset[v]) )
			{
			    if ( First )
			    {
				AsciiOut(Context, " elts=", Context->schema.attribute_value_names[T->Tested][vv]);
				First = false;
			    }
			    else
			    {
				AsciiOut(Context, ",", Context->schema.attribute_value_names[T->Tested][vv]);
			    }
			}
		    }
		    /*  Make sure have printed at least one element  */

		    if ( First ) AsciiOut(Context, " elts=", "N/A");
		}
		break;
	}
	fprintf(Context->io.model_file, "\n");

	ForEach(v, 1, T->Forks)
	{
	    OutTree(Context, T->Branch[v]);
	}
    }
    else
    {
	fprintf(Context->io.model_file, "\n");
    }
}



/*************************************************************************/
/*								  	 */
/*	Save the current ruleset in rules file				 */
/*								  	 */
/*************************************************************************/


void SaveRules(c50_context *Context, CRuleSet RS, String Extension)
/*   ---------  */
{
    int		ri, d;
    CRule	R;
    Condition	C;
    DiscrValue	v;
    Boolean	First;

    CheckFile(Context, Extension, true);

    fprintf(Context->io.model_file, "rules=\"%d\"", RS->SNRules);
    AsciiOut(Context, " default=", Context->schema.class_names[RS->SDefault]);
    fprintf(Context->io.model_file, "\n");

    ForEach(ri, 1, RS->SNRules)
    {
	R = RS->SRule[ri];
	fprintf(Context->io.model_file, "conds=\"%d\" cover=\"%g\" ok=\"%g\" lift=\"%g\"",
		     R->Size, R->Cover, R->Correct,
		     (R->Correct + 1) / ((R->Cover + 2) * R->Prior));
	AsciiOut(Context, " class=", Context->schema.class_names[R->Rhs]);
	fprintf(Context->io.model_file, "\n");

	ForEach(d, 1, R->Size)
	{
	    C = R->Lhs[d];

	    fprintf(Context->io.model_file, "type=\"%d\"", C->NodeType);
	    AsciiOut(Context, " att=", Context->schema.attribute_names[C->Tested]);

	    switch ( C->NodeType )
	    {
		case BrDiscr:
		    AsciiOut(Context, " val=", Context->schema.attribute_value_names[C->Tested][C->TestValue]);
		    break;

		case BrThresh:
		    if ( C->TestValue == 1 )	/* N/A */
		    {
			fprintf(Context->io.model_file, " val=\"N/A\"");
		    }
		    else
		    {
			fprintf(Context->io.model_file, " cut=\"%.*g\" result=\"%c\"",
				     PREC+1, C->Cut,
				     ( C->TestValue == 2 ? '<' : '>' ));
		    }
		    break;

		case BrSubset:
		    First=true;
		    ForEach(v, 1, Context->schema.max_attribute_value[C->Tested])
		    {
			if ( In(v, C->Subset) )
			{
			    if ( First )
			    {
				AsciiOut(Context, " elts=", Context->schema.attribute_value_names[C->Tested][v]);
				First = false;
			    }
			    else
			    {
				AsciiOut(Context, ",", Context->schema.attribute_value_names[C->Tested][v]);
			    }
			}
		    }
		    break;
	    }

	    fprintf(Context->io.model_file, "\n");
	}
    }
}



/*************************************************************************/
/*									 */
/*	Write ASCII string with prefix, escaping any quotes		 */
/*									 */
/*************************************************************************/


void AsciiOut(c50_context *Context, String Pre, String S)
/*   --------  */
{
    fprintf(Context->io.model_file, "%s\"", Pre);
    while ( *S )
    {
	if ( *S == '"' || *S == '\\' ) fputc('\\', Context->io.model_file);
	fputc(*S++, Context->io.model_file);
    }
    fputc('"', Context->io.model_file);
}



/*************************************************************************/
/*								  	 */
/*	Read the header information (id, saved names, models)		 */
/*								  	 */
/*************************************************************************/


static void ReadHeaderFrom(c50_context *Context, c50_input *Input,
			   c50_input *CostsInput,
			   Boolean AllowFileCosts)
/*          --------------  */
{
    Attribute	Att;
    DiscrValue	v;
    char	*p, *Unquoted, Dummy;
    int		Year, Month, Day;
    FILE	*F;

    while ( true )
    {
	switch ( ReadProp(Context, Input, &Dummy) )
	{
	    case ERRORP:
		return;

	    case IDP:
		/*  Recover year run and set base date for timestamps  */

		if ( sscanf(Context->property_value + strlen(Context->property_value) - 11,
			    "%d-%d-%d\"", &Year, &Month, &Day) == 3 )
		{
		    SetTSBase(Context, Year);
		}
		break;

	    case COSTSP:
		/*  Recover costs file used to generate model  */

		if ( CostsInput )
		{
		    GetMCostsInput(Context, CostsInput);
		}
		else
		if ( AllowFileCosts && (F = GetFile(Context, ".costs", "r")) )
		{
		    GetMCosts(Context, F);
		}
		else
		{
		    Error(Context, NOFILE, Context->io.file_name, "costs input required by model");
		}
		break;
	    case SAMPLEP:
		sscanf(Context->property_value, "\"%f\"", &Context->options.sample_fraction);
		break;

	    case INITP:
		sscanf(Context->property_value, "\"%d\"", &Context->io.random_initial_seed);
		break;

	    case ATTP:
		Unquoted = RemoveQuotes(Context->property_value);
		Att = Which(Unquoted, Context->schema.attribute_names, 1, Context->schema.max_attribute);
		if ( ! Att || Exclude(Att) )
		{
		    Error(Context, MODELFILE, E_MFATT, Unquoted);
		}
		break;

	    case ELTSP:
		Context->schema.max_attribute_value[Att] = 1;
		Context->schema.attribute_value_names[Att][1] = strdup("N/A");

		for ( p = Context->property_value ; *p ; )
		{
		    p = RemoveQuotes(p);
		    v = ++Context->schema.max_attribute_value[Att];
		    Context->schema.attribute_value_names[Att][v] = strdup(p);

		    for ( p += strlen(p) ; *p != '"' ; p++ )
			;
		    p++;
		    if ( *p == ',' ) p++;
		}
		Context->schema.attribute_value_names[Att][Context->schema.max_attribute_value[Att]+1] = "<other>";
		Context->schema.max_discrete_value = Max(Context->schema.max_discrete_value, Context->schema.max_attribute_value[Att]+1);
		break;

	    case ENTRIESP:
		sscanf(Context->property_value, "\"%d\"", &Context->options.trials);
		Context->model_entry = 0;
		return;
	}
    }
}



void ReadHeader(c50_context *Context, c50_input *Input)
/*   ----------  */
{
    ReadHeaderFrom(Context, Input, Nil, true);
}



void ReadHeaderMemory(c50_context *Context, c50_input *Input,
		      c50_input *CostsInput)
/*   ----------------  */
{
    ReadHeaderFrom(Context, Input, CostsInput, false);
}



/*************************************************************************/
/*									 */
/*	Retrieve decision tree with extension Extension			 */
/*									 */
/*************************************************************************/


Tree GetTree(c50_context *Context, String Extension)
/*   -------  */
{
    CheckFile(Context, Extension, false);

    return InTree(Context, &Context->classifier_input);
}



Tree InTree(c50_context *Context, c50_input *Input)
/*   ------  */
{
    Tree T=Nil;

    return InTreeAt(Context, Input, &T);
}



Tree InTreeAt(c50_context *Context, c50_input *Input, Tree *Slot)
/*   --------  */
{
    Tree	T;
    DiscrValue	v, Subset=0;
    char	Delim, *p, *Unquoted;
    ClassNo	c;
    int		X;
    double	XD;

    T = (Tree) AllocZero(1, TreeRec);
    *Slot = T;

    do
    {
	switch ( ReadProp(Context, Input, &Delim) )
	{
	    case ERRORP:
		return Nil;

	    case TYPEP:
		sscanf(Context->property_value, "\"%d\"", &X); T->NodeType = X;
		break;

	    case CLASSP:
		Unquoted = RemoveQuotes(Context->property_value);
		T->Leaf = Which(Unquoted, Context->schema.class_names, 1, Context->schema.max_class);
		if ( ! T->Leaf ) Error(Context, MODELFILE, E_MFCLASS, Unquoted);
		break;

	    case ATTP:
		Unquoted = RemoveQuotes(Context->property_value);
		T->Tested = Which(Unquoted, Context->schema.attribute_names, 1, Context->schema.max_attribute);
		if ( ! T->Tested || Exclude(T->Tested) )
		{
		    Error(Context, MODELFILE, E_MFATT, Unquoted);
		}
		break;

	    case CUTP:
		sscanf(Context->property_value, "\"%lf\"", &XD);	T->Cut = XD;
		T->Lower = T->Mid = T->Upper = T->Cut;
		break;

	    case LOWP:
		sscanf(Context->property_value, "\"%lf\"", &XD);	T->Lower = XD;
		break;

	    case MIDP:
		sscanf(Context->property_value, "\"%lf\"", &XD);	T->Mid = XD;
		break;

	    case HIGHP:
		sscanf(Context->property_value, "\"%lf\"", &XD);	T->Upper = XD;
		break;

	    case FORKSP:
		sscanf(Context->property_value, "\"%d\"", &T->Forks);
		break;

	    case FREQP:
		T->ClassDist = Alloc(Context->schema.max_class+1, CaseCount);
		p = Context->property_value+1;

		ForEach(c, 1, Context->schema.max_class)
		{
		    T->ClassDist[c] = strtod(p, &p);
		    T->Cases += T->ClassDist[c];
		    p++;
		}
		break;

	    case ELTSP:
		if ( ! Subset++ )
		{
		    T->Subset = AllocZero(T->Forks+1, Set);
		}

		T->Subset[Subset] = MakeSubset(Context, T->Tested);
		break;
	}
    }
    while ( Delim == ' ' );

    if ( T->ClassDist )
    {
	T->Errors = T->Cases - T->ClassDist[T->Leaf];
    }
    else
    {
	T->ClassDist = Alloc(1, CaseCount);
    }

    if ( T->NodeType )
    {
	T->Branch = AllocZero(T->Forks+1, Tree);
	ForEach(v, 1, T->Forks)
	{
	    InTreeAt(Context, Input, &T->Branch[v]);
	}
    }

    return T;
}



/*************************************************************************/
/*									 */
/*	Retrieve ruleset with extension Extension			 */
/*	(Separate functions for ruleset, single rule, single condition)	 */
/*									 */
/*************************************************************************/


CRuleSet GetRules(c50_context *Context, String Extension)
/*	 --------  */
{
    CheckFile(Context, Extension, false);

    return InRules(Context, &Context->classifier_input);
}



CRuleSet InRules(c50_context *Context, c50_input *Input)
/*	 -------  */
{
    CRuleSet RS=Nil;

    return InRulesAt(Context, Input, &RS);
}



CRuleSet InRulesAt(c50_context *Context, c50_input *Input, CRuleSet *Slot)
/*	 ---------  */
{
    CRuleSet	RS;
    RuleNo	r;
    char	Delim, *Unquoted;

    RS = Alloc(1, RuleSetRec);
    *Slot = RS;

    do
    {
	switch ( ReadProp(Context, Input, &Delim) )
	{
	    case ERRORP:
		return Nil;

	    case RULESP:
		sscanf(Context->property_value, "\"%d\"", &RS->SNRules);
		CheckActiveSpace(Context, RS->SNRules);
		break;

	    case DEFAULTP:
		Unquoted = RemoveQuotes(Context->property_value);
		RS->SDefault = Which(Unquoted, Context->schema.class_names, 1, Context->schema.max_class);
		if ( ! RS->SDefault ) Error(Context, MODELFILE, E_MFCLASS, Unquoted);
		break;
	}
    }
    while ( Delim == ' ' );

    /*  Read each rule  */

    RS->SRule = Alloc(RS->SNRules+1, CRule);
    ForEach(r, 1, RS->SNRules)
    {
	if ( InRuleAt(Context, Input, &RS->SRule[r]) )
	{
	    RS->SRule[r]->RNo = r;
	    RS->SRule[r]->TNo = Context->model_entry;
	}
    }
    ConstructRuleTree(Context, RS);
    Context->model_entry++;
    return RS;
}



CRule InRule(c50_context *Context, c50_input *Input)
/*    ------  */
{
    CRule R=Nil;

    return InRuleAt(Context, Input, &R);
}



CRule InRuleAt(c50_context *Context, c50_input *Input, CRule *Slot)
/*    --------  */
{
    CRule	R;
    int		d;
    char	Delim, *Unquoted;
    float	Lift;

    R = Alloc(1, RuleRec);
    *Slot = R;

    do
    {
	switch ( ReadProp(Context, Input, &Delim) )
	{
	    case ERRORP:
		return Nil;

	    case CONDSP:
		sscanf(Context->property_value, "\"%d\"", &R->Size);
		break;

	    case COVERP:
		sscanf(Context->property_value, "\"%f\"", &R->Cover);
		break;

	    case OKP:
		sscanf(Context->property_value, "\"%f\"", &R->Correct);
		break;

	    case LIFTP:
		sscanf(Context->property_value, "\"%f\"", &Lift);
		R->Prior = (R->Correct + 1) / ((R->Cover + 2) * Lift);
		break;

	    case CLASSP:
		Unquoted = RemoveQuotes(Context->property_value);
		R->Rhs = Which(Unquoted, Context->schema.class_names, 1, Context->schema.max_class);
		if ( ! R->Rhs ) Error(Context, MODELFILE, E_MFCLASS, Unquoted);
		break;
	}
    }
    while ( Delim == ' ' );

    R->Lhs = Alloc(R->Size+1, Condition);
    ForEach(d, 1, R->Size)
    {
	InConditionAt(Context, Input, &R->Lhs[d]);
    }

    R->Vote = 1000 * (R->Correct + 1.0) / (R->Cover + 2.0) + 0.5;

    return R;
}



Condition InCondition(c50_context *Context, c50_input *Input)
/*        -----------  */
{
    Condition C=Nil;

    return InConditionAt(Context, Input, &C);
}



Condition InConditionAt(c50_context *Context, c50_input *Input,
			Condition *Slot)
/*        -------------  */
{
    Condition	C;
    char	Delim, *Unquoted;
    int		X;
    double	XD;

    C = Alloc(1, CondRec);
    *Slot = C;

    do
    {
	switch ( ReadProp(Context, Input, &Delim) )
	{
	    case ERRORP:
		return Nil;

	    case TYPEP:
		sscanf(Context->property_value, "\"%d\"", &X); C->NodeType = X;
		break;

	    case ATTP:
		Unquoted = RemoveQuotes(Context->property_value);
		C->Tested = Which(Unquoted, Context->schema.attribute_names, 1, Context->schema.max_attribute);
		if ( ! C->Tested || Exclude(C->Tested) )
		{
		    Error(Context, MODELFILE, E_MFATT, Unquoted);
		}
		break;

	    case CUTP:
		sscanf(Context->property_value, "\"%lf\"", &XD);	C->Cut = XD;
		break;

	    case RESULTP:
		C->TestValue = ( Context->property_value[1] == '<' ? 2 : 3 );
		break;

	    case VALP:
		if ( Continuous(C->Tested) )
		{
		    C->TestValue = 1;
		}
		else
		{
		    Unquoted = RemoveQuotes(Context->property_value);
		    C->TestValue = Which(Unquoted,
					 Context->schema.attribute_value_names[C->Tested],
					 1, Context->schema.max_attribute_value[C->Tested]);
		    if ( ! C->TestValue ) Error(Context, MODELFILE, E_MFATTVAL, Unquoted);
		}
		break;

	    case ELTSP:
		C->Subset = MakeSubset(Context, C->Tested);
		C->TestValue = 1;
		break;
	}
    }
    while ( Delim == ' ' );

    return C;
}



/*************************************************************************/
/*									 */
/*	ASCII reading utilities						 */
/*									 */
/*************************************************************************/


int ReadProp(c50_context *Context, c50_input *Input, char *Delim)
/*  --------  */
{
    int		c, i;
    char	*p;
    Boolean	Quote=false;

    if ( ! Context->property_value )
    {
	Context->property_value_size = 10000;
	Context->property_value =
	    Alloc(Context->property_value_size + 3, char);
    }

    for ( p = Context->property_name ; (c = c50_input_getc(Input)) != '=' ;  )
    {
	if ( p - Context->property_name >= 19 || c == EOF )
	{
	    Error(Context, MODELFILE, E_MFEOF, "");
	    Context->property_name[0] = Context->property_value[0] = *Delim = '\00';
	    return 0;
	}
	*p++ = c;
    }
    *p = '\00';

    for ( p = Context->property_value ;
	  ((c = c50_input_getc(Input)) != ' ' && c != '\n') || Quote ; )
    {
	if ( c == EOF )
	{
	    Error(Context, MODELFILE, E_MFEOF, "");
	    Context->property_name[0] = Context->property_value[0] = '\00';
	    return 0;
	}

	if ( (i = p - Context->property_value) >= Context->property_value_size )
	{
	    Realloc(Context->property_value, (Context->property_value_size += 10000) + 3, char);
	    p = Context->property_value + i;
	}

	*p++ = c;
	if ( c == '\\' )
	{
	    *p++ = c50_input_getc(Input);
	}
	else
	if ( c == '"' )
	{
	    Quote = ! Quote;
	}
    }
    *p = '\00';
    *Delim = c;

    return WhichProperty(Context->property_name);
}


String RemoveQuotes(String S)
/*     ------------  */
{
    char	*p, *Start;

    p = Start = S;
    
    for ( S++ ; *S != '"' ; S++ )
    {
	if ( *S == '\\' ) S++;
	*p++ = *S;
	*S = '-';
    }
    *p = '\00';

    return Start;
}



Set MakeSubset(c50_context *Context, Attribute Att)
/*  ----------  */
{
    int		Bytes, b;
    char	*p;
    Set		S;

    Bytes = (Context->schema.max_attribute_value[Att]>>3) + 1;
    S = AllocZero(Bytes, Byte);

    for ( p = Context->property_value ; *p ; )
    {
	p = RemoveQuotes(p);
	b = Which(p, Context->schema.attribute_value_names[Att], 1, Context->schema.max_attribute_value[Att]);
	if ( ! b ) Error(Context, MODELFILE, E_MFATTVAL, p);
	SetBit(b, S);

	for ( p += strlen(p) ; *p != '"' ; p++ )
	    ;
	p++;
	if ( *p == ',' ) p++;
    }

    return S;
}



/*************************************************************************/
/*								  	 */
/*	Character stream read for binary routines			 */
/*								  	 */
/*************************************************************************/


void StreamIn(c50_input *Input, String S, int n)
/*   --------  */
{
    while ( n-- ) *S++ = c50_input_getc(Input);
}
