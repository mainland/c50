/* Modified 2026 by Geoffrey Mainland: native C++ library integration. */
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

#include <stdarg.h>
#include <cerrno>
#include <memory>
#include <limits>

static constexpr size_t RuleConditionChunk = 100;
static constexpr size_t SerializedRuleChunk = 100;

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


static void ValidateNumber(c50_context *Context, int Property)
{
    const char *value = Context->property_value + 1;
    char *end;
    double low = 0, high = FLT_MAX;
    bool integer = false;
    switch (Property)
    {
        case TYPEP: integer = true; high = BrSubset; break;
        case ENTRIESP: integer = true; low = 1; high = 1000; break;
        case FORKSP: integer = true; low = 1; high = INT_MAX - 1; break;
        case RULESP: case CONDSP: integer = true; high = INT_MAX - 1; break;
        case INITP: integer = true; high = 4095; break;
        case COSTSP: integer = true; low = high = 1; break;
        case SAMPLEP: high = 1; break;
        case COVERP: case OKP: case LIFTP: break;
        case CUTP: case LOWP: case MIDP: case HIGHP:
            high = std::numeric_limits<ContValue>::max(); low = -high; break;
        default: return;
    }
    errno = 0;
    const double number = integer ? strtol(value, &end, 10) : strtod(value, &end);
    if (end == value || *end != '"' || end[1] || errno == ERANGE ||
        !isfinite(number) || number < low || number > high ||
        (Property == LIFTP && static_cast<float>(number) <= 0))
        Error(Context, MODELFILE, "invalid numeric property", Context->property_name);
}

static unsigned PropertyBit(int property) { return 1u << property; }

static int ReadRecordProperty(c50_context *Context, c50_input *Input,
                               char *Delim, unsigned &Seen, unsigned Allowed)
{
    const int property = ReadProp(Context, Input, Delim);
    const unsigned bit = PropertyBit(property);
    if (!(Allowed & bit) || ((Seen & bit) && property != ELTSP))
        Error(Context, MODELFILE, "unexpected or repeated property", Context->property_name);
    Seen |= bit;
    return property;
}

static void ModelWriteError(c50_context *Context)
{
    if (Context->classifier_output.kind == C50_OUTPUT_MEMORY)
        throw std::bad_alloc();
    c50_record_error(Context, c50::error_code::io_error,
                     "could not write classifier");
    C50Exit(Context, 1);
}


static void ModelPrintf(c50_context *Context, const char *Format, ...)
{
    int Length;
    va_list Arguments;

    va_start(Arguments, Format);
    Length = c50_output_vprintf(&Context->classifier_output, Format, Arguments);
    va_end(Arguments);
    if ( Length < 0 ) ModelWriteError(Context);
}


static void ModelPutc(c50_context *Context, int Character)
{
    if ( c50_output_putc(Character, &Context->classifier_output) == EOF )
    {
        ModelWriteError(Context);
    }
}


/*************************************************************************/
/*									 */
/*	Check whether file is open.  If it is not, open it and		 */
/*	read/write sampling information and discrete names		 */
/*									 */
/*************************************************************************/


void CheckFile(c50_context *Context, const char *Extension, Boolean Write)
/*   ---------  */
{
    if ( Write )
    {
	if ( ! Context->classifier_output_active ||
	     ! Context->last_model_extension ||
	     strcmp(Context->last_model_extension, Extension) )
	{
	    if ( Context->classifier_output_active &&
		 Context->last_model_extension )
	    {
		ModelPrintf(Context, "\n");
		c50_output_close(&Context->classifier_output);
		Context->classifier_output_active = false;
	    }
	    Context->last_model_extension = Extension;
	    WriteFilePrefix(Context, Extension);
	}
    }
    else if ( ! Context->io.model_file ||
	      ! Context->last_model_extension ||
	      strcmp(Context->last_model_extension, Extension) )
    {
	Context->last_model_extension = Extension;
	CheckClose(Context->io.model_file);
	Context->io.model_file = Nil;
	ReadFilePrefix(Context, Extension);
    }
}



/*************************************************************************/
/*									 */
/*	Write information on system, sampling				 */
/*									 */
/*************************************************************************/


void WriteFilePrefix(c50_context *Context, const char *Extension)
/*   ---------------  */
{
    time_t	clock;
    struct tm	now;
    int		month;

    if ( ! Context->classifier_output_active )
    {
	FILE *ModelFile = GetFile(Context, Extension, "w");
	if ( ! ModelFile ) Error(Context, NOFILE, Context->io.file_name, E_ForWrite);
	c50_output_init_file(&Context->classifier_output, ModelFile, true);
	Context->classifier_output_active = true;
    }

    clock = time(0);
#ifdef _WIN32
    if ( localtime_s(&now, &clock) )
#else
    if ( ! localtime_r(&clock, &now) )
#endif
    {
	c50_record_error(Context, c50::error_code::internal_error,
			 "could not determine classifier timestamp");
	C50Exit(Context, 1);
    }
    month = now.tm_mon + 1;
    ModelPrintf(Context, "id=\"See5/C5.0 %s %d-%d%d-%d%d\"\n",
	    RELEASE,
	    now.tm_year + 1900,
	    month / 10, month % 10,
	    now.tm_mday / 10, now.tm_mday % 10);

    if ( Context->costs.matrix )
    {
	ModelPrintf(Context, "costs=\"1\"\n");
    }

    if ( Context->options.sample_fraction > 0 )
    {
	ModelPrintf(Context, "sample=\"%g\" init=\"%d\"\n", Context->options.sample_fraction, Context->io.random_initial_seed);
    }

    SaveDiscreteNames(Context);

    ModelPrintf(Context, "entries=\"%d\"\n", Context->options.trials);
}



/*************************************************************************/
/*									 */
/*	Read header information						 */
/*									 */
/*************************************************************************/


void ReadFilePrefix(c50_context *Context, const char *Extension)
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
	ModelPrintf(Context, "\n");
    }
}



/*************************************************************************/
/*									 */
/*	Save entire decision tree T in file with extension Extension	 */
/*									 */
/*************************************************************************/


void SaveTree(c50_context *Context, Tree T, const char *Extension)
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

    ModelPrintf(Context, "type=\"%d\"", T->NodeType);
    AsciiOut(Context, " class=", Context->schema.class_names[T->Leaf]);
    if ( T->Cases > 0 )
    {
	ModelPrintf(Context, " freq=\"%g", T->ClassDist[1]);
	ForEach(c, 2, Context->schema.max_class)
	{
	    ModelPrintf(Context, ",%g", T->ClassDist[c]);
	}
	ModelPrintf(Context, "\"");
    }

    if ( T->NodeType )
    {
	AsciiOut(Context, " att=", Context->schema.attribute_names[T->Tested]);
	ModelPrintf(Context, " forks=\"%d\"", T->Forks);

	switch ( T->NodeType )
	{
	    case BrDiscr:
		break;

	    case BrThresh:
		ModelPrintf(Context, " cut=\"%.*g\"", PREC+1, T->Cut);
		if ( T->Upper > T->Cut )
		{
		    ModelPrintf(Context, " low=\"%.*g\" mid=\"%.*g\" high=\"%.*g\"",
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
	ModelPrintf(Context, "\n");

	ForEach(v, 1, T->Forks)
	{
	    OutTree(Context, T->Branch[v]);
	}
    }
    else
    {
	ModelPrintf(Context, "\n");
    }
}



/*************************************************************************/
/*								  	 */
/*	Save the current ruleset in rules file				 */
/*								  	 */
/*************************************************************************/


void SaveRules(c50_context *Context, CRuleSet RS, const char *Extension)
/*   ---------  */
{
    int		ri, d;
    CRule	R;
    Condition	C;
    DiscrValue	v;
    Boolean	First;

    CheckFile(Context, Extension, true);

    ModelPrintf(Context, "rules=\"%d\"", RS->SNRules);
    AsciiOut(Context, " default=", Context->schema.class_names[RS->SDefault]);
    ModelPrintf(Context, "\n");

    ForEach(ri, 1, RS->SNRules)
    {
	R = RS->SRule[ri];
	ModelPrintf(Context, "conds=\"%d\" cover=\"%g\" ok=\"%g\" lift=\"%g\"",
		     R->Size, R->Cover, R->Correct,
		     (R->Correct + 1) / ((R->Cover + 2) * R->Prior));
	AsciiOut(Context, " class=", Context->schema.class_names[R->Rhs]);
	ModelPrintf(Context, "\n");

	ForEach(d, 1, R->Size)
	{
	    C = R->Lhs[d];

	    ModelPrintf(Context, "type=\"%d\"", C->NodeType);
	    AsciiOut(Context, " att=", Context->schema.attribute_names[C->Tested]);

	    switch ( C->NodeType )
	    {
		case BrDiscr:
		    AsciiOut(Context, " val=", Context->schema.attribute_value_names[C->Tested][C->TestValue]);
		    break;

		case BrThresh:
		    if ( C->TestValue == 1 )	/* N/A */
		    {
			ModelPrintf(Context, " val=\"N/A\"");
		    }
		    else
		    {
			ModelPrintf(Context, " cut=\"%.*g\" result=\"%c\"",
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

	    ModelPrintf(Context, "\n");
	}
    }
}



/*************************************************************************/
/*									 */
/*	Write ASCII string with prefix, escaping any quotes		 */
/*									 */
/*************************************************************************/


void AsciiOut(c50_context *Context, const char *Pre, const char *S)
/*   --------  */
{
    ModelPrintf(Context, "%s\"", Pre);
    while ( *S )
    {
	if ( *S == '"' || *S == '\\' ) ModelPutc(Context, '\\');
	ModelPutc(Context, *S++);
    }
    ModelPutc(Context, '"');
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
    Attribute	Att=0;
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

		if ( strlen(Context->property_value) >= 11 &&
                     sscanf(Context->property_value + strlen(Context->property_value) - 11,
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
                if (!Att || !StatBit(Att, DISCRETE) ||
                    Context->schema.max_attribute_value[Att] != 1)
                    Error(Context, MODELFILE, "invalid dynamic attribute dictionary", "");

		for ( p = Context->property_value ; *p ; )
		{
		    p = RemoveQuotes(p);
                    if (Context->schema.max_attribute_value[Att] >=
                        reinterpret_cast<intptr_t>(Context->schema.attribute_value_names[Att][0]))
                        Error(Context, MODELFILE, "too many dynamic attribute values", p);
		    v = Context->schema.max_attribute_value[Att] + 1;
		    EnsureDynamicValueSpace(Context, Att, v);
		    Context->schema.attribute_value_names[Att][v] = Pstrdup(Context, p);
		    Context->schema.max_attribute_value[Att] = v;

		    for ( p += strlen(p) ; *p != '"' ; p++ )
			;
		    p++;
		    if ( *p == ',' ) p++;
		}
		Context->schema.attribute_value_names[Att][Context->schema.max_attribute_value[Att]+1] =
		    Context->schema.other_attribute_value_name;
		Context->schema.max_discrete_value = Max(Context->schema.max_discrete_value, Context->schema.max_attribute_value[Att]+1);
		break;

	    case ENTRIESP:
		sscanf(Context->property_value, "\"%d\"", &Context->options.trials);
		Context->model_entry = 0;
		return;
            default:
                Error(Context, MODELFILE, "unexpected header property", Context->property_name);
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


Tree GetTree(c50_context *Context, const char *Extension)
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

    unsigned Seen = 0;
    do
    {
	switch ( ReadRecordProperty(Context, Input, &Delim, Seen,
                                  PropertyBit(TYPEP) | PropertyBit(CLASSP) | PropertyBit(ATTP) | PropertyBit(CUTP) | PropertyBit(LOWP) | PropertyBit(MIDP) | PropertyBit(HIGHP) | PropertyBit(FORKSP) | PropertyBit(FREQP) | PropertyBit(ELTSP)) )
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
                p = Context->property_value + 1;
                ForEach(c, 1, Context->schema.max_class)
                {
                    char *end;
                    const double frequency = strtod(p, &end);
                    if (end == p || !isfinite(frequency) || frequency < 0 ||
                        frequency > FLT_MAX ||
                        *end != (c == Context->schema.max_class ? '"' : ','))
                        Error(Context, MODELFILE, "invalid class frequencies", "");
                    T->ClassDist[c] = frequency;
                    T->Cases += T->ClassDist[c];
                    p = end + 1;
                }
                if (*p || !isfinite(T->Cases))
                    Error(Context, MODELFILE, "invalid class frequencies", "");
		break;

	    case ELTSP:
                if (T->NodeType != BrSubset || !T->Tested || !Discrete(T->Tested) ||
                    T->Forks < 1 || T->Forks > Context->schema.max_attribute_value[T->Tested] ||
                    Subset >= T->Forks)
                    Error(Context, MODELFILE, "invalid subset branch", "");
		if ( ! Subset++ )
		{
		    T->Subset = AllocZero(T->Forks+1, Set);
		}

		T->Subset[Subset] = MakeSubset(Context, T->Tested);
		break;
	}
    }
    while ( Delim == ' ' );

    if (!(Seen & PropertyBit(TYPEP)) || !T->Leaf)
        Error(Context, MODELFILE, "missing tree type or class", "");
    if (T->NodeType)
    {
        if (!T->Tested || T->Forks < 1 ||
            (T->NodeType == BrThresh &&
             (!Continuous(T->Tested) || T->Forks != 3 || !(Seen & PropertyBit(CUTP)) ||
              T->Lower > T->Mid || T->Mid > T->Upper)) ||
            (T->NodeType != BrThresh &&
             (!Discrete(T->Tested) || T->Forks > Context->schema.max_attribute_value[T->Tested])) ||
            (T->NodeType == BrSubset && Subset != T->Forks))
            Error(Context, MODELFILE, "inconsistent tree branches", "");
    }
    else if (T->Forks || T->Tested || Subset)
        Error(Context, MODELFILE, "leaf has split properties", "");

    if ( T->ClassDist )
    {
	T->Errors = T->Cases - T->ClassDist[T->Leaf];
    }
    else
    {
	T->ClassDist = Alloc(Context->schema.max_class + 1, CaseCount);
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


CRuleSet GetRules(c50_context *Context, const char *Extension)
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
    RuleNo	r, RuleCount=0;
    char	Delim, *Unquoted;

    RS = Alloc(1, RuleSetRec);
    *Slot = RS;

    unsigned Seen = 0;
    do
    {
	switch ( ReadRecordProperty(Context, Input, &Delim, Seen,
                                  PropertyBit(RULESP) | PropertyBit(DEFAULTP)) )
	{
	    case ERRORP:
		return Nil;

	    case RULESP:
		sscanf(Context->property_value, "\"%d\"", &RuleCount);
		break;

	    case DEFAULTP:
		Unquoted = RemoveQuotes(Context->property_value);
		RS->SDefault = Which(Unquoted, Context->schema.class_names, 1, Context->schema.max_class);
		if ( ! RS->SDefault ) Error(Context, MODELFILE, E_MFCLASS, Unquoted);
		break;
	}
    }
    while ( Delim == ' ' );

    if (!(Seen & PropertyBit(RULESP)) || RuleCount < 0 || !RS->SDefault)
        Error(Context, MODELFILE, "missing ruleset count or default", "");

    /*  Read each rule  */

    const size_t FullSize = static_cast<size_t>(RuleCount) + 1;
    size_t Allocated = FullSize < SerializedRuleChunk + 1 ?
                       FullSize : SerializedRuleChunk + 1;
    RS->SRule = Alloc(Allocated, CRule);
    ForEach(r, 1, RuleCount)
    {
	if ( static_cast<size_t>(r) >= Allocated )
	{
	    const size_t NewSize =
		Allocated + SerializedRuleChunk < FullSize ?
		Allocated + SerializedRuleChunk : FullSize;
	    RS->SRule = static_cast<CRule *>(
		Prealloc(Context, RS->SRule, NewSize * sizeof(CRule)));
	    Allocated = NewSize;
	}
	RS->SRule[r] = Nil;
	RS->SNRules = r;
	if ( InRuleAt(Context, Input, &RS->SRule[r]) )
	{
	    RS->SRule[r]->RNo = r;
	    RS->SRule[r]->TNo = Context->model_entry;
	}
    }
    CheckActiveSpace(Context, RS->SNRules);
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
    int		d, ConditionCount=0;
    char	Delim, *Unquoted;
    float	Lift;

    R = Alloc(1, RuleRec);
    *Slot = R;

    unsigned Seen = 0;
    do
    {
	switch ( ReadRecordProperty(Context, Input, &Delim, Seen,
                                  PropertyBit(CONDSP) | PropertyBit(COVERP) | PropertyBit(OKP) | PropertyBit(LIFTP) | PropertyBit(CLASSP)) )
	{
	    case ERRORP:
		return Nil;

	    case CONDSP:
		sscanf(Context->property_value, "\"%d\"", &ConditionCount);
		break;

	    case COVERP:
		sscanf(Context->property_value, "\"%f\"", &R->Cover);
		break;

	    case OKP:
		sscanf(Context->property_value, "\"%f\"", &R->Correct);
		break;

	    case LIFTP:
		sscanf(Context->property_value, "\"%f\"", &Lift);
		break;

	    case CLASSP:
		Unquoted = RemoveQuotes(Context->property_value);
		R->Rhs = Which(Unquoted, Context->schema.class_names, 1, Context->schema.max_class);
		if ( ! R->Rhs ) Error(Context, MODELFILE, E_MFCLASS, Unquoted);
		break;
	}
    }
    while ( Delim == ' ' );

    const unsigned required = PropertyBit(CONDSP) | PropertyBit(COVERP) |
                              PropertyBit(OKP) | PropertyBit(LIFTP) | PropertyBit(CLASSP);
    if ((Seen & required) != required || ConditionCount < 0 ||
        R->Correct > R->Cover)
        Error(Context, MODELFILE, "incomplete or inconsistent rule", "");
    R->Prior = (R->Correct + 1) / ((R->Cover + 2) * Lift);
    if (!isfinite(R->Prior) || R->Prior <= 0)
        Error(Context, MODELFILE, "invalid rule prior", "");

    const size_t FullSize = static_cast<size_t>(ConditionCount) + 1;
    size_t Allocated = FullSize < RuleConditionChunk + 1 ?
                       FullSize : RuleConditionChunk + 1;
    R->Lhs = Alloc(Allocated, Condition);
    ForEach(d, 1, ConditionCount)
    {
	if ( static_cast<size_t>(d) >= Allocated )
	{
	    const size_t NewSize =
		Allocated + RuleConditionChunk < FullSize ?
		Allocated + RuleConditionChunk : FullSize;
	    R->Lhs = static_cast<Condition *>(
		Prealloc(Context, R->Lhs, NewSize * sizeof(Condition)));
	    Allocated = NewSize;
	}
	R->Lhs[d] = Nil;
	R->Size = d;
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

    unsigned Seen = 0;
    do
    {
	switch ( ReadRecordProperty(Context, Input, &Delim, Seen,
                                  PropertyBit(TYPEP) | PropertyBit(ATTP) | PropertyBit(CUTP) | PropertyBit(RESULTP) | PropertyBit(VALP) | PropertyBit(ELTSP)) )
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
                if (strcmp(Context->property_value, "\"<\"") &&
                    strcmp(Context->property_value, "\">\""))
                    Error(Context, MODELFILE, "invalid threshold outcome", "");
		C->TestValue = ( Context->property_value[1] == '<' ? 2 : 3 );
		break;

	    case VALP:
                if (!C->Tested)
                    Error(Context, MODELFILE, "condition value precedes attribute", "");
		if ( Continuous(C->Tested) )
		{
                    if (strcmp(Context->property_value, "\"N/A\""))
                        Error(Context, MODELFILE, "invalid not-applicable value", "");
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
                if (C->NodeType != BrSubset || C->Subset)
                    Error(Context, MODELFILE, "invalid condition subset", "");
		C->Subset = MakeSubset(Context, C->Tested);
		C->TestValue = 1;
		break;
	}
    }
    while ( Delim == ' ' );

    if (!C->Tested || !C->TestValue ||
        (C->NodeType == BrDiscr && (!Discrete(C->Tested) || !(Seen & PropertyBit(VALP)))) ||
        (C->NodeType == BrThresh &&
         (!Continuous(C->Tested) ||
          (!(Seen & PropertyBit(VALP)) && !(Seen & PropertyBit(CUTP))))) ||
        (C->NodeType == BrSubset && (!Discrete(C->Tested) || !C->Subset)) ||
        C->NodeType < BrDiscr || C->NodeType > BrSubset)
        Error(Context, MODELFILE, "incomplete or inconsistent condition", "");
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

	if ( p - Context->property_value >= INT_MAX - 10003 )
            throw std::bad_alloc();
	if ( (i = p - Context->property_value) >= Context->property_value_size )
	{
            const int capacity = Context->property_value_size + 10000;
            Realloc(Context->property_value, capacity + 3, char);
            Context->property_value_size = capacity;
	    p = Context->property_value + i;
	}

	*p++ = c;
	if ( c == '\\' )
	{
            c = c50_input_getc(Input);
            if (c == EOF) Error(Context, MODELFILE, E_MFEOF, "");
	    *p++ = c;
	}
	else
	if ( c == '"' )
	{
	    Quote = ! Quote;
	}
    }
    *p = '\00';
    *Delim = c;

    const int property = WhichProperty(Context->property_name);
    if (!property) Error(Context, MODELFILE, "unknown property", Context->property_name);

    // All scalar properties are quoted. Only elts permits a quoted list.
    // Validate before the legacy unquoting helpers modify this buffer.
    for (const char *value = Context->property_value; ; )
    {
        if (*value++ != '"')
            Error(Context, MODELFILE, "expected quoted property", Context->property_name);
        while (*value && *value != '"')
        {
            if (*value == '\\' && !*++value)
                Error(Context, MODELFILE, "incomplete escape", Context->property_name);
            ++value;
        }
        if (*value++ != '"')
            Error(Context, MODELFILE, "unterminated property", Context->property_name);
        if (!*value) break;
        if (property != ELTSP || *value++ != ',')
            Error(Context, MODELFILE, "invalid property suffix", Context->property_name);
    }
    ValidateNumber(Context, property);
    return property;
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

    if (Att < 1 || Att > Context->schema.max_attribute || !Discrete(Att))
        Error(Context, MODELFILE, "subset requires a discrete attribute", "");
    Bytes = (Context->schema.max_attribute_value[Att]>>3) + 1;
    S = AllocZero(Bytes, Byte);
    std::unique_ptr<Byte, decltype(&free)> owner(S, &free);

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

    return owner.release();
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
