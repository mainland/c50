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
/*	Get names of classes, attributes and attribute values		 */
/*	-----------------------------------------------------		 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"
#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>

/*************************************************************************/
/*									 */
/*	Read a name from file f into string s, setting the delimiter.	 */
/*									 */
/*	- Embedded periods are permitted, but periods followed by space	 */
/*	  characters act as delimiters.					 */
/*	- Embedded spaces are permitted, but multiple spaces are	 */
/*	  replaced by a single space.					 */
/*	- Any character can be escaped by '\'.				 */
/*	- The remainder of a line following '|' is ignored.		 */
/*									 */
/*	Colons are sometimes delimiters depending on ColonOpt		 */
/*									 */
/*************************************************************************/


Boolean ReadNameInput(c50_context *Context, c50_input *f, String s, int n,
		      char ColonOpt)
/*      -------------  */
{
    register char *Sp=s;
    register int  c;
    char	  Msg[2];

    /*  Skip to first non-space character  */

    while ( (c = InChar(Context, f)) == '|' || Space(c) )
    {
	if ( c == '|' )
	{
	    while ( ( c = InChar(Context, f) ) != '\n' && c != EOF )
		;
	}
    }

    /*  Return false if no names to read  */

    if ( c == EOF )
    {
	Context->delimiter = EOF;
	return false;
    }

    /*  Read in characters up to the next delimiter  */

    while ( c != ColonOpt && c != ',' && c != '\n' && c != '|' && c != EOF )
    {
	if ( --n <= 0 )
	{
	    if ( Context->io.output ) Error(Context, LONGNAME, "", "");
	}

	if ( c == '.' )
	{
	    if ( (c = InChar(Context, f)) == '|' || Space(c) || c == EOF ) break;
	    *Sp++ = '.';
	    continue;
	}

	if ( c == '\\' )
	{
	    c = InChar(Context, f);
	}

	if ( Space(c) )
	{
	    *Sp++ = ' ';

	    while ( ( c = InChar(Context, f) ) == ' ' || c == '\t' )
		;
	}
	else
	{
	    *Sp++ = c;
	    c = InChar(Context, f);
	}
    }

    if ( c == '|' )
    {
	while ( ( c = InChar(Context, f) ) != '\n' && c != EOF )
	    ;
    }
    Context->delimiter = c;

    /*  Special case for ':='  */

    if ( Context->delimiter == ':' )
    {
	if ( *Context->line_buffer_position == '=' )
	{
	    Context->delimiter = '=';
	    Context->line_buffer_position++;
	}
    }

    /*  Strip trailing spaces  */

    while ( Sp > s && Space(*(Sp-1)) ) Sp--;

    if ( Sp == s )
    {
	Msg[0] = ( Space(c) ? '.' : c );
	Msg[1] = '\00';
	Error(Context, MISSNAME, Context->io.file_name, Msg);
    }

    *Sp++ = '\0';
    return true;
}



Boolean ReadName(c50_context *Context, FILE *f, String s, int n,
		 char ColonOpt)
/*      --------  */
{
    c50_input Input;

    c50_input_init_file(&Input, f);
    return ReadNameInput(Context, &Input, s, n, ColonOpt);
}



/*************************************************************************/
/*									 */
/*	Read names of classes, attributes and legal attribute values.	 */
/*	On completion, names are stored in:				 */
/*	  Context->schema.class_names	-	class names				 */
/*	  Context->schema.attribute_names	-	attribute names				 */
/*	  Context->schema.attribute_value_names	-	attribute value names			 */
/*	with:								 */
/*	  Context->schema.max_attribute_value	-	number of values for each attribute	 */
/*									 */
/*	Other global variables set are:					 */
/*	  Context->schema.max_attribute	-	maximum attribute number		 */
/*	  Context->schema.max_class	-	maximum class number			 */
/*	  Context->schema.max_discrete_value	-	maximum discrete values for an attribute */
/*									 */
/*	The caller retains ownership of Nf.				 */
/*									 */
/*************************************************************************/


void GetNames(c50_context *Context, c50_input *Nf)
/*   --------  */
{
    char	Buffer[1000]="", *EndBuff;
    int		AttCeiling=100, ClassCeiling=100;
    Attribute	Att;
    ClassNo	c;

    Context->io.error_count = Context->io.attribute_exclusions = 0;
    Context->io.line_number  = 0;
    Context->line_buffer_position     = Context->line_buffer;
    *Context->line_buffer_position    = 0;

    Context->schema.max_class = Context->schema.class_attribute = Context->schema.label_attribute = Context->schema.case_weight_attribute = 0;

    /*  Get class names from names file.  This entry can be:
	- a list of discrete values separated by commas
	- the name of the discrete attribute to use as the class
	- the name of a continuous attribute followed by a colon and
	  a comma-separated list of thresholds used to segment it  */

    Context->schema.class_names = AllocZero(ClassCeiling, String);
    do
    {
	ReadNameInput(Context, Nf, Buffer, 1000, ':');

	if ( ++Context->schema.max_class >= ClassCeiling)
	{
	    ClassCeiling += 100;
	    Realloc(Context->schema.class_names, ClassCeiling, String);
	}
	Context->schema.class_names[Context->schema.max_class] = strdup(Buffer);
    }
    while ( Context->delimiter == ',' );

    if ( Context->delimiter == ':' )
    {
	/*  Thresholds for continuous class attribute  */

	Context->schema.class_thresholds = Alloc(ClassCeiling, ContValue);
	Context->schema.max_class = 0;

	do
	{
	    ReadNameInput(Context, Nf, Buffer, 1000, ':');

	    if ( ++Context->schema.max_class >= ClassCeiling)
	    {
		ClassCeiling += 100;
		Realloc(Context->schema.class_thresholds, ClassCeiling, ContValue);
	    }

	    Context->schema.class_thresholds[Context->schema.max_class] = strtod(Buffer, &EndBuff);
	    if ( EndBuff == Buffer || *EndBuff != '\0' )
	    {
		Error(Context, BADCLASSTHRESH, Buffer, Nil);
	    }
	    else
	    if ( Context->schema.max_class > 1 &&
		 Context->schema.class_thresholds[Context->schema.max_class] <= Context->schema.class_thresholds[Context->schema.max_class-1] )
	    {
		Error(Context, LEQCLASSTHRESH, Buffer, Nil);
	    }
	}
	while ( Context->delimiter == ',' );
    }

    /*  Get attribute and attribute value names from names file  */

    Context->schema.attribute_names	  = AllocZero(AttCeiling, String);
    Context->schema.max_attribute_value	  = AllocZero(AttCeiling, DiscrValue);
    Context->schema.attribute_value_names	  = AllocZero(AttCeiling, String *);
    Context->schema.special_status = AllocZero(AttCeiling, char);
    Context->schema.attribute_definitions	  = AllocZero(AttCeiling, Definition);
    Context->schema.attribute_definition_uses	  = AllocZero(AttCeiling, Attribute *);

    Context->schema.max_attribute = 0;
    while ( ReadNameInput(Context, Nf, Buffer, 1000, ':') )
    {
	if ( Context->delimiter != ':' && Context->delimiter != '=' )
	{
	    Error(Context, BADATTNAME, Buffer, "");
	}

	/*  Check for attributes included/excluded  */

	if ( ( *Buffer == 'a' || *Buffer == 'A' ) &&
	     ! memcmp(Buffer+1, "ttributes ", 10) &&
	     ! memcmp(Buffer+strlen(Buffer)-6, "cluded", 6) )
	{
	    Context->io.attribute_exclusions = ( ! memcmp(Buffer+strlen(Buffer)-8, "in", 2) ? 1 : -1 );
	    if ( Context->io.attribute_exclusions == 1 )
	    {
		ForEach(Att, 1, Context->schema.max_attribute)
		{
		    Context->schema.special_status[Att] |= SKIP;
		}
	    }

	    while ( ReadNameInput(Context, Nf, Buffer, 1000, ':') )
	    {
		Att = Which(Buffer, Context->schema.attribute_names, 1, Context->schema.max_attribute);
		if ( ! Att )
		{
		    Error(Context, UNKNOWNATT, Buffer, Nil);
		}
		else
		if ( Context->io.attribute_exclusions == 1 )
		{
		    Context->schema.special_status[Att] -= SKIP;
		}
		else
		{
		    Context->schema.special_status[Att] |= SKIP;
		}
	    }

	    break;
	}

	if ( Which(Buffer, Context->schema.attribute_names, 1, Context->schema.max_attribute) > 0 )
	{
	    Error(Context, DUPATTNAME, Buffer, Nil);
	}

	if ( ++Context->schema.max_attribute >= AttCeiling )
	{
	    AttCeiling += 100;
	    Realloc(Context->schema.attribute_names, AttCeiling, String);
	    Realloc(Context->schema.max_attribute_value, AttCeiling, DiscrValue);
	    Realloc(Context->schema.attribute_value_names, AttCeiling, String *);
	    Realloc(Context->schema.special_status, AttCeiling, char);
	    Realloc(Context->schema.attribute_definitions, AttCeiling, Definition);
	    Realloc(Context->schema.attribute_definition_uses, AttCeiling, Attribute *);
	}

	Context->schema.attribute_names[Context->schema.max_attribute]       = strdup(Buffer);
	Context->schema.special_status[Context->schema.max_attribute] = Nil;
	Context->schema.attribute_definitions[Context->schema.max_attribute]        = Nil;
	Context->schema.max_attribute_value[Context->schema.max_attribute]     = 0;
	Context->schema.attribute_definition_uses[Context->schema.max_attribute]    = Nil;

	if ( Context->delimiter == '=' )
	{
	    if ( Context->schema.max_class == 1 && ! strcmp(Context->schema.class_names[1], Context->schema.attribute_names[Context->schema.max_attribute]) )
	    {
		ErrorContext(Context, BADDEF3, Nil, Nil);
	    }

	    ImplicitAtt(Context, Nf);
	    ListAttsUsed(Context);
	}
	else
	{
	    ExplicitAtt(Context, Nf);
	}

	/*  Check for case weight attribute, which must be type continuous  */

	if (  ! strcmp(Context->schema.attribute_names[Context->schema.max_attribute], "case weight") )
	{
	    Context->schema.case_weight_attribute = Context->schema.max_attribute;

	    if ( ! Continuous(Context->schema.case_weight_attribute) )
	    {
		Error(Context, CWTATTERR, "", "");
	    }
	}
    }

    /*  Check whether class is one of the attributes  */

    if ( Context->schema.max_class == 1 || Context->schema.class_thresholds )
    {
	/*  Class attribute must be present and must be either
	    a discrete attribute or a thresholded continuous attribute  */

	Context->schema.class_attribute = Which(Context->schema.class_names[1], Context->schema.attribute_names, 1, Context->schema.max_attribute);

	if ( Context->schema.class_attribute <= 0 || Exclude(Context->schema.class_attribute) )
	{
	    Error(Context, NOTARGET, Context->schema.class_names[1], "");
	}
	else
	if ( Context->schema.class_thresholds &&
	     ( ! Continuous(Context->schema.class_attribute) ||
	       StatBit(Context->schema.class_attribute, DATEVAL|STIMEVAL|TSTMPVAL) ) )
	{
	    Error(Context, BADCTARGET, Context->schema.class_names[1], "");
	}
	else
	if ( ! Context->schema.class_thresholds &&
	     ( Continuous(Context->schema.class_attribute) || StatBit(Context->schema.class_attribute, DISCRETE) ) )
	{
	    Error(Context, BADDTARGET, Context->schema.class_names[1], "");
	}

	Free(Context->schema.class_names[1]);

	if ( ! Context->schema.class_thresholds )
	{
	    Free(Context->schema.class_names);
	    Context->schema.max_class  = Context->schema.max_attribute_value[Context->schema.class_attribute];
	    Context->schema.class_names = Context->schema.attribute_value_names[Context->schema.class_attribute];
	}
	else
	{
	    /*  Set up class names as segments of continuous target att  */

	    Context->schema.max_class++;
	    Realloc(Context->schema.class_names, Context->schema.max_class+1, String);

	    sprintf(Buffer, "%s <= %g", Context->schema.attribute_names[Context->schema.class_attribute], Context->schema.class_thresholds[1]);
	    Context->schema.class_names[1] = strdup(Buffer);

	    ForEach(c, 2, Context->schema.max_class-1)
	    {
		sprintf(Buffer, "%g < %s <= %g",
			Context->schema.class_thresholds[c-1], Context->schema.attribute_names[Context->schema.class_attribute], Context->schema.class_thresholds[c]);
		Context->schema.class_names[c] = strdup(Buffer);
	    }

	    sprintf(Buffer, "%s > %g",
		    Context->schema.attribute_names[Context->schema.class_attribute], Context->schema.class_thresholds[Context->schema.max_class-1]);
	    Context->schema.class_names[Context->schema.max_class] = strdup(Buffer);
	}
    }

    /*  Ignore case weight attribute if it is excluded; otherwise,
	it cannot be used in models  */

    if ( Context->schema.case_weight_attribute )
    {
	if ( Skip(Context->schema.case_weight_attribute) )
	{
	    Context->schema.case_weight_attribute = 0;
	}
	else
	{
	    Context->schema.special_status[Context->schema.case_weight_attribute] |= SKIP;
	}
    }

    Context->schema.class_names[0] = "?";

    if ( Context->io.error_count > 0 ) Goodbye(1);
}



/*************************************************************************/
/*									 */
/*	Continuous or discrete attribute				 */
/*									 */
/*************************************************************************/


void ExplicitAtt(c50_context *Context, c50_input *Nf)
/*   -----------  */
{
    char	Buffer[1000]="", *p;
    DiscrValue	v;
    int		ValCeiling=100, BaseYear;
    time_t	clock;

    /*  Read attribute type or first discrete value  */

    if ( ! ( ReadNameInput(Context, Nf, Buffer, 1000, ':') ) )
    {
	Error(Context, EOFINATT, Context->schema.attribute_names[Context->schema.max_attribute], "");
    }

    Context->schema.max_attribute_value[Context->schema.max_attribute] = 0;

    if ( Context->delimiter != ',' )
    {
	/*  Typed attribute  */

	if ( ! strcmp(Buffer, "continuous") )
	{
	}
	else
	if ( ! strcmp(Buffer, "timestamp") )
	{
	    Context->schema.special_status[Context->schema.max_attribute] = TSTMPVAL;

	    /*  Set the base date if not done already  */

	    if ( ! Context->io.timestamp_base )
	    {
		clock = time(0);
		BaseYear = gmtime(&clock)->tm_year + 1900;
		SetTSBase(Context, BaseYear);
	    }
	}
	else
	if ( ! strcmp(Buffer, "date") )
	{
	    Context->schema.special_status[Context->schema.max_attribute] = DATEVAL;
	}
	else
	if ( ! strcmp(Buffer, "time") )
	{
	    Context->schema.special_status[Context->schema.max_attribute] = STIMEVAL;
	}
	else
	if ( ! memcmp(Buffer, "discrete", 8) )
	{
	    Context->schema.special_status[Context->schema.max_attribute] = DISCRETE;

	    /*  Read max values and reserve space  */

	    v = atoi(&Buffer[8]);
	    if ( v < 2 )
	    {
		Error(Context, BADDISCRETE, Context->schema.attribute_names[Context->schema.max_attribute], "");
	    }

	    Context->schema.attribute_value_names[Context->schema.max_attribute] = Alloc(v+3, String);
	    Context->schema.attribute_value_names[Context->schema.max_attribute][0] = (String) (intptr_t) (v+1);
	    Context->schema.attribute_value_names[Context->schema.max_attribute][(Context->schema.max_attribute_value[Context->schema.max_attribute]=1)] = strdup("N/A");
	}
	else
	if ( ! strcmp(Buffer, "ignore") )
	{
	    Context->schema.special_status[Context->schema.max_attribute] = EXCLUDE;
	}
	else
	if ( ! strcmp(Buffer, "label") )
	{
	    Context->schema.label_attribute = Context->schema.max_attribute;
	    Context->schema.special_status[Context->schema.max_attribute] = EXCLUDE;
	}
	else
	{
	    /*  Cannot have only one discrete value for an attribute  */

	    Error(Context, SINGLEATTVAL, Context->schema.attribute_names[Context->schema.max_attribute], Buffer);
	}
    }
    else
    {
	/*  Discrete attribute with explicit values  */

	Context->schema.attribute_value_names[Context->schema.max_attribute] = AllocZero(ValCeiling, String);

	/*  Add "N/A" unless this attribute is the class  */

	if ( Context->schema.max_class > 1 || strcmp(Context->schema.class_names[1], Context->schema.attribute_names[Context->schema.max_attribute]) )
	{
	    Context->schema.attribute_value_names[Context->schema.max_attribute][(Context->schema.max_attribute_value[Context->schema.max_attribute]=1)] = strdup("N/A");
	}
	else
	{
	    Context->schema.max_attribute_value[Context->schema.max_attribute] = 0;
	}

	p = Buffer;

	/*  Special check for ordered attribute  */

	if ( ! memcmp(Buffer, "[ordered]", 9) )
	{
	    Context->schema.special_status[Context->schema.max_attribute] = ORDERED;

	    for ( p = Buffer+9 ; Space(*p) ; p++ )
		;
	}

	/*  Record first real explicit value  */

	Context->schema.attribute_value_names[Context->schema.max_attribute][++Context->schema.max_attribute_value[Context->schema.max_attribute]] = strdup(p);

	/*  Record remaining values  */

	do
	{
	    if ( ! ( ReadNameInput(Context, Nf, Buffer, 1000, ':') ) )
	    {
		Error(Context, EOFINATT, Context->schema.attribute_names[Context->schema.max_attribute], "");
	    }

	    if ( ++Context->schema.max_attribute_value[Context->schema.max_attribute] >= ValCeiling )
	    {
		ValCeiling += 100;
		Realloc(Context->schema.attribute_value_names[Context->schema.max_attribute], ValCeiling, String);
	    }

	    Context->schema.attribute_value_names[Context->schema.max_attribute][Context->schema.max_attribute_value[Context->schema.max_attribute]] = strdup(Buffer);
	}
	while ( Context->delimiter == ',' );

	/*  Cancel ordered status if <3 real values  */

	if ( Ordered(Context->schema.max_attribute) && Context->schema.max_attribute_value[Context->schema.max_attribute] <= 3 )
	{
	    Context->schema.special_status[Context->schema.max_attribute] = 0;
	}
	if ( Context->schema.max_attribute_value[Context->schema.max_attribute] > Context->schema.max_discrete_value ) Context->schema.max_discrete_value = Context->schema.max_attribute_value[Context->schema.max_attribute];
    }
}



/*************************************************************************/
/*									 */
/*	Locate value Val in List[First] to List[Last]			 */
/*									 */
/*************************************************************************/


int Which(String Val, String *List, int First, int Last)
/*  -----  */
{
    int	n=First;

    while ( n <= Last && strcmp(Val, List[n]) ) n++;

    return ( n <= Last ? n : First-1 );
}



/*************************************************************************/
/*									 */
/*	Build list of attributes used in current attribute definition	 */
/*	    Context->schema.attribute_definition_uses[Att][0] = number of atts used			 */
/*	    Context->schema.attribute_definition_uses[Att][1..] are the atts				 */
/*									 */
/*************************************************************************/


void ListAttsUsed(c50_context *Context)
/*   ------------  */
{
    Attribute	Att;
    Boolean	*DefUses;
    Definition	D;
    int		e, NUsed=0;

    DefUses = AllocZero(Context->schema.max_attribute+1, Boolean);

    D = Context->schema.attribute_definitions[Context->schema.max_attribute];

    for ( e = 0 ; ; e++ )
    {
	if ( DefOp(D[e]) == OP_ATT )
	{
	    Att = (Attribute) (intptr_t) DefSVal(D[e]);
	    if ( ! DefUses[Att] )
	    {
		DefUses[Att] = true;
		NUsed++;
	    }
	}
	else
	if ( DefOp(D[e]) == OP_END )
	{
	    break;
	}
    }

    if ( NUsed )
    {
	Context->schema.attribute_definition_uses[Context->schema.max_attribute] = Alloc(NUsed+1, Attribute);
	Context->schema.attribute_definition_uses[Context->schema.max_attribute][0] = NUsed;

	NUsed=0;
	ForEach(Att, 1, Context->schema.max_attribute-1)
	{
	    if ( DefUses[Att] )
	    {
		Context->schema.attribute_definition_uses[Context->schema.max_attribute][++NUsed] = Att;
	    }
	}
    }

    Free(DefUses);
}



/*************************************************************************/
/*									 */
/*	Free up all space allocated by GetNames()			 */
/*									 */
/*************************************************************************/


void FreeNames(c50_context *Context)
/*   ---------  */
{
    Attribute a, t;

    if ( ! Context->schema.attribute_names ) return;

    ForEach(a, 1, Context->schema.max_attribute)
    {
	if ( Context->schema.attribute_value_names[a] != Context->schema.class_names && Discrete(a) )
	{
	    FreeVector((void **) Context->schema.attribute_value_names[a], 1, Context->schema.max_attribute_value[a]);
	}
    }
    FreeUnlessNil(Context->schema.attribute_value_names);				Context->schema.attribute_value_names = Nil;
    FreeUnlessNil(Context->schema.max_attribute_value);				Context->schema.max_attribute_value = Nil;
    FreeUnlessNil(Context->schema.class_thresholds);				Context->schema.class_thresholds = Nil;
    FreeVector((void **) Context->schema.attribute_names, 1, Context->schema.max_attribute);		Context->schema.attribute_names = Nil;
    FreeVector((void **) Context->schema.class_names, 1, Context->schema.max_class);	Context->schema.class_names = Nil;

    FreeUnlessNil(Context->schema.special_status);			Context->schema.special_status = Nil;

    /*  Definitions (if any)  */

    if ( Context->schema.attribute_definitions )
    {
	ForEach(a, 1, Context->schema.max_attribute)
	{
	    if ( Context->schema.attribute_definitions[a] )
	    {
		for ( t = 0 ; DefOp(Context->schema.attribute_definitions[a][t]) != OP_END ; t++ )
		{
		    if ( DefOp(Context->schema.attribute_definitions[a][t]) == OP_STR )
		    {
			Free(DefSVal(Context->schema.attribute_definitions[a][t]));
		    }
		}

		Free(Context->schema.attribute_definitions[a]);
		Free(Context->schema.attribute_definition_uses[a]);
	    }
	}
	Free(Context->schema.attribute_definitions);					Context->schema.attribute_definitions = Nil;
	Free(Context->schema.attribute_definition_uses);				Context->schema.attribute_definition_uses = Nil;
    }
}



/*************************************************************************/
/*									 */
/*	Read next char keeping track of line numbers			 */
/*									 */
/*************************************************************************/


int InChar(c50_context *Context, c50_input *f)
/*  ------  */
{
    if ( ! *Context->line_buffer_position )
    {
	Context->line_buffer_position = Context->line_buffer;

	if ( ! c50_input_gets(Context->line_buffer, C50_LINE_BUFFER_CAPACITY, f) )
	{
	    Context->line_buffer[0] = '\00';
	    return EOF;
	}

	Context->io.line_number++;
    }
	
    return (int) *Context->line_buffer_position++;
}
