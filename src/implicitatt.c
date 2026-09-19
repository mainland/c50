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
/*	Routines to handle implicitly-defined attributes		 */
/*	------------------------------------------------		 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"
#include <ctype.h>
#include <stdint.h>

typedef struct c50_implicit_state
{
    char *buffer;
    int buffer_size;
    int buffer_position;
    EltRec *type_stack;
    int type_stack_size;
    int type_stack_position;
    int definition_size;
    int definition_position;
    Boolean previous_error;
} c50_implicit_state;

#define FailSyn(Msg) {DefSyntaxError(Context, Msg); return false;}
#define FailSem(Msg) \
    {DefSemanticsError(Context, Fi, Msg, OpCode); return false;}

typedef  union  _xstack_elt
         {
            DiscrValue  _discr_val;
            ContValue   _cont_val;
            String      _string_val;
         }
	 XStackElt;

#define	cval		_cont_val
#define	sval		_string_val
#define	dval		_discr_val



/*************************************************************************/
/*									 */
/*	A definition is handled in two stages:				 */
/*	  - The definition is read (up to a line ending with a period)	 */
/*	    replacing multiple whitespace characters with one space	 */
/*	  - The definition is then read (using a recursive descent	 */
/*	    parser), building up a reverse polish expression		 */
/*	Syntax and semantics errors are flagged				 */
/*									 */
/*************************************************************************/


void ImplicitAtt(c50_context *Context, c50_input *Nf)
/*   -----------  */
{
    c50_implicit_state State = {0};

    Context->implicit_state = &State;

    /*  Get definition as a string in Context->implicit_state->buffer  */

    ReadDefinition(Context, Nf);

    Context->implicit_state->previous_error = false;
    Context->implicit_state->buffer_position = 0;

    /*  Allocate initial stack and attribute definition  */

    Context->implicit_state->type_stack = Alloc(Context->implicit_state->type_stack_size=50, EltRec);
    Context->implicit_state->type_stack_position = 0;

    AttDef[MaxAtt] = Alloc(Context->implicit_state->definition_size = 100, DefElt);
    Context->implicit_state->definition_position = 0;

    /*  Parse Context->implicit_state->buffer as an expression terminated by a period  */

    Expression(Context);
    if ( ! Find(Context, ".") )
    {
	DefSyntaxError(Context, "'.' ending definition");
    }

    /*  Final check -- defined attribute must not be of type String  */

    if ( ! Context->implicit_state->previous_error )
    {
	if ( Context->implicit_state->definition_position == 1 && DefOp(AttDef[MaxAtt][0]) == OP_ATT &&
	     strcmp(AttName[MaxAtt], "case weight") )
	{
	    Error(SAMEATT,
		  AttName[ (Attribute) (intptr_t) DefSVal(AttDef[MaxAtt][0]) ],
		  Nil);
	}

	if ( Context->implicit_state->type_stack[0].Type == 'B' )
	{
	    /*  Defined attributes should never have a value N/A  */

	    MaxAttVal[MaxAtt] = 3;
	    AttValName[MaxAtt] = AllocZero(4, String);
	    AttValName[MaxAtt][1] = strdup("??");
	    AttValName[MaxAtt][2] = strdup("t");
	    AttValName[MaxAtt][3] = strdup("f");
	}
	else
	{
	    MaxAttVal[MaxAtt] = 0;
	}
    }

    if ( Context->implicit_state->previous_error )
    {
	Context->implicit_state->definition_position = 0;
	SpecialStatus[MaxAtt] = EXCLUDE;
    }

    /*  Write a terminating marker  */

    DefOp(AttDef[MaxAtt][Context->implicit_state->definition_position]) = OP_END;

    Free(Context->implicit_state->buffer);
    Free(Context->implicit_state->type_stack);
    Context->implicit_state = NULL;
}



/*************************************************************************/
/*									 */
/*	Read the text of a definition.  Skip comments, collapse		 */
/*	multiple whitespace characters.					 */
/*									 */
/*************************************************************************/


void ReadDefinition(c50_context *Context, c50_input *f)
/*   --------------  */
{
    Boolean	LastWasPeriod=false;
    char	c;

    Context->implicit_state->buffer = Alloc(Context->implicit_state->buffer_size=50, char);
    Context->implicit_state->buffer_position = 0;

    while ( true )
    {
	c = InChar(Context, f);

	if ( c == '|' )
	{
	    while ( ( c = InChar(Context, f) ) != '\n' && c != EOF )
		;
	}

	if ( c == EOF || ( c == '\n' && LastWasPeriod ) )
	{
	    /*  The definition is complete.  Add a period if it's
		not there already and terminate the string  */

	    if ( ! LastWasPeriod ) Append(Context, '.');
	    Append(Context, 0);

	    return;
	}

	if ( Space(c) )
	{
	    Append(Context, ' ');
	}
	else
	if ( c == '\\' )
	{
	    /*  Escaped character -- bypass any special meaning  */

	    Append(Context, InChar(Context, f));
	}
	else
	{
	    LastWasPeriod = ( c == '.' );
	    Append(Context, c);
	}
    }
}



/*************************************************************************/
/*									 */
/*	Append a character to Context->implicit_state->buffer, resizing it if necessary		 */
/*									 */
/*************************************************************************/


void Append(c50_context *Context, char c)
/*   ------  */
{
    if ( c == ' ' && (! Context->implicit_state->buffer_position || Context->implicit_state->buffer[Context->implicit_state->buffer_position-1] == ' ' ) ) return;

    if ( Context->implicit_state->buffer_position >= Context->implicit_state->buffer_size )
    {
	Realloc(Context->implicit_state->buffer, Context->implicit_state->buffer_size += 50, char);
    }

    Context->implicit_state->buffer[Context->implicit_state->buffer_position++] = c;
}



/*************************************************************************/
/*									 */
/*	Recursive descent parser with syntax error checking.		 */
/*	The reverse polish is built up by calls to Dump() and DumpOp(),	 */
/*	which also check for semantic validity.				 */
/*									 */
/*	For possible error messages, each routine also keeps track of	 */
/*	the beginning of the construct that it recognises (in Fi).	 */
/*									 */
/*************************************************************************/


Boolean Expression(c50_context *Context)
/*      ----------  */
{
    int		Fi=Context->implicit_state->buffer_position;

    if ( Context->implicit_state->buffer[Context->implicit_state->buffer_position] == ' ' ) Context->implicit_state->buffer_position++;

    if ( ! Conjunct(Context) ) FailSyn("expression");

    while ( Find(Context, "or") )
    {
	Context->implicit_state->buffer_position += 2;

	if ( ! Conjunct(Context) ) FailSyn("expression");

	DumpOp(Context, OP_OR, Fi);
    }

    return true;
}



Boolean Conjunct(c50_context *Context)
/*      --------  */
{
    int		Fi=Context->implicit_state->buffer_position;

    if ( ! SExpression(Context) ) FailSyn("expression");

    while ( Find(Context, "and") )
    {
	Context->implicit_state->buffer_position += 3;

	if ( ! SExpression(Context) ) FailSyn("expression");

	DumpOp(Context, OP_AND, Fi);
    }

    return true;
}



static const char RelOps[] = ">=\0<=\0!=\0<>\0>\0<\0=\0";

Boolean SExpression(c50_context *Context)
/*      -----------  */
{
    int		o, Fi=Context->implicit_state->buffer_position;

    if ( ! AExpression(Context) ) FailSyn("expression");

    if ( (o = FindOne(Context, RelOps)) >= 0 )
    {
	Context->implicit_state->buffer_position += ( o < 4 ? 2 : 1 );

	if ( ! AExpression(Context) ) FailSyn("expression");

	DumpOp(Context, ( o == 0 ? OP_GE :
		 o == 1 ? OP_LE :
		 o == 4 ? OP_GT :
		 o == 5 ? OP_LT :
		 o == 2 || o == 3 ?
			( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type == 'S' ? OP_SNE : OP_NE ) :
			( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type == 'S' ? OP_SEQ : OP_EQ ) ), Fi);
    }

    return true;
}



static const char AddOps[] = "+\0-\0";

Boolean AExpression(c50_context *Context)
/*      -----------  */
{
    int		o, Fi=Context->implicit_state->buffer_position;

    if ( Context->implicit_state->buffer[Context->implicit_state->buffer_position] == ' ' ) Context->implicit_state->buffer_position++;

    if ( (o = FindOne(Context, AddOps)) >= 0 )
    {
	Context->implicit_state->buffer_position += 1;
    }

    if ( ! Term(Context) ) FailSyn("expression");

    if ( o == 1 ) DumpOp(Context, OP_UMINUS, Fi);

    while ( (o = FindOne(Context, AddOps)) >= 0 )
    {
	Context->implicit_state->buffer_position += 1;

	if ( ! Term(Context) ) FailSyn("arithmetic expression");

	DumpOp(Context, (char)(OP_PLUS + o), Fi);
    }

    return true;
}



static const char MultOps[] = "*\0/\0%\0";

Boolean Term(c50_context *Context)
/*      ----  */
{
    int		o, Fi=Context->implicit_state->buffer_position;

    if ( ! Factor(Context) ) FailSyn("expression");

    while ( (o = FindOne(Context, MultOps)) >= 0 )
    {
	Context->implicit_state->buffer_position += 1;

	if ( ! Factor(Context) ) FailSyn("arithmetic expression");

	DumpOp(Context, (char)(OP_MULT + o), Fi);
    }

    return true;
}



Boolean Factor(c50_context *Context)
/*      ----  */
{
    int		Fi=Context->implicit_state->buffer_position;

    if ( ! Primary(Context) ) FailSyn("value");

    while ( Find(Context, "^") )
    {
	Context->implicit_state->buffer_position += 1;

	if ( ! Primary(Context) ) FailSyn("exponent");

	DumpOp(Context, OP_POW, Fi);
    }

    return true;
}



Boolean Primary(c50_context *Context)
/*      -------  */
{
    if ( Atom(Context) )
    {
	return true;
    }
    else
    if ( Find(Context, "(") )
    {
	Context->implicit_state->buffer_position++;
	if ( ! Expression(Context) ) FailSyn("expression in parentheses");
	if ( ! Find(Context, ")") ) FailSyn("')'");
	Context->implicit_state->buffer_position++;
	return true;
    }
    else
    {
	FailSyn("attribute, value, or '('");
    }
}



static const char Funcs[] = "sin\0cos\0tan\0log\0exp\0int\0";

Boolean Atom(c50_context *Context)
/*      ----  */
{
    char	*EndPtr, *Str, Date[11], Time[9];
    int		o, FirstBN, Fi=Context->implicit_state->buffer_position;
    ContValue	F;
    Attribute	Att;

    if ( Context->implicit_state->buffer[Context->implicit_state->buffer_position] == ' ' ) Context->implicit_state->buffer_position++;

    if ( Context->implicit_state->buffer[Context->implicit_state->buffer_position] == '"' )
    {
	FirstBN = ++Context->implicit_state->buffer_position;
	while ( Context->implicit_state->buffer[Context->implicit_state->buffer_position] != '"' )
	{
	    if ( ! Context->implicit_state->buffer[Context->implicit_state->buffer_position] ) FailSyn("closing '\"'");
	    Context->implicit_state->buffer_position++;
	}

	/*  Make a copy of the string without double quotes  */

	Context->implicit_state->buffer[Context->implicit_state->buffer_position] = '\00';
	Str = strdup(Context->implicit_state->buffer + FirstBN);

	Context->implicit_state->buffer[Context->implicit_state->buffer_position++] = '"';
	Dump(Context, OP_STR, 0, Str, Fi);
    }
    else
    if ( (Att = FindAttName(Context)) )
    {
	Context->implicit_state->buffer_position += strlen(AttName[Att]);

	Dump(Context, OP_ATT, 0, (String) (intptr_t) Att, Fi);
    }
    else
    if ( isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position]) )
    {
	/*  Check for date or time first  */

	if ( ( ( Context->implicit_state->buffer[Context->implicit_state->buffer_position+4] == '/' && Context->implicit_state->buffer[Context->implicit_state->buffer_position+7] == '/' ) ||
	       ( Context->implicit_state->buffer[Context->implicit_state->buffer_position+4] == '-' && Context->implicit_state->buffer[Context->implicit_state->buffer_position+7] == '-' ) ) &&
	     isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+1]) && isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+2]) &&
		isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+3]) &&
	     isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+5]) && isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+6]) &&
	     isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+8]) && isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+9]) )
	{
	    memcpy(Date, Context->implicit_state->buffer+Context->implicit_state->buffer_position, 10);
	    Date[10] = '\00';
	    if ( (F = DateToDay(Date)) == 0 )
	    {
		Error(BADDEF1, Date, "date");
	    }

	    Context->implicit_state->buffer_position += 10;
	}
	else
	if ( Context->implicit_state->buffer[Context->implicit_state->buffer_position+2] == ':' && Context->implicit_state->buffer[Context->implicit_state->buffer_position+5] == ':' &&
	     isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+1]) &&
	     isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+3]) && isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+4]) &&
	     isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+6]) && isdigit(Context->implicit_state->buffer[Context->implicit_state->buffer_position+7]) )
	{
	    memcpy(Time, Context->implicit_state->buffer+Context->implicit_state->buffer_position, 8);
	    Time[8] = '\00';
	    if ( (F = TimeToSecs(Time)) == 0 )
	    {
		Error(BADDEF1, Time, "time");
	    }

	    Context->implicit_state->buffer_position += 8;
	}
	else
	{
	    F = strtod(Context->implicit_state->buffer+Context->implicit_state->buffer_position, &EndPtr);

	    /*  Check for period after integer  */

	    if ( EndPtr > Context->implicit_state->buffer+Context->implicit_state->buffer_position+1 && *(EndPtr-1) == '.' )
	    {
		EndPtr--;
	    }

	    Context->implicit_state->buffer_position = EndPtr - Context->implicit_state->buffer;
	}

	Dump(Context, OP_NUM, F, Nil, Fi);
    }
    else
    if ( (o = FindOne(Context, Funcs)) >= 0 )
    {
	Context->implicit_state->buffer_position += 3;

	if ( ! Find(Context, "(") ) FailSyn("'(' after function name");
	Context->implicit_state->buffer_position++;

	if ( ! Expression(Context) ) FailSyn("expression");

	if ( ! Find(Context, ")") ) FailSyn("')' after function argument");
	Context->implicit_state->buffer_position++;

	DumpOp(Context, (char)(OP_SIN + o), Fi);
    }
    else
    if ( Context->implicit_state->buffer[Context->implicit_state->buffer_position] == '?' )
    {
	Context->implicit_state->buffer_position++;
	if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type == 'N' )
	{
	    Dump(Context, OP_NUM, UNKNOWN, Nil, Fi);
	}
	else
	{
	    Dump(Context, OP_STR, 0, Nil, Fi);
	}
    }
    else
    if ( ! memcmp(Context->implicit_state->buffer+Context->implicit_state->buffer_position, "N/A", 3) )
    {
	Context->implicit_state->buffer_position += 3;
	if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type == 'N' )
	{
	    Dump(Context, OP_NUM, NA, Nil, Fi);
	}
	else
	{
	    Dump(Context, OP_STR, 0, strdup("N/A"), Fi);
	}
    }
    else
    {
	return false;
    }

    return true;
}



/*************************************************************************/
/*									 */
/*	Skip spaces and check for specific string			 */
/*									 */
/*************************************************************************/


Boolean Find(c50_context *Context, const char *S)
/*      ----  */
{
    if ( Context->implicit_state->buffer[Context->implicit_state->buffer_position] == ' ' ) Context->implicit_state->buffer_position++;

    return ( ! Context->implicit_state->buffer[Context->implicit_state->buffer_position] ? false : ! memcmp(Context->implicit_state->buffer+Context->implicit_state->buffer_position, S, strlen(S)) );
}



/*************************************************************************/
/*									 */
/*	Find one of a zero-terminated list of alternatives		 */
/*									 */
/*************************************************************************/


int FindOne(c50_context *Context, const char *Alt)
/*  -------  */
{
    int		a;
    const char	*S;

    for ( a = 0, S = Alt ; *S ; a++, S += strlen(S) + 1 )
    {
	if ( Find(Context, S) ) return a;
    }

    return -1;
}



/*************************************************************************/
/*									 */
/*	Find an attribute name						 */
/*									 */
/*************************************************************************/


Attribute FindAttName(c50_context *Context)
/*        -----------  */
{
    Attribute	Att, LongestAtt=0;

    ForEach(Att, 1, MaxAtt-1)
    {
	if ( ! Exclude(Att) && Find(Context, AttName[Att]) )
	{
	    if ( ! LongestAtt ||
		 strlen(AttName[Att]) > strlen(AttName[LongestAtt]) )
	    {
		LongestAtt = Att;
	    }
	}
    }

    if ( LongestAtt && ( MaxClass == 1 || ClassThresh ) &&
	 ! strcmp(ClassName[1], AttName[LongestAtt]) )
    {
	Error(BADDEF4, Nil, Nil);
    }

    return LongestAtt;
}



/*************************************************************************/
/*									 */
/*	Error message routines.  Syntax errors come from the		 */
/*	recursive descent parser, semantics errors from the routines	 */
/*	that build up the equivalent polish				 */
/*									 */
/*************************************************************************/


void DefSyntaxError(c50_context *Context, String Msg)
/*   --------------  */
{
    String	RestOfText;
    int		i=10;

    if ( ! Context->implicit_state->previous_error )
    {
	RestOfText = Context->implicit_state->buffer + Context->implicit_state->buffer_position;

	/*  Abbreviate text if longer than 12 characters  */

	if ( CharWidth(RestOfText) > 12 )
	{
#ifdef UTF8
	    /*  Find beginning of UTF-8 character  */

	    for ( ; (RestOfText[i] & 0x80) ; i++)
		;
#endif
	    RestOfText[i] = RestOfText[i+1] = '.';
	}

	Error(BADDEF1, RestOfText, Msg);
	Context->implicit_state->previous_error = true;
    }
}



void DefSemanticsError(c50_context *Context, int Fi, String Msg, int OpCode)
/*   -----------------  */
{
    char	Exp[1000], XMsg[1008], Op[1000];

    if ( ! Context->implicit_state->previous_error )
    {
	/*  Abbreviate the input if necessary  */

	if ( Context->implicit_state->buffer_position - Fi > 23 )
	{
	    snprintf(Exp, sizeof(Exp), "%.10s...%.10s",
		     Context->implicit_state->buffer+Fi, Context->implicit_state->buffer+Context->implicit_state->buffer_position-10);
	}
	else
	{
	    snprintf(Exp, sizeof(Exp), "%.*s", Context->implicit_state->buffer_position - Fi, Context->implicit_state->buffer+Fi);
	}

	switch ( OpCode )
	{
	    case OP_AND:	snprintf(Op, sizeof(Op), "%s", "and"); break;
	    case OP_OR:		snprintf(Op, sizeof(Op), "%s", "or"); break;
	    case OP_SEQ:
	    case OP_EQ:		snprintf(Op, sizeof(Op), "%s", "="); break;
	    case OP_SNE:
	    case OP_NE:		snprintf(Op, sizeof(Op), "%s", "<>"); break;
	    case OP_GT:		snprintf(Op, sizeof(Op), "%s", ">"); break;
	    case OP_GE:		snprintf(Op, sizeof(Op), "%s", ">="); break;
	    case OP_LT:		snprintf(Op, sizeof(Op), "%s", "<"); break;
	    case OP_LE:		snprintf(Op, sizeof(Op), "%s", "<="); break;
	    case OP_PLUS:	snprintf(Op, sizeof(Op), "%s", "+"); break;
	    case OP_MINUS:	snprintf(Op, sizeof(Op), "%s", "-"); break;
	    case OP_UMINUS:	snprintf(Op, sizeof(Op), "%s", "unary -"); break;
	    case OP_MULT:	snprintf(Op, sizeof(Op), "%s", "*"); break;
	    case OP_DIV:	snprintf(Op, sizeof(Op), "%s", "/"); break;
	    case OP_MOD:	snprintf(Op, sizeof(Op), "%s", "%"); break;
	    case OP_POW:	snprintf(Op, sizeof(Op), "%s", "^"); break;
	    case OP_SIN:	snprintf(Op, sizeof(Op), "%s", "sin"); break;
	    case OP_COS:	snprintf(Op, sizeof(Op), "%s", "cos"); break;
	    case OP_TAN:	snprintf(Op, sizeof(Op), "%s", "tan"); break;
	    case OP_LOG:	snprintf(Op, sizeof(Op), "%s", "log"); break;
	    case OP_EXP:	snprintf(Op, sizeof(Op), "%s", "exp"); break;
	    case OP_INT:	snprintf(Op, sizeof(Op), "%s", "int");
	}

	snprintf(XMsg, sizeof(XMsg), "%s with '%s'", Msg, Op);
	Error(BADDEF2, Exp, XMsg);
	Context->implicit_state->previous_error = true;
    }
}



/*************************************************************************/
/*									 */
/*	Reverse polish routines.  These use a model of the stack	 */
/*	during expression evaluation to detect type conflicts etc	 */
/*									 */
/*************************************************************************/



void Dump(c50_context *Context, char OpCode, ContValue F, String S, int Fi)
/*   ----  */
{
    if ( Context->implicit_state->buffer[Fi] == ' ' ) Fi++;

    if ( ! UpdateTStack(Context, OpCode, F, S, Fi) ) return;

    /*  Make sure enough room for this element  */

    if ( Context->implicit_state->definition_position >= Context->implicit_state->definition_size-1 )
    {
	Realloc(AttDef[MaxAtt], Context->implicit_state->definition_size += 100, DefElt);
    }

    DefOp(AttDef[MaxAtt][Context->implicit_state->definition_position]) = OpCode;
    if ( OpCode == OP_ATT || OpCode == OP_STR )
    {
	DefSVal(AttDef[MaxAtt][Context->implicit_state->definition_position]) = S;
    }
    else
    {
	DefNVal(AttDef[MaxAtt][Context->implicit_state->definition_position]) = F;
    }

    Context->implicit_state->definition_position++;
}



void DumpOp(c50_context *Context, char OpCode, int Fi)
/*   ------  */
{
    Dump(Context, OpCode, 0, Nil, Fi);
}



Boolean UpdateTStack(c50_context *Context, char OpCode, ContValue F, String S,
		     int Fi)
/*      ------------  */
{
    (void) F;

    if ( Context->implicit_state->type_stack_position >= Context->implicit_state->type_stack_size )
    {
	Realloc(Context->implicit_state->type_stack, Context->implicit_state->type_stack_size += 50, EltRec);
    }

    switch ( OpCode )
    {
	case OP_ATT:
		Context->implicit_state->type_stack[Context->implicit_state->type_stack_position].Type =
		    ( Continuous((Attribute) (intptr_t) S) ? 'N' : 'S' );
		break;

	case OP_NUM:
		Context->implicit_state->type_stack[Context->implicit_state->type_stack_position].Type = 'N';
		break;

	case OP_STR:
		Context->implicit_state->type_stack[Context->implicit_state->type_stack_position].Type = 'S';
		break;

	case OP_AND:
	case OP_OR:
		if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-2].Type != 'B' || Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type != 'B' )
		{
		    FailSem("non-logical value");
		}
		Context->implicit_state->type_stack_position -= 2;
		break;

	case OP_EQ:
	case OP_NE:
		if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-2].Type != Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type )
		{
		    FailSem("incompatible values");
		}
		Context->implicit_state->type_stack_position -= 2;
		Context->implicit_state->type_stack[Context->implicit_state->type_stack_position].Type = 'B';
		break;

	case OP_GT:
	case OP_GE:
	case OP_LT:
	case OP_LE:
		if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-2].Type != 'N' || Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type != 'N' )
		{
		    FailSem("non-arithmetic value");
		}
		Context->implicit_state->type_stack_position -= 2;
		Context->implicit_state->type_stack[Context->implicit_state->type_stack_position].Type = 'B';
		break;

	case OP_SEQ:
	case OP_SNE:
		if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-2].Type != 'S' || Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type != 'S' )
		{
		    FailSem("incompatible values");
		}
		Context->implicit_state->type_stack_position -= 2;
		Context->implicit_state->type_stack[Context->implicit_state->type_stack_position].Type = 'B';
		break;

	case OP_PLUS:
	case OP_MINUS:
	case OP_MULT:
	case OP_DIV:
	case OP_MOD:
	case OP_POW:
		if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-2].Type != 'N' || Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type != 'N' )
		{
		    FailSem("non-arithmetic value");
		}
		Context->implicit_state->type_stack_position -= 2;
		break;

	case OP_UMINUS:
		if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type != 'N' )
		{
		    FailSem("non-arithmetic value");
		}
		Context->implicit_state->type_stack_position--;
		break;

	case OP_SIN:
	case OP_COS:
	case OP_TAN:
	case OP_LOG:
	case OP_EXP:
	case OP_INT:
		if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type != 'N' )
		{
		    FailSem("non-arithmetic argument");
		}
		Context->implicit_state->type_stack_position--;
    }

    Context->implicit_state->type_stack[Context->implicit_state->type_stack_position].Fi = Fi;
    Context->implicit_state->type_stack[Context->implicit_state->type_stack_position].Li = Context->implicit_state->buffer_position-1;
    Context->implicit_state->type_stack_position++;

    return true;
}



/*************************************************************************/
/*									 */
/*	Evaluate an implicit attribute for a case			 */
/*									 */
/*************************************************************************/

#define	CUnknownVal(AV)		(AV.cval==UNKNOWN)
#define	DUnknownVal(AV)		(AV.dval==UNKNOWN)
#define DUNA(a)	(DUnknownVal(XStack[a]) || NotApplicVal(XStack[a]))
#define CUNA(a)	(CUnknownVal(XStack[a]) || NotApplicVal(XStack[a]))
#define	C1(x)	(CUNA(XSN-1) ? UNKNOWN : (x))
#define	C2(x)	(CUNA(XSN-1) || CUNA(XSN-2) ? UNKNOWN : (x))
#define	CD2(x)	(CUNA(XSN-1) || CUNA(XSN-2) ? UNKNOWN : (x))
#define	D2(x)	(DUNA(XSN-1) || DUNA(XSN-2) ? UNKNOWN : (x))


AttValue EvaluateDef(Definition D, DataRec Case)
/*       -----------  */
{
    XStackElt	XStack[100];			/* allows 100-level nesting  */
    int		XSN=0, DN, bv1, bv2, Mult;
    double	cv1, cv2;
    String	sv1, sv2;
    Attribute	Att;
    DefElt	DElt;
    AttValue	ReturnVal;

    for ( DN = 0 ; ; DN++)
    {
	switch ( DefOp((DElt = D[DN])) )
	{
	    case OP_ATT:
		    Att = (Attribute) (intptr_t) DefSVal(DElt);

		    if ( Continuous(Att) )
		    {
			XStack[XSN++].cval = CVal(Case, Att);
		    }
		    else
		    {
			XStack[XSN++].sval =
			    ( Unknown(Case, Att) && ! NotApplic(Case, Att) ? 0 :
			      AttValName[Att][XDVal(Case, Att)] );
		    }
		    break;

	    case OP_NUM:
		    XStack[XSN++].cval = DefNVal(DElt);
		    break;

	    case OP_STR:
		    XStack[XSN++].sval = DefSVal(DElt);
		    break;

	    case OP_AND:
		    bv1 = XStack[XSN-2].dval;
		    bv2 = XStack[XSN-1].dval;
		    XStack[XSN-2].dval = ( bv1 == 3 || bv2 == 3 ? 3 :
					   D2(bv1 == 2 && bv2 == 2 ? 2 : 3) );
		    XSN--;
		    break;

	    case OP_OR:
		    bv1 = XStack[XSN-2].dval;
		    bv2 = XStack[XSN-1].dval;
		    XStack[XSN-2].dval = ( bv1 == 2 || bv2 == 2 ? 2 :
					   D2(bv1 == 2 || bv2 == 2 ? 2 : 3) );
		    XSN--;
		    break;

	    case OP_EQ:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = ( cv1 == cv2 ? 2 : 3 );
		    XSN--;
		    break;

	    case OP_NE:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = ( cv1 != cv2 ? 2 : 3 );
		    XSN--;
		    break;

	    case OP_GT:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = CD2(cv1 > cv2 ? 2 : 3);
		    XSN--;
		    break;

	    case OP_GE:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = CD2(cv1 >= cv2 ? 2 : 3);
		    XSN--;
		    break;

	    case OP_LT:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = CD2(cv1 < cv2 ? 2 : 3);
		    XSN--;
		    break;

	    case OP_LE:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = CD2(cv1 <= cv2 ? 2 : 3);
		    XSN--;
		    break;

	    case OP_SEQ:
		    sv1 = XStack[XSN-2].sval;
		    sv2 = XStack[XSN-1].sval;
		    XStack[XSN-2].dval =
			( ! sv1 && ! sv2 ? 2 :
			  ! sv1 || ! sv2 ? 3 :
			  ! strcmp(sv1, sv2) ? 2 : 3 );
		    XSN--;
		    break;

	    case OP_SNE:
		    sv1 = XStack[XSN-2].sval;
		    sv2 = XStack[XSN-1].sval;
		    XStack[XSN-2].dval =
			( ! sv1 && ! sv2 ? 3 :
			  ! sv1 || ! sv2 ? 2 :
			  strcmp(sv1, sv2) ? 2 : 3 );
		    XSN--;
		    break;

	    case OP_PLUS:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval = C2(cv1 + cv2);
		    XSN--;
		    break;

	    case OP_MINUS:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval = C2(cv1 - cv2);
		    XSN--;
		    break;

	    case OP_MULT:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval = C2(cv1 * cv2);
		    XSN--;
		    break;

	    case OP_DIV:
		    /*  Note: have to set precision of result  */

		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    if ( ! cv2 ||
			 CUnknownVal(XStack[XSN-2]) ||
			 CUnknownVal(XStack[XSN-1]) ||
			 NotApplicVal(XStack[XSN-2]) ||
			 NotApplicVal(XStack[XSN-1]) )
		    {
			XStack[XSN-2].cval = UNKNOWN;
		    }
		    else
		    {
			Mult = Denominator(cv1);
			cv1 = cv1 / cv2;
			while ( fabs(cv2) > 1 )
			{
			    Mult *= 10;
			    cv2 /= 10;
			}
			XStack[XSN-2].cval = rint(cv1 * Mult) / Mult;
		    }
		    XSN--;
		    break;

	    case OP_MOD:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval = C2(fmod(cv1, cv2));
		    XSN--;
		    break;

	    case OP_POW:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval =
			( CUNA(XSN-1) || CUNA(XSN-2) ||
			  ( cv1 < 0 && ceil(cv2) != cv2 ) ? UNKNOWN :
			  pow(cv1, cv2) );
		    XSN--;
		    break;

	    case OP_UMINUS:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(-cv1);
		    break;

	    case OP_SIN:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(sin(cv1));
		    break;

	    case OP_COS:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(cos(cv1));
		    break;

	    case OP_TAN:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(tan(cv1));
		    break;

	    case OP_LOG:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval =
			( CUNA(XSN-1) || cv1 <= 0 ? UNKNOWN : log(cv1) );
		    break;

	    case OP_EXP:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(exp(cv1));
		    break;

	    case OP_INT:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(rint(cv1));
		    break;

	    case OP_END:
		    ReturnVal.cval = XStack[0].cval;	/* cval >= dval bytes */
		    return ReturnVal;
	}
    }
}
