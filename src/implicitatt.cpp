/* Modified 2026 by Geoffrey Mainland: validate implicit expression bounds and ownership. */
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
#include <vector>
#include <stdint.h>
#include <memory>

struct c50_implicit_state
{
    explicit c50_implicit_state(c50_context *owner) : context(owner)
    {
        context->implicit_state = this;
    }
    ~c50_implicit_state()
    {
        context->implicit_state = nullptr;
    }
    c50_implicit_state(const c50_implicit_state &) = delete;
    c50_implicit_state &operator=(const c50_implicit_state &) = delete;

    c50_context *context;
    std::vector<char> buffer;
    int buffer_length = 0;
    int buffer_position = 0;
    std::vector<EltRec> type_stack;
    int type_stack_position = 0;
    int definition_size = 0;
    int definition_position = 0;
    Boolean previous_error = false;
    int expression_depth = 0;
};

struct c50_expression_depth
{
    explicit c50_expression_depth(c50_implicit_state *owner) : state(owner)
    {
        ++state->expression_depth;
    }
    ~c50_expression_depth()
    {
        --state->expression_depth;
    }
    c50_expression_depth(const c50_expression_depth &) = delete;
    c50_expression_depth &operator=(const c50_expression_depth &) = delete;

    c50_implicit_state *state;
};

static constexpr int MaxExpressionDepth = 100;

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
    c50_implicit_state State(Context);

    /*  Get definition as a string in Context->implicit_state->buffer  */

    ReadDefinition(Context, Nf);

    Context->implicit_state->previous_error = false;
    Context->implicit_state->buffer_position = 0;

    /*  Allocate initial stack and attribute definition  */

    Context->implicit_state->type_stack.reserve(50);
    Context->implicit_state->type_stack_position = 0;

    Context->schema.attribute_definitions[Context->schema.max_attribute] = Alloc(Context->implicit_state->definition_size = 100, DefElt);
    Context->implicit_state->definition_position = 0;
    DefOp(Context->schema.attribute_definitions[Context->schema.max_attribute][0]) = OP_END;

    /*  Parse Context->implicit_state->buffer as an expression terminated by a period  */

    Expression(Context);
    if ( ! Find(Context, ".") )
    {
	DefSyntaxError(Context, "'.' ending definition");
    }

    /*  Final check -- defined attribute must not be of type String  */

    if ( ! Context->implicit_state->previous_error )
    {
	if ( Context->implicit_state->type_stack_position != 1 )
	{
	    throw c50::exception(c50::error_code::internal_error,
	                         "invalid implicit expression type stack");
	}
	if ( Context->implicit_state->definition_position == 1 && DefOp(Context->schema.attribute_definitions[Context->schema.max_attribute][0]) == OP_ATT &&
	     strcmp(Context->schema.attribute_names[Context->schema.max_attribute], "case weight") )
	{
	    ErrorContext(Context, SAMEATT,
		  Context->schema.attribute_names[ (Attribute) (intptr_t) DefSVal(Context->schema.attribute_definitions[Context->schema.max_attribute][0]) ],
		  Nil);
	}

	if ( Context->implicit_state->type_stack[0].Type == 'B' )
	{
	    /*  Defined attributes should never have a value N/A  */

	    Context->schema.max_attribute_value[Context->schema.max_attribute] = 3;
	    Context->schema.attribute_value_names[Context->schema.max_attribute] = AllocZero(4, String);
	    Context->schema.attribute_value_names[Context->schema.max_attribute][1] = Pstrdup(Context, "??");
	    Context->schema.attribute_value_names[Context->schema.max_attribute][2] = Pstrdup(Context, "t");
	    Context->schema.attribute_value_names[Context->schema.max_attribute][3] = Pstrdup(Context, "f");
	}
	else
	{
	    Context->schema.max_attribute_value[Context->schema.max_attribute] = 0;
	}
    }

    if ( Context->implicit_state->previous_error )
    {
        for ( int index = 0; index < State.definition_position; ++index )
        {
            auto &element = Context->schema.attribute_definitions[Context->schema.max_attribute][index];
            if ( DefOp(element) == OP_STR ) free(DefSVal(element));
        }
	Context->implicit_state->definition_position = 0;
	Context->schema.special_status[Context->schema.max_attribute] = EXCLUDE;
    }

    /*  Write a terminating marker  */

    DefOp(Context->schema.attribute_definitions[Context->schema.max_attribute][Context->implicit_state->definition_position]) = OP_END;
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
    int		c;

    Context->implicit_state->buffer.clear();
    Context->implicit_state->buffer.reserve(50);
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
	    Context->implicit_state->buffer_length =
		Context->implicit_state->buffer_position - 1;

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

	    c = InChar(Context, f);
	    if ( c == EOF )
	    {
		Append(Context, '.');
		Append(Context, 0);
		Context->implicit_state->buffer_length =
		    Context->implicit_state->buffer_position - 1;
		return;
	    }
	    Append(Context, static_cast<char>(c));
	}
	else
	{
	    LastWasPeriod = ( c == '.' );
	    Append(Context, static_cast<char>(c));
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

    Context->implicit_state->buffer.push_back(c);
    Context->implicit_state->buffer_position++;
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

    if ( Context->implicit_state->expression_depth >= MaxExpressionDepth )
        FailSyn("expression nested at most 100 levels");
    c50_expression_depth Depth(Context->implicit_state);

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
    int		o, FirstBN, Remaining, Fi=Context->implicit_state->buffer_position;
    ContValue	F;
    Attribute	Att;

    if ( Context->implicit_state->buffer_position < Context->implicit_state->buffer_length &&
	 Context->implicit_state->buffer[Context->implicit_state->buffer_position] == ' ' ) Context->implicit_state->buffer_position++;
    if ( Context->implicit_state->buffer_position >= Context->implicit_state->buffer_length ) return false;
    Remaining = Context->implicit_state->buffer_length -
		Context->implicit_state->buffer_position;

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
	Str = Pstrdup(Context, Context->implicit_state->buffer.data() + FirstBN);

	Context->implicit_state->buffer[Context->implicit_state->buffer_position++] = '"';
	Dump(Context, OP_STR, 0, Str, Fi);
    }
    else
    if ( (Att = FindAttName(Context)) )
    {
	Context->implicit_state->buffer_position += strlen(Context->schema.attribute_names[Att]);

	Dump(Context, OP_ATT, 0, (String) (intptr_t) Att, Fi);
    }
    else
    if ( isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position]) )
    {
	/*  Check for date or time first  */

	if ( Remaining >= 10 &&
	     ( ( Context->implicit_state->buffer[Context->implicit_state->buffer_position+4] == '/' && Context->implicit_state->buffer[Context->implicit_state->buffer_position+7] == '/' ) ||
	       ( Context->implicit_state->buffer[Context->implicit_state->buffer_position+4] == '-' && Context->implicit_state->buffer[Context->implicit_state->buffer_position+7] == '-' ) ) &&
	     isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+1]) && isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+2]) &&
		isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+3]) &&
	     isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+5]) && isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+6]) &&
	     isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+8]) && isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+9]) )
	{
	    memcpy(Date, Context->implicit_state->buffer.data() + Context->implicit_state->buffer_position, 10);
	    Date[10] = '\00';
	    if ( (F = DateToDay(Date)) == 0 )
	    {
		ErrorContext(Context, BADDEF1, Date, "date");
	    }

	    Context->implicit_state->buffer_position += 10;
	}
	else
	if ( Remaining >= 8 &&
	     Context->implicit_state->buffer[Context->implicit_state->buffer_position+2] == ':' && Context->implicit_state->buffer[Context->implicit_state->buffer_position+5] == ':' &&
	     isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+1]) &&
	     isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+3]) && isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+4]) &&
	     isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+6]) && isdigit((unsigned char) Context->implicit_state->buffer[Context->implicit_state->buffer_position+7]) )
	{
	    memcpy(Time, Context->implicit_state->buffer.data() + Context->implicit_state->buffer_position, 8);
	    Time[8] = '\00';
	    if ( (F = TimeToSecs(Time)) == 0 )
	    {
		ErrorContext(Context, BADDEF1, Time, "time");
	    }

	    Context->implicit_state->buffer_position += 8;
	}
	else
	{
	    F = strtod(Context->implicit_state->buffer.data() + Context->implicit_state->buffer_position, &EndPtr);
	    if ( ! isfinite(F) ) FailSyn("finite number");

	    /*  Check for period after integer  */

	    if ( EndPtr > Context->implicit_state->buffer.data() + Context->implicit_state->buffer_position+1 && *(EndPtr-1) == '.' )
	    {
		EndPtr--;
	    }

	    Context->implicit_state->buffer_position = EndPtr - Context->implicit_state->buffer.data();
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
	if ( ! Context->implicit_state->type_stack_position ) return false;
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
    if ( Find(Context, "N/A") )
    {
	if ( ! Context->implicit_state->type_stack_position ) return false;
	Context->implicit_state->buffer_position += 3;
	if ( Context->implicit_state->type_stack[Context->implicit_state->type_stack_position-1].Type == 'N' )
	{
	    Dump(Context, OP_NUM, NA, Nil, Fi);
	}
	else
	{
	    Dump(Context, OP_STR, 0, Pstrdup(Context, "N/A"), Fi);
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
    if ( Context->implicit_state->buffer_position < Context->implicit_state->buffer_length &&
	 Context->implicit_state->buffer[Context->implicit_state->buffer_position] == ' ' ) Context->implicit_state->buffer_position++;

    const int Remaining = Context->implicit_state->buffer_length -
			  Context->implicit_state->buffer_position;
    const size_t Length = strlen(S);
    return ( Remaining > 0 && Length <= static_cast<size_t>(Remaining) &&
	     ! memcmp(Context->implicit_state->buffer.data() + Context->implicit_state->buffer_position, S, Length) );
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

    ForEach(Att, 1, Context->schema.max_attribute-1)
    {
	if ( ! Exclude(Att) && Find(Context, Context->schema.attribute_names[Att]) )
	{
	    if ( ! LongestAtt ||
		 strlen(Context->schema.attribute_names[Att]) > strlen(Context->schema.attribute_names[LongestAtt]) )
	    {
		LongestAtt = Att;
	    }
	}
    }

    if ( LongestAtt && ( Context->schema.max_class == 1 || Context->schema.class_thresholds ) &&
	 ! strcmp(Context->schema.class_names[1], Context->schema.attribute_names[LongestAtt]) )
    {
	ErrorContext(Context, BADDEF4, Nil, Nil);
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


void DefSyntaxError(c50_context *Context, const char *Msg)
/*   --------------  */
{
    String	RestOfText;
    int		i=10;

    if ( ! Context->implicit_state->previous_error )
    {
	RestOfText = Context->implicit_state->buffer.data() + Context->implicit_state->buffer_position;

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

	ErrorContext(Context, BADDEF1, RestOfText, Msg);
	Context->implicit_state->previous_error = true;
    }
}



void DefSemanticsError(c50_context *Context, int Fi, const char *Msg,
                       int OpCode)
/*   -----------------  */
{
    if ( Context->implicit_state->previous_error ) return;

    if ( Fi < 0 || Fi > Context->implicit_state->buffer_position )
    {
        throw c50::exception(c50::error_code::internal_error,
                             "invalid implicit expression bounds");
    }
    const int Length = Context->implicit_state->buffer_position - Fi;
    const char *Start = Context->implicit_state->buffer.data() + Fi;
    std::string Exp;
    if ( Length > 23 )
    {
        Exp.assign(Start, 10);
        Exp += "...";
        Exp.append(Start + Length - 10, 10);
    }
    else
    {
        Exp.assign(Start, static_cast<size_t>(Length));
    }

    const char *Op = "unknown operator";
    switch ( OpCode )
    {
        case OP_AND: Op = "and"; break;
        case OP_OR: Op = "or"; break;
        case OP_SEQ:
        case OP_EQ: Op = "="; break;
        case OP_SNE:
        case OP_NE: Op = "<>"; break;
        case OP_GT: Op = ">"; break;
        case OP_GE: Op = ">="; break;
        case OP_LT: Op = "<"; break;
        case OP_LE: Op = "<="; break;
        case OP_PLUS: Op = "+"; break;
        case OP_MINUS: Op = "-"; break;
        case OP_UMINUS: Op = "unary -"; break;
        case OP_MULT: Op = "*"; break;
        case OP_DIV: Op = "/"; break;
        case OP_MOD: Op = "%"; break;
        case OP_POW: Op = "^"; break;
        case OP_SIN: Op = "sin"; break;
        case OP_COS: Op = "cos"; break;
        case OP_TAN: Op = "tan"; break;
        case OP_LOG: Op = "log"; break;
        case OP_EXP: Op = "exp"; break;
        case OP_INT: Op = "int"; break;
    }

    const std::string XMsg = std::string(Msg) + " with " +
                             static_cast<char>(39) + Op +
                             static_cast<char>(39);
    ErrorContext(Context, BADDEF2, Exp.c_str(), XMsg.c_str());
    Context->implicit_state->previous_error = true;
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
    // A string remains local until its instruction is published successfully.
    std::unique_ptr<char, decltype(&free)> StringOwner(
        OpCode == OP_STR ? S : nullptr, &free);

    if ( Context->implicit_state->buffer[Fi] == ' ' ) Fi++;

    if ( ! UpdateTStack(Context, OpCode, F, S, Fi) ) return;

    /*  Make sure enough room for this element  */

    if ( Context->implicit_state->definition_position >= Context->implicit_state->definition_size-1 )
    {
	Realloc(Context->schema.attribute_definitions[Context->schema.max_attribute], Context->implicit_state->definition_size += 100, DefElt);
    }

    DefOp(Context->schema.attribute_definitions[Context->schema.max_attribute][Context->implicit_state->definition_position]) = OpCode;
    if ( OpCode == OP_ATT || OpCode == OP_STR )
    {
	DefSVal(Context->schema.attribute_definitions[Context->schema.max_attribute][Context->implicit_state->definition_position]) = S;
    }
    else
    {
	DefNVal(Context->schema.attribute_definitions[Context->schema.max_attribute][Context->implicit_state->definition_position]) = F;
    }

    Context->implicit_state->definition_position++;
    DefOp(Context->schema.attribute_definitions[Context->schema.max_attribute][Context->implicit_state->definition_position]) = OP_END;
    StringOwner.release();
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

    int Required = 0;
    switch ( OpCode )
    {
	case OP_AND: case OP_OR: case OP_EQ: case OP_NE:
	case OP_GT: case OP_GE: case OP_LT: case OP_LE:
	case OP_SEQ: case OP_SNE: case OP_PLUS: case OP_MINUS:
	case OP_MULT: case OP_DIV: case OP_MOD: case OP_POW:
	    Required = 2;
	    break;
	case OP_UMINUS: case OP_SIN: case OP_COS: case OP_TAN:
	case OP_LOG: case OP_EXP: case OP_INT:
	    Required = 1;
	    break;
    }
    if ( Context->implicit_state->type_stack_position < Required )
    {
	throw c50::exception(c50::error_code::internal_error,
	                     "invalid implicit expression type stack");
    }

    if ( static_cast<size_t>(Context->implicit_state->type_stack_position) >=
         Context->implicit_state->type_stack.size() )
    {
	Context->implicit_state->type_stack.resize(
	    static_cast<size_t>(Context->implicit_state->type_stack_position) + 1);
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
#define DUNA(a)	(DUnknownVal(StackAt(a)) || NotApplicVal(StackAt(a)))
#define CUNA(a)	(CUnknownVal(StackAt(a)) || NotApplicVal(StackAt(a)))
#define	C1(x)	(CUNA(XSN-1) ? UNKNOWN : (x))
#define	C2(x)	(CUNA(XSN-1) || CUNA(XSN-2) ? UNKNOWN : (x))
#define	CD2(x)	(CUNA(XSN-1) || CUNA(XSN-2) ? UNKNOWN : (x))
#define	D2(x)	(DUNA(XSN-1) || DUNA(XSN-2) ? UNKNOWN : (x))


AttValue EvaluateDef(c50_context *Context, Definition D, DataRec Case)
/*       -----------  */
{
    std::vector<XStackElt> XStack;
    XStack.reserve(MaxExpressionDepth);
    int		XSN=0, DN, bv1, bv2;
    double	cv1, cv2, Mult;
    String	sv1, sv2;
    Attribute	Att;
    DefElt	DElt;
    AttValue	ReturnVal;

    const auto InvalidStack = []() -> void {
	throw c50::exception(c50::error_code::internal_error,
	                     "invalid compiled implicit expression");
    };
    const auto StackAt = [&](int Index) -> XStackElt & {
	if ( Index < 0 || Index >= XSN ) InvalidStack();
	return XStack[static_cast<size_t>(Index)];
    };
    const auto StackPush = [&]() -> XStackElt & {
	if ( XSN >= MaxExpressionDepth ) InvalidStack();
	if ( static_cast<size_t>(XSN) == XStack.size() ) XStack.emplace_back();
	return XStack[static_cast<size_t>(XSN++)];
    };

    for ( DN = 0 ; ; DN++)
    {
	switch ( DefOp((DElt = D[DN])) )
	{
	    case OP_ATT:
		    Att = (Attribute) (intptr_t) DefSVal(DElt);

		    if ( Continuous(Att) )
		    {
			StackPush().cval = CVal(Case, Att);
		    }
		    else
		    {
			StackPush().sval =
			    ( Unknown(Case, Att) && ! NotApplic(Context, Case, Att) ? 0 :
			      Context->schema.attribute_value_names[Att][XDVal(Case, Att)] );
		    }
		    break;

	    case OP_NUM:
		    StackPush().cval = DefNVal(DElt);
		    break;

	    case OP_STR:
		    StackPush().sval = DefSVal(DElt);
		    break;

	    case OP_AND:
		    bv1 = StackAt(XSN-2).dval;
		    bv2 = StackAt(XSN-1).dval;
		    StackAt(XSN-2).dval = ( bv1 == 3 || bv2 == 3 ? 3 :
					   D2(bv1 == 2 && bv2 == 2 ? 2 : 3) );
		    XSN--;
		    break;

	    case OP_OR:
		    bv1 = StackAt(XSN-2).dval;
		    bv2 = StackAt(XSN-1).dval;
		    StackAt(XSN-2).dval = ( bv1 == 2 || bv2 == 2 ? 2 :
					   D2(bv1 == 2 || bv2 == 2 ? 2 : 3) );
		    XSN--;
		    break;

	    case OP_EQ:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).dval = ( cv1 == cv2 ? 2 : 3 );
		    XSN--;
		    break;

	    case OP_NE:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).dval = ( cv1 != cv2 ? 2 : 3 );
		    XSN--;
		    break;

	    case OP_GT:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).dval = CD2(cv1 > cv2 ? 2 : 3);
		    XSN--;
		    break;

	    case OP_GE:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).dval = CD2(cv1 >= cv2 ? 2 : 3);
		    XSN--;
		    break;

	    case OP_LT:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).dval = CD2(cv1 < cv2 ? 2 : 3);
		    XSN--;
		    break;

	    case OP_LE:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).dval = CD2(cv1 <= cv2 ? 2 : 3);
		    XSN--;
		    break;

	    case OP_SEQ:
		    sv1 = StackAt(XSN-2).sval;
		    sv2 = StackAt(XSN-1).sval;
		    StackAt(XSN-2).dval =
			( ! sv1 && ! sv2 ? 2 :
			  ! sv1 || ! sv2 ? 3 :
			  ! strcmp(sv1, sv2) ? 2 : 3 );
		    XSN--;
		    break;

	    case OP_SNE:
		    sv1 = StackAt(XSN-2).sval;
		    sv2 = StackAt(XSN-1).sval;
		    StackAt(XSN-2).dval =
			( ! sv1 && ! sv2 ? 3 :
			  ! sv1 || ! sv2 ? 2 :
			  strcmp(sv1, sv2) ? 2 : 3 );
		    XSN--;
		    break;

	    case OP_PLUS:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).cval = C2(cv1 + cv2);
		    XSN--;
		    break;

	    case OP_MINUS:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).cval = C2(cv1 - cv2);
		    XSN--;
		    break;

	    case OP_MULT:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).cval = C2(cv1 * cv2);
		    XSN--;
		    break;

	    case OP_DIV:
		    /*  Note: have to set precision of result  */

		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    /*  An infinite divisor would never finish the precision
			scaling loop below, so treat it like a zero divisor  */

		    if ( ! cv2 || isinf(cv2) ||
			 CUnknownVal(StackAt(XSN-2)) ||
			 CUnknownVal(StackAt(XSN-1)) ||
			 NotApplicVal(StackAt(XSN-2)) ||
			 NotApplicVal(StackAt(XSN-1)) )
		    {
			StackAt(XSN-2).cval = UNKNOWN;
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
			StackAt(XSN-2).cval = rint(cv1 * Mult) / Mult;
		    }
		    XSN--;
		    break;

	    case OP_MOD:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).cval = C2(fmod(cv1, cv2));
		    XSN--;
		    break;

	    case OP_POW:
		    cv1 = StackAt(XSN-2).cval;
		    cv2 = StackAt(XSN-1).cval;
		    StackAt(XSN-2).cval =
			( CUNA(XSN-1) || CUNA(XSN-2) ||
			  ( cv1 < 0 && ceil(cv2) != cv2 ) ? UNKNOWN :
			  pow(cv1, cv2) );
		    XSN--;
		    break;

	    case OP_UMINUS:
		    cv1 = StackAt(XSN-1).cval;
		    StackAt(XSN-1).cval = C1(-cv1);
		    break;

	    case OP_SIN:
		    cv1 = StackAt(XSN-1).cval;
		    StackAt(XSN-1).cval = C1(sin(cv1));
		    break;

	    case OP_COS:
		    cv1 = StackAt(XSN-1).cval;
		    StackAt(XSN-1).cval = C1(cos(cv1));
		    break;

	    case OP_TAN:
		    cv1 = StackAt(XSN-1).cval;
		    StackAt(XSN-1).cval = C1(tan(cv1));
		    break;

	    case OP_LOG:
		    cv1 = StackAt(XSN-1).cval;
		    StackAt(XSN-1).cval =
			( CUNA(XSN-1) || cv1 <= 0 ? UNKNOWN : log(cv1) );
		    break;

	    case OP_EXP:
		    cv1 = StackAt(XSN-1).cval;
		    StackAt(XSN-1).cval = C1(exp(cv1));
		    break;

	    case OP_INT:
		    cv1 = StackAt(XSN-1).cval;
		    StackAt(XSN-1).cval = C1(rint(cv1));
		    break;

	    case OP_END:
		    if ( XSN != 1 ) InvalidStack();
		    ReturnVal.cval = StackAt(0).cval;	/* cval >= dval bytes */
		    return ReturnVal;
	}
    }
}
