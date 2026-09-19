/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "defns.i"
#include "extern.i"
#include "c50_api_internal.h"

static void Usage(void)
{
    fprintf(stderr, "Usage: prediction-probe <filestem> <tree|rules>\n");
}

int main(int argc, char **argv)
{
    FILE *F;
    c50_input NamesInput;
    ClassNo Actual, Predicted, c;
    CaseNo i;
    String Extension;
    c50_context *Context = NULL;

    if ( c50_context_create(&Context) != C50_STATUS_OK ) return 1;

    if ( argc != 3 ||
         ( strcmp(argv[2], "tree") && strcmp(argv[2], "rules") ) )
    {
        Usage();
        return 1;
    }

    Of = stderr;
    FileStem = argv[1];
    RULES = ! strcmp(argv[2], "rules");
    Extension = ( RULES ? ".rules" : ".tree" );

    if ( ! (F = GetFile(".names", "r")) ) Error(NOFILE, "", "");
    c50_input_init_file(&NamesInput, F);
    GetNames(Context, &NamesInput);
    fclose(F);

    Context->cases.some_missing = AllocZero(Context->schema.max_attribute+1, Boolean);
    Context->cases.some_not_applicable = AllocZero(Context->schema.max_attribute+1, Boolean);

    CheckFile(Context, Extension, false);
    MaxTree = TRIALS-1;

    if ( RULES )
    {
        RuleSet = AllocZero(TRIALS+1, CRuleSet);
        ForEach(Trial, 0, TRIALS-1)
        {
            RuleSet[Trial] = GetRules(Context, Extension);
        }
        Context->most_specific_rules = Alloc(Context->schema.max_class+1, CRule);
    }
    else
    {
        Pruned = AllocZero(TRIALS+1, Tree);
        ForEach(Trial, 0, TRIALS-1)
        {
            Pruned[Trial] = GetTree(Context, Extension);
        }
    }

    Context->default_class =
        ( RULES ? RuleSet[0]->SDefault : Pruned[0]->Leaf );
    Context->class_sum = AllocZero(Context->schema.max_class+1, float);
    Context->votes = AllocZero(Context->schema.max_class+1, float);
    Context->trial_predictions = AllocZero(TRIALS, ClassNo);

    if ( ! (F = GetFile(".test", "r")) ) Error(NOFILE, "", "");
    GetData(Context, F, false, false);

    printf("case,actual,predicted,confidence");
    ForEach(c, 1, Context->schema.max_class)
    {
        printf(",score(%s)", Context->schema.class_names[c]);
    }
    putchar('\n');

    ForEach(i, 0, Context->cases.max_case)
    {
        Actual = Class(Context->cases.records[i]);
        Predicted = Classify(Context, Context->cases.records[i]);
        printf("%d,%s,%s,%.7g", i+1, Context->schema.class_names[Actual],
               Context->schema.class_names[Predicted], Context->confidence);
        ForEach(c, 1, Context->schema.max_class)
        {
            printf(",%.7g", Context->class_sum[c]);
        }
        putchar('\n');
    }

    Cleanup(Context);
    c50_context_destroy(Context);
    return 0;
}
