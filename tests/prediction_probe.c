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
    Context->options.rules = ! strcmp(argv[2], "rules");
    Extension = ( Context->options.rules ? ".rules" : ".tree" );

    if ( ! (F = GetFile(".names", "r")) ) Error(NOFILE, "", "");
    c50_input_init_file(&NamesInput, F);
    GetNames(Context, &NamesInput);
    fclose(F);

    Context->cases.some_missing = AllocZero(Context->schema.max_attribute+1, Boolean);
    Context->cases.some_not_applicable = AllocZero(Context->schema.max_attribute+1, Boolean);

    CheckFile(Context, Extension, false);
    Context->trees.max_tree = Context->options.trials-1;

    if ( Context->options.rules )
    {
        Context->rules.sets = AllocZero(Context->options.trials+1, CRuleSet);
        ForEach(Context->trees.trial, 0, Context->options.trials-1)
        {
            Context->rules.sets[Context->trees.trial] = GetRules(Context, Extension);
        }
        Context->most_specific_rules = Alloc(Context->schema.max_class+1, CRule);
    }
    else
    {
        Context->trees.pruned = AllocZero(Context->options.trials+1, Tree);
        ForEach(Context->trees.trial, 0, Context->options.trials-1)
        {
            Context->trees.pruned[Context->trees.trial] = GetTree(Context, Extension);
        }
    }

    Context->default_class =
        ( Context->options.rules ? Context->rules.sets[0]->SDefault : Context->trees.pruned[0]->Leaf );
    Context->class_sum = AllocZero(Context->schema.max_class+1, float);
    Context->votes = AllocZero(Context->schema.max_class+1, float);
    Context->trial_predictions = AllocZero(Context->options.trials, ClassNo);

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
