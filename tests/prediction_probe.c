/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "defns.i"
#include "extern.i"

static void Usage(void)
{
    fprintf(stderr, "Usage: prediction-probe <filestem> <tree|rules>\n");
}

int main(int argc, char **argv)
{
    FILE *F;
    ClassNo Actual, Predicted, c;
    CaseNo i;
    String Extension;

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
    GetNames(F);

    SomeMiss = AllocZero(MaxAtt+1, Boolean);
    SomeNA = AllocZero(MaxAtt+1, Boolean);

    CheckFile(Extension, false);
    MaxTree = TRIALS-1;

    if ( RULES )
    {
        RuleSet = AllocZero(TRIALS+1, CRuleSet);
        ForEach(Trial, 0, TRIALS-1)
        {
            RuleSet[Trial] = GetRules(Extension);
        }
        MostSpec = Alloc(MaxClass+1, CRule);
    }
    else
    {
        Pruned = AllocZero(TRIALS+1, Tree);
        ForEach(Trial, 0, TRIALS-1)
        {
            Pruned[Trial] = GetTree(Extension);
        }
    }

    Default = ( RULES ? RuleSet[0]->SDefault : Pruned[0]->Leaf );
    ClassSum = AllocZero(MaxClass+1, float);
    Vote = AllocZero(MaxClass+1, float);
    TrialPred = AllocZero(TRIALS, ClassNo);

    if ( ! (F = GetFile(".test", "r")) ) Error(NOFILE, "", "");
    GetData(F, false, false);

    printf("case,actual,predicted,confidence");
    ForEach(c, 1, MaxClass)
    {
        printf(",score(%s)", ClassName[c]);
    }
    putchar('\n');

    ForEach(i, 0, MaxCase)
    {
        Actual = Class(Case[i]);
        Predicted = Classify(Case[i]);
        printf("%d,%s,%s,%.7g", i+1, ClassName[Actual],
               ClassName[Predicted], Confidence);
        ForEach(c, 1, MaxClass)
        {
            printf(",%.7g", ClassSum[c]);
        }
        putchar('\n');
    }

    Cleanup();
    return 0;
}
