/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <c50/c50.h>

int main(void)
{
    static const char names[] =
        "low, high.\n\n"
        "signal: continuous.\n";
    static const char training[] =
        "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n";
    c50_context *context = NULL;
    c50_model *model = NULL;
    int result = 1;

    if ( c50_context_create(&context) != C50_STATUS_OK ) return 1;
    if ( c50_model_train(context, C50_MODEL_TREE, NULL,
                         names, sizeof(names) - 1,
                         training, sizeof(training) - 1,
                         NULL, 0, &model) == C50_STATUS_OK && model )
    {
        result = 0;
    }
    c50_model_destroy(model);
    c50_context_destroy(context);
    return result;
}
