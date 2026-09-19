/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <c50/c50.h>

static char *ReadFile(const char *path, size_t *size)
{
    FILE *file;
    long length;
    char *data;

    file = fopen(path, "rb");
    if ( ! file ) return NULL;
    if ( fseek(file, 0, SEEK_END) || (length = ftell(file)) < 0 ||
         fseek(file, 0, SEEK_SET) )
    {
        fclose(file);
        return NULL;
    }
    data = malloc(length ? (size_t) length : 1);
    if ( ! data || fread(data, 1, (size_t) length, file) != (size_t) length )
    {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size = (size_t) length;
    return data;
}

static char *ReadExtension(const char *stem, const char *extension, size_t *size)
{
    size_t path_size = strlen(stem) + strlen(extension) + 1;
    char *path = malloc(path_size);
    char *data;

    if ( ! path ) return NULL;
    snprintf(path, path_size, "%s%s", stem, extension);
    data = ReadFile(path, size);
    free(path);
    return data;
}

int main(int argc, char **argv)
{
    const char *model_extension;
    c50_model_kind kind;
    char *names = NULL, *model_data = NULL, *cases = NULL;
    size_t names_size = 0, model_size = 0, cases_size = 0;
    c50_context *context = NULL;
    c50_model *model = NULL;
    c50_predictions *predictions = NULL;
    c50_status status;
    size_t row, class_index;
    int result = 1;

    if ( argc != 3 || (strcmp(argv[2], "tree") && strcmp(argv[2], "rules")) )
    {
        fprintf(stderr, "Usage: api-prediction-probe <filestem> <tree|rules>\n");
        return 1;
    }

    kind = ! strcmp(argv[2], "rules") ? C50_MODEL_RULES : C50_MODEL_TREE;
    model_extension = kind == C50_MODEL_RULES ? ".rules" : ".tree";
    names = ReadExtension(argv[1], ".names", &names_size);
    model_data = ReadExtension(argv[1], model_extension, &model_size);
    cases = ReadExtension(argv[1], ".test", &cases_size);
    if ( ! names || ! model_data || ! cases ) goto cleanup;
    if ( c50_context_create(&context) != C50_STATUS_OK ) goto cleanup;

    status = c50_model_load(context, kind, names, names_size,
                            model_data, model_size, NULL, 0, &model);
    if ( status != C50_STATUS_OK )
    {
        fprintf(stderr, "%s\n", c50_context_error_message(context));
        goto cleanup;
    }
    status = c50_model_predict(context, model, cases, cases_size, &predictions);
    if ( status != C50_STATUS_OK )
    {
        fprintf(stderr, "%s\n", c50_context_error_message(context));
        goto cleanup;
    }

    printf("case,predicted,confidence");
    for ( class_index = 0;
          class_index < c50_predictions_class_count(predictions);
          class_index++ )
    {
        printf(",score(%s)",
               c50_predictions_class_name(predictions, class_index));
    }
    putchar('\n');
    for ( row = 0; row < c50_predictions_row_count(predictions); row++ )
    {
        printf("%lu,%s,%.7g", (unsigned long) row + 1,
               c50_predictions_class_name(
                   predictions, c50_predictions_class_index(predictions, row)),
               c50_predictions_confidence(predictions, row));
        for ( class_index = 0;
              class_index < c50_predictions_class_count(predictions);
              class_index++ )
        {
            printf(",%.7g",
                   c50_predictions_score(predictions, row, class_index));
        }
        putchar('\n');
    }
    result = 0;

cleanup:
    c50_predictions_destroy(predictions);
    c50_model_destroy(model);
    c50_context_destroy(context);
    free(names);
    free(model_data);
    free(cases);
    return result;
}
