/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <new>

#include <c50/c50.hpp>
#include <vector>
#include <utility>

#include "c50_api_internal.h"
#include "defns.i"
#include "extern.i"

using c50_model = c50::detail::model_data;
using c50_predictions = c50::detail::prediction_data;
using c50_model_kind = c50::model_kind;
using c50_options = c50::options;
using c50_dense_dataset = c50::dense_dataset;
using owned_model = std::unique_ptr<c50_model>;
using owned_predictions = std::unique_ptr<c50_predictions>;

struct file_closer
{
    void operator()(FILE *file) const { if (file) fclose(file); }
};
using owned_file = std::unique_ptr<FILE, file_closer>;

struct c50::detail::model_data
{
    model_kind kind;
    std::string names_data;
    std::string model_data;
    std::string costs_data;
};

struct c50::detail::predictor_data
{
    model_data model;
    std::unique_ptr<c50_context> workspace;

    explicit predictor_data(const model_data &source) : model(source) {}
    ~predictor_data() { reset(); }

    void reset()
    {
        if (workspace) Cleanup(workspace.get());
        workspace.reset();
    }
};

struct c50::detail::prediction_data
{
    std::vector<std::string> class_names;
    std::vector<size_t> class_indices;
    std::vector<double> confidences;
    std::vector<double> scores;
};

typedef struct c50_model_load_state
{
    c50_model_kind kind;
    const char *names_data;
    size_t names_size;
    const char *model_data;
    size_t model_size;
    const char *costs_data;
    size_t costs_size;
    owned_model model;
} c50_model_load_state;

typedef struct c50_train_state
{
    c50_model_kind kind;
    c50_options options;
    const char *names_data;
    size_t names_size;
    const char *training_data;
    size_t training_size;
    const c50_dense_dataset *dense_dataset;
    const char *costs_data;
    size_t costs_size;
    owned_file diagnostics;
    owned_model model;
} c50_train_state;

typedef struct c50_predict_state
{
    const c50_model *model;
    const char *cases_data;
    size_t cases_size;
    const c50_dense_dataset *dense_dataset;
    owned_predictions predictions;
} c50_predict_state;

struct c50_inspect_state
{
    const c50_model *model;
    c50::model_inspection result;
};

static void RequireInputEnd(c50_context *Context, c50_input *Input)
{
    int c;
    do
    {
        c = c50_input_getc(Input);
    } while ( c != EOF && isspace(static_cast<unsigned char>(c)) );

    if ( c != EOF )
    {
        c50_record_error(Context, c50::error_code::parse_error,
                         "unexpected trailing model data");
        C50Exit(Context, 1);
    }
}

static void ParseModel(c50_context *Context, const c50_model *model)
{
    c50_input names_input, model_input, costs_input, *costs = NULL;

    Context->io.output = NULL;
    Context->io.file_stem = "memory";
    Context->io.file_name = "memory.model";
    Context->options.rules = model->kind == c50::model_kind::rules;
    Context->options.trials = 1;
    Context->trees.max_tree = -1;
    Context->options.sample_fraction = 0;

    c50_input_init_memory(&names_input, model->names_data.data(),
                          model->names_data.size());
    GetNames(Context, &names_input);

    c50_input_init_memory(&model_input, model->model_data.data(),
                          model->model_data.size());
    if ( model->costs_data.size() )
    {
        c50_input_init_memory(&costs_input, model->costs_data.data(),
                              model->costs_data.size());
        costs = &costs_input;
    }
    ReadHeaderMemory(Context, &model_input, costs);
    if ( Context->options.trials < 1 )
    {
        c50_record_error(Context, c50::error_code::parse_error,
                         "model contains no classifier entries");
        C50Exit(Context, 1);
    }

    Context->trees.max_tree = Context->options.trials - 1;
    if ( Context->options.rules )
    {
        Context->rules.sets = AllocZero(Context->options.trials + 1, CRuleSet);
        ForEach(Context->trees.trial, 0, Context->options.trials - 1)
        {
            InRulesAt(Context, &model_input, &Context->rules.sets[Context->trees.trial]);
        }
    }
    else
    {
        Context->trees.pruned = AllocZero(Context->options.trials + 1, Tree);
        ForEach(Context->trees.trial, 0, Context->options.trials - 1)
        {
            InTreeAt(Context, &model_input, &Context->trees.pruned[Context->trees.trial]);
            if (Context->trees.pruned[Context->trees.trial]->Cases < Epsilon)
            {
                c50_record_error(Context, c50::error_code::parse_error,
                                 "tree root has insufficient class frequencies");
                C50Exit(Context, 1);
            }
        }
    }
    RequireInputEnd(Context, &model_input);
}

static void InspectModel(c50_context *Context, void *user_data)
{
    auto *state = static_cast<c50_inspect_state *>(user_data);
    ParseModel(Context, state->model);
    state->result = c50_build_inspection(Context);
}

static void LoadModel(c50_context *Context, void *user_data)
{
    c50_model_load_state *state =
        static_cast<c50_model_load_state *>(user_data);

    state->model = std::make_unique<c50_model>();
    state->model->kind = state->kind;
    state->model->names_data.assign(state->names_data, state->names_size);
    state->model->model_data.assign(state->model_data, state->model_size);
    if (state->costs_size)
        state->model->costs_data.assign(state->costs_data, state->costs_size);

    ParseModel(Context, state->model.get());
}

static void CleanupModelLoad(c50_context *Context, void *user_data)
{
    (void) user_data;
    Cleanup(Context);
    Context->io.output = NULL;
}

static const char *ValidateOptions(const c50_options *options)
{
    if ( options->trials < 1 || options->trials > 1000 )
    {
        return "options.trials must be between 1 and 1000";
    }
    if ( ! isfinite(options->minimum_cases) ||
         options->minimum_cases < 1 || options->minimum_cases > 1000000 )
    {
        return "options.minimum_cases must be between 1 and 1000000";
    }
    if ( ! isfinite(options->confidence_factor) ||
         options->confidence_factor < 0 || options->confidence_factor > 1 )
    {
        return "options.confidence_factor must be between 0 and 1";
    }
    if ( ! isfinite(options->sample_fraction) ||
         options->sample_fraction < 0 || options->sample_fraction > 0.999 )
    {
        return "options.sample_fraction must be between 0 and 0.999";
    }
    if ( options->random_seed > 4095 )
    {
        return "options.random_seed must be between 0 and 4095";
    }
    return NULL;
}

static const char *ValidateDenseDataset(const c50_dense_dataset *dataset,
                                        int training)
{
    if ( dataset->row_count > INT_MAX )
    {
        return "dense dataset has too many rows";
    }
    if ( dataset->feature_count > INT_MAX )
    {
        return "dense dataset has too many features";
    }
    if ( training && ! dataset->row_count )
    {
        return "dense training dataset contains no rows";
    }
    if ( dataset->row_count && ! dataset->values )
    {
        return "dense dataset values are NULL";
    }
    if ( dataset->row_count &&
         dataset->row_stride < dataset->feature_count )
    {
        return "dense dataset row_stride is smaller than feature_count";
    }
    if ( dataset->row_count && dataset->row_stride &&
         dataset->row_count > SIZE_MAX / dataset->row_stride )
    {
        return "dense dataset dimensions overflow size_t";
    }
    if ( training && ! dataset->class_indices )
    {
        return "dense training class_indices are NULL";
    }
    return NULL;
}

static void DenseError(c50_context *Context, c50::error_code status,
                       const char *message)
{
    c50_record_error(Context, status, message);
    C50Exit(Context, 1);
}

static void ValidateDenseSchema(c50_context *Context,
                                const c50_dense_dataset *dataset)
{
    Attribute Att;

    if ( Context->schema.class_attribute )
    {
        DenseError(Context, c50::error_code::unsupported,
                   "dense data does not support a class attribute");
    }
    if ( dataset->feature_count !=
         (size_t) Context->schema.max_attribute )
    {
        DenseError(Context, c50::error_code::invalid_argument,
                   "dense feature count does not match names data");
    }

    ForEach(Att, 1, Context->schema.max_attribute)
    {
        if ( Context->schema.attribute_definitions[Att] )
        {
            DenseError(Context, c50::error_code::unsupported,
                       "dense data does not support implicit attributes");
        }
        if ( Exclude(Att) )
        {
            DenseError(Context, c50::error_code::unsupported,
                       "dense data does not support ignored or label attributes");
        }
        if ( DateVal(Att) || TimeVal(Att) || TStampVal(Att) )
        {
            DenseError(Context, c50::error_code::unsupported,
                       "dense data does not support date or time attributes");
        }
        if ( StatBit(Att, DISCRETE) )
        {
            DenseError(Context, c50::error_code::unsupported,
                       "dense categorical features require explicit values");
        }
    }
}

static void LoadDenseData(c50_context *Context,
                          const c50_dense_dataset *dataset, Boolean Train)
{
    size_t row;
    Attribute Att;
    CaseNo WantTrain=0, LeftTrain=0;
    Boolean SelectTrain;
    DataRec DVec;

    ValidateDenseSchema(Context, dataset);
    Context->cases.max_case = -1;
    Context->max_label = 0;
    Context->cases.records =
        Alloc(dataset->row_count ? dataset->row_count : 1, DataRec);

    if ( Train && Context->options.sample_fraction )
    {
        Context->sample_from = (CaseNo) dataset->row_count;
        ResetKR(&Context->random, Context->io.random_initial_seed);
        WantTrain = Context->sample_from * Context->options.sample_fraction + 0.5;
        LeftTrain = Context->sample_from;
    }

    for ( row = 0; row < dataset->row_count; row++ )
    {
        DVec = NewCase(Context);
        Weight(DVec) = 1;

        ForEach(Att, 1, Context->schema.max_attribute)
        {
            const double value =
                dataset->values[row * dataset->row_stride + Att - 1];

            if ( isnan(value) )
            {
                DVal(DVec, Att) = UNKNOWN;
                Context->cases.some_missing[Att] = true;
            }
            else if ( Discrete(Att) )
            {
                const size_t value_count =
                    (size_t) Context->schema.max_attribute_value[Att] - 1;
                size_t value_index;

                if ( ! isfinite(value) || value < 0 || floor(value) != value ||
                     value >= (double) value_count )
                {
                    DenseError(Context, c50::error_code::invalid_argument,
                               "dense categorical value is outside its schema");
                }
                value_index = (size_t) value;
                DVal(DVec, Att) = (DiscrValue) value_index + 2;
            }
            else
            {
                const ContValue converted = (ContValue) value;

                if ( ! isfinite(value) || ! isfinite(converted) )
                {
                    DenseError(Context, c50::error_code::invalid_argument,
                               "dense continuous value is not finite");
                }
                CVal(DVec, Att) = converted;
            }
        }

        if ( Train )
        {
            if ( dataset->class_indices[row] >=
                 (size_t) Context->schema.max_class )
            {
                DenseError(Context, c50::error_code::invalid_argument,
                           "dense class index is outside the class schema");
            }
            Class(DVec) = (ClassNo) dataset->class_indices[row] + 1;
        }
        else
        {
            Class(DVec) = 0;
        }

        if ( Train && Context->options.sample_fraction )
        {
            SelectTrain =
                KRandom(&Context->random) < WantTrain / (float) LeftTrain--;
            if ( SelectTrain ) WantTrain--;
            if ( ! SelectTrain )
            {
                FreeLastCase(Context, DVec);
                continue;
            }
        }

        Context->cases.records[++Context->cases.max_case] = DVec;
    }
}

static void TrainModel(c50_context *Context, void *user_data)
{
    c50_train_state *state = static_cast<c50_train_state *>(user_data);
    c50_input names_input, training_input, costs_input;
    unsigned char *serialized;
    size_t serialized_size;

    state->diagnostics.reset(tmpfile());
    if ( ! state->diagnostics )
    {
        c50_record_error(Context, c50::error_code::io_error,
                         "could not create training diagnostics stream");
        C50Exit(Context, 1);
    }

    Context->io.output = state->diagnostics.get();
    Context->progress.update_file = state->diagnostics.get();
    Context->io.file_stem = "memory";
    Context->io.attribute_exclusions = 0;
    Context->io.random_initial_seed = (int) state->options.random_seed;
    Context->schema.max_discrete_value = 3;
    Context->cases.max_case = -1;
    Context->trees.max_tree = -1;
    /* Winnowing runs before ConstructClassifiers sets the trial, and must
       not see values left by a previous training on this context. */
    Context->trees.trial = 0;
    Context->average_case_weight = 0;
    Context->attributes_winnowed = false;
    Context->costs.unit_weights = true;
    Context->costs.weighted = false;
    Context->options.verbosity = 0;
    Context->options.trials = (int) state->options.trials;
    Context->options.folds = 10;
    Context->options.utility_bands = 0;
    Context->options.subset_splits = state->options.subset_splits;
    Context->options.boosting = state->options.trials > 1;
    Context->options.probabilistic_thresholds =
        state->options.probabilistic_thresholds;
    Context->options.rules = state->kind == c50::model_kind::rules;
    Context->options.cross_validation = false;
    Context->options.ignore_costs = state->options.ignore_costs;
    Context->options.winnow = state->options.winnow;
    Context->options.global_pruning = state->options.global_pruning;
    Context->options.minimum_cases = (float) state->options.minimum_cases;
    Context->options.leaf_ratio = 0;
    Context->options.confidence_factor =
        (float) state->options.confidence_factor;
    Context->options.sample_fraction =
        (float) state->options.sample_fraction;
    Context->splits.sample_fraction = 1;
    Context->last_model_extension = NULL;

    c50_output_init_memory(&Context->classifier_output);
    Context->classifier_output_active = true;

    c50_input_init_memory(&names_input, state->names_data, state->names_size);
    GetNames(Context, &names_input);

    Context->cases.some_missing =
        AllocZero(Context->schema.max_attribute + 1, Boolean);
    Context->cases.some_not_applicable =
        AllocZero(Context->schema.max_attribute + 1, Boolean);

    if ( state->dense_dataset )
    {
        LoadDenseData(Context, state->dense_dataset, true);
    }
    else
    {
        c50_input_init_memory(&training_input, state->training_data,
                              state->training_size);
        GetDataInput(Context, &training_input, true, false);
    }
    if ( Context->cases.max_case < 0 )
    {
        c50_record_error(Context, c50::error_code::parse_error,
                         "training data contains no cases");
        C50Exit(Context, 1);
    }

    if ( ! Context->options.ignore_costs && state->costs_size )
    {
        c50_input_init_memory(&costs_input, state->costs_data,
                              state->costs_size);
        GetMCostsInput(Context, &costs_input);
    }

    InitialiseTreeData(Context);
    if ( Context->options.rules )
    {
        Context->rules.sets =
            AllocZero(Context->options.trials + 1, CRuleSet);
    }
    if ( Context->options.winnow )
    {
        NotifyStage(Context, WINNOWATTS);
        Progress(Context, -Context->schema.max_attribute);
        WinnowAtts(Context);
    }
    ConstructClassifiers(Context);

    state->model = std::make_unique<c50_model>();
    serialized = c50_output_take_memory(&Context->classifier_output,
                                        &serialized_size);
    Context->classifier_output_active = false;
    std::unique_ptr<unsigned char, decltype(&free)> buffer(serialized, &free);
    if (!buffer) throw std::bad_alloc();

    state->model->kind = state->kind;
    state->model->names_data.assign(state->names_data, state->names_size);
    state->model->model_data.assign(reinterpret_cast<char *>(buffer.get()),
                                    serialized_size);
    if (!state->options.ignore_costs && state->costs_size)
        state->model->costs_data.assign(state->costs_data, state->costs_size);

}

static void CleanupTraining(c50_context *Context, void *user_data)
{
    c50_train_state *state = static_cast<c50_train_state *>(user_data);

    Context->progress.update_file = NULL;
    Cleanup(Context);
    c50_clear_prediction_state(Context);
    Context->io.output = NULL;
    state->diagnostics.reset();
}

static void PredictParsedModel(c50_context *Context, void *user_data)
{
    c50_predict_state *state = static_cast<c50_predict_state *>(user_data);
    c50_input cases_input;
    c50_predictions *predictions;
    CaseNo row;
    ClassNo class_number, predicted;

    Context->io.error_count = 0;
    Context->line_buffer.clear();
    Context->line_buffer_position = 0;

    /* Sampling is a training-time setting recorded for reproducibility.  It
       must not discard cases supplied to the prediction API. */
    Context->options.sample_fraction = 0;

    Context->cases.some_missing = AllocZero(Context->schema.max_attribute + 1, Boolean);
    Context->cases.some_not_applicable = AllocZero(Context->schema.max_attribute + 1, Boolean);
    if ( Context->options.rules ) Context->most_specific_rules = Alloc(Context->schema.max_class + 1, CRule);
    Context->default_class =
        ( Context->options.rules ? Context->rules.sets[0]->SDefault : Context->trees.pruned[0]->Leaf );
    Context->class_sum = AllocZero(Context->schema.max_class + 1, float);
    Context->votes = AllocZero(Context->schema.max_class + 1, float);
    Context->trial_predictions = AllocZero(Context->options.trials, ClassNo);

    if ( state->dense_dataset )
    {
        LoadDenseData(Context, state->dense_dataset, false);
    }
    else
    {
        c50_input_init_memory(&cases_input, state->cases_data,
                              state->cases_size);
        GetDataInput(Context, &cases_input, false, true);
    }

    state->predictions = std::make_unique<c50_predictions>();
    predictions = state->predictions.get();
    ForEach(class_number, 1, Context->schema.max_class)
        predictions->class_names.emplace_back(Context->schema.class_names[class_number]);

    const size_t row_count = Context->cases.max_case + 1;
    const size_t class_count = predictions->class_names.size();
    if (row_count && class_count > SIZE_MAX / row_count)
        throw std::bad_alloc();
    predictions->class_indices.resize(row_count);
    predictions->confidences.resize(row_count);
    predictions->scores.resize(row_count * class_count);

    ForEach(row, 0, Context->cases.max_case)
    {
        predicted = Classify(Context, Context->cases.records[row]);
        predictions->class_indices[row] = predicted - 1;
        predictions->confidences[row] = Context->confidence;
        ForEach(class_number, 1, Context->schema.max_class)
        {
            predictions->scores[
                row * class_count + class_number - 1] =
                Context->class_sum[class_number];
        }
    }
}

static void PredictModel(c50_context *Context, void *user_data)
{
    auto *state = static_cast<c50_predict_state *>(user_data);
    ParseModel(Context, state->model);
    PredictParsedModel(Context, state);
}

static void CleanupPredictionBatch(c50_context *Context, void *)
{
    FreeData(Context);
    free(Context->cases.some_missing);
    free(Context->cases.some_not_applicable);
    Context->cases.some_missing = nullptr;
    Context->cases.some_not_applicable = nullptr;
    c50_clear_prediction_state(Context);
}

static void PreparePredictor(c50_context *Context, void *user_data)
{
    ParseModel(Context, static_cast<c50_model *>(user_data));
}

static void EnsurePredictor(c50::detail::predictor_data &data)
{
    if (!data.workspace) {
        data.workspace = c50_make_context();
        c50_run_operation(data.workspace.get(), PreparePredictor, nullptr, &data.model);
    }
}

static owned_predictions RunPredictor(c50::detail::predictor_data &data,
                                      c50_predict_state &state)
{
    try {
        EnsurePredictor(data);
        c50_run_operation(data.workspace.get(), PredictParsedModel,
                          CleanupPredictionBatch, &state);
    } catch (...) {
        // Discard all mutable parser state after failure. The next call reparses.
        data.reset();
        throw;
    }
    return std::move(state.predictions);
}

static void CleanupPrediction(c50_context *Context, void *user_data)
{
    (void) user_data;
    Cleanup(Context);
    c50_clear_prediction_state(Context);
    Context->io.output = NULL;
}

namespace c50 {
namespace {
void validate_text(std::string_view text)
{
    if (text.find('\0') != std::string_view::npos)
        throw exception(error_code::invalid_argument, "text input contains a NUL byte");
}

void validate_model_inputs(model_kind kind, std::string_view names,
                           std::string_view costs)
{
    if (kind != model_kind::tree && kind != model_kind::rules)
        throw exception(error_code::invalid_argument, "invalid model kind");
    if (names.empty())
        throw exception(error_code::invalid_argument, "names input is required");
    validate_text(names);
    validate_text(costs);
}

void validate_options(const options &settings)
{
    if (const auto error = ValidateOptions(&settings))
        throw exception(error_code::invalid_argument, error);
}

dense_dataset validate_dense(dense_dataset dataset, bool training)
{
    if (!dataset.row_stride) dataset.row_stride = dataset.feature_count;
    if (const auto error = ValidateDenseDataset(&dataset, training))
        throw exception(error_code::invalid_argument, error);
    return dataset;
}
}

dense_dataset::dense_dataset(const double *data, std::size_t rows,
                             std::size_t features, const std::size_t *classes,
                             std::size_t stride)
    : values(data), row_count(rows), feature_count(features),
      class_indices(classes), row_stride(stride) {}

model::model(std::unique_ptr<detail::model_data> data) : data_(std::move(data)) {}
model::~model() = default;
model::model(model &&) noexcept = default;
model &model::operator=(model &&) noexcept = default;
model_inspection model::inspect(context &workspace) const
{
    c50_inspect_state state{data_.get(), {}};
    c50_run_operation(workspace.state_.get(), InspectModel, CleanupModelLoad, &state);
    return std::move(state.result);
}

predictor model::prepare_predictor() const
{
    auto data = std::make_unique<detail::predictor_data>(*data_);
    EnsurePredictor(*data);
    return predictor(std::move(data));
}

predictor::predictor(std::unique_ptr<detail::predictor_data> data) : data_(std::move(data)) {}
predictor::~predictor() = default;
predictor::predictor(predictor &&) noexcept = default;
predictor &predictor::operator=(predictor &&) noexcept = default;

predictions predictor::predict(std::string_view cases)
{
    validate_text(cases);
    c50_predict_state state{};
    state.cases_data = cases.data();
    state.cases_size = cases.size();
    return predictions(RunPredictor(*data_, state));
}

predictions predictor::predict(const dense_dataset &cases)
{
    auto dataset = validate_dense(cases, false);
    c50_predict_state state{};
    state.dense_dataset = &dataset;
    return predictions(RunPredictor(*data_, state));
}

model_kind model::kind() const noexcept { return data_->kind; }
const std::string &model::names_data() const noexcept { return data_->names_data; }
const std::string &model::serialized_data() const noexcept { return data_->model_data; }
const std::string &model::costs_data() const noexcept { return data_->costs_data; }

model model::train(context &workspace, model_kind kind, std::string_view names,
                   std::string_view training, const options &settings,
                   std::string_view costs)
{
    validate_model_inputs(kind, names, costs);
    validate_options(settings);
    validate_text(training);
    if (training.empty())
        throw exception(error_code::invalid_argument, "training input is required");
    c50_train_state state{};
    state.kind = kind;
    state.options = settings;
    state.names_data = names.data();
    state.names_size = names.size();
    state.training_data = training.data();
    state.training_size = training.size();
    state.costs_data = costs.data();
    state.costs_size = costs.size();
    c50_run_operation(workspace.state_.get(), TrainModel, CleanupTraining, &state);
    return model(std::move(state.model));
}

model model::train(context &workspace, model_kind kind, std::string_view names,
                   const dense_dataset &training, const options &settings,
                   std::string_view costs)
{
    validate_model_inputs(kind, names, costs);
    validate_options(settings);
    auto dataset = validate_dense(training, true);
    c50_train_state state{};
    state.kind = kind;
    state.options = settings;
    state.names_data = names.data();
    state.names_size = names.size();
    state.dense_dataset = &dataset;
    state.costs_data = costs.data();
    state.costs_size = costs.size();
    c50_run_operation(workspace.state_.get(), TrainModel, CleanupTraining, &state);
    return model(std::move(state.model));
}

model model::load(context &workspace, model_kind kind, std::string_view names,
                  std::string_view serialized, std::string_view costs)
{
    validate_model_inputs(kind, names, costs);
    validate_text(serialized);
    if (serialized.empty())
        throw exception(error_code::invalid_argument, "serialized model is required");
    c50_model_load_state state{};
    state.kind = kind;
    state.names_data = names.data();
    state.names_size = names.size();
    state.model_data = serialized.data();
    state.model_size = serialized.size();
    state.costs_data = costs.data();
    state.costs_size = costs.size();
    c50_run_operation(workspace.state_.get(), LoadModel, CleanupModelLoad, &state);
    return model(std::move(state.model));
}

predictions model::predict(context &workspace, std::string_view cases) const
{
    validate_text(cases);
    c50_predict_state state{};
    state.model = data_.get();
    state.cases_data = cases.data();
    state.cases_size = cases.size();
    c50_run_operation(workspace.state_.get(), PredictModel, CleanupPrediction, &state);
    return predictions(std::move(state.predictions));
}

predictions model::predict(context &workspace, const dense_dataset &cases) const
{
    auto dataset = validate_dense(cases, false);
    c50_predict_state state{};
    state.model = data_.get();
    state.dense_dataset = &dataset;
    c50_run_operation(workspace.state_.get(), PredictModel, CleanupPrediction, &state);
    return predictions(std::move(state.predictions));
}

predictions::predictions(std::unique_ptr<detail::prediction_data> data)
    : data_(std::move(data)) {}
predictions::~predictions() = default;
predictions::predictions(predictions &&) noexcept = default;
predictions &predictions::operator=(predictions &&) noexcept = default;
std::size_t predictions::size() const noexcept { return data_->class_indices.size(); }
std::size_t predictions::class_count() const noexcept { return data_->class_names.size(); }
const std::string &predictions::class_name(std::size_t index) const
{ return data_->class_names.at(index); }
std::size_t predictions::class_index(std::size_t row) const
{ return data_->class_indices.at(row); }
double predictions::confidence(std::size_t row) const
{ return data_->confidences.at(row); }
double predictions::score(std::size_t row, std::size_t index) const
{
    if (row >= size() || index >= class_count())
        throw std::out_of_range("prediction score index out of range");
    return data_->scores[row * class_count() + index];
}
} // namespace c50
