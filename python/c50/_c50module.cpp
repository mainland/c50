/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <exception>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <c50/c50.hpp>

namespace py = pybind11;

namespace
{

const char train_doc[] = R"doc(
Train a C5.0 classifier from text held in memory.

The names, training, and optional costs strings use the corresponding C5.0
file formats. The returned model owns its serialized representation and is
independent of the native context used during training.
``split_workers`` selects 1 through 8 native split-evaluation workers;
the default is one.
)doc";

const char load_doc[] = R"doc(
Load and validate a serialized C5.0 classifier.

The returned model owns copies of the names, serialized classifier, and
optional costs strings.
)doc";

const char train_dense_doc[] = R"doc(
Train a C5.0 classifier from a dense NumPy feature matrix.

Continuous columns contain their numeric values. Categorical columns contain
zero-based indices into the explicit values declared by ``names``. NaN denotes
a missing feature. ``class_indices`` requires an integer dtype and contains
one nonnegative zero-based class index per
row.
``split_workers`` selects 1 through 8 native split-evaluation workers;
the default is one.
)doc";

const char predict_details_dense_doc[] = R"doc(
Predict a dense NumPy feature matrix.

The matrix uses the same continuous, categorical, and missing-value encoding
as ``train_dense``.
)doc";

template<class T>
using dense_values = py::array_t<
    T, py::array::c_style | py::array::forcecast>;

template<class T>
std::vector<std::size_t> copy_checked_classes(const dense_values<T> &classes)
{
    std::vector<std::size_t> result(static_cast<std::size_t>(classes.shape(0)));
    const auto values = classes.template unchecked<1>();
    for (py::ssize_t row = 0; row < values.shape(0); ++row)
    {
        const auto value = values(row);
        if constexpr (std::is_signed_v<T>)
            if (value < 0)
                throw py::value_error("class_indices must be nonnegative");
        if (value >= std::numeric_limits<int>::max())
            throw py::value_error("class_indices exceed the native class range");
        result[static_cast<std::size_t>(row)] = static_cast<std::size_t>(value);
    }
    return result;
}

std::vector<std::size_t> checked_classes(const py::array &classes)
{
    const char kind = classes.dtype().kind();
    if (kind != 'i' && kind != 'u')
        throw py::value_error("class_indices must have an integer dtype");
    if (kind == 'i')
        return copy_checked_classes(dense_values<std::int64_t>(classes));
    return copy_checked_classes(dense_values<std::uint64_t>(classes));
}

const char predict_details_doc[] = R"doc(
Predict cases and return labels, confidences, and per-class scores.

Each case row uses C5.0 data-file syntax and includes a final class field,
which may be ``?`` when unknown.
)doc";

const char predict_doc[] = R"doc(
Predict the class label for each case row.

Each case row uses C5.0 data-file syntax and includes a final class field,
which may be ``?`` when unknown.
)doc";

const char predict_proba_doc[] = R"doc(
Return C5.0's per-class scores for each case row.

Columns follow ``classes_`` order. Each case row uses C5.0 data-file syntax
and includes a final class field, which may be ``?`` when unknown.
)doc";

class python_predictor
{
public:
    explicit python_predictor(c50::predictor predictor) : predictor_(std::move(predictor)) {}

    c50::predictions predict_details(const std::string &cases)
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return predictor_.predict(cases);
    }

    c50::predictions predict_details_dense(const dense_values<double> &values)
    {
        if (values.ndim() != 2)
            throw py::value_error("values must be a two-dimensional array");
        std::vector<double> copy(static_cast<std::size_t>(values.size()));
        std::copy_n(values.data(), copy.size(), copy.data());
        const c50::dense_dataset dataset(copy.data(), static_cast<std::size_t>(values.shape(0)),
                                        static_cast<std::size_t>(values.shape(1)));
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return predictor_.predict(dataset);
    }

private:
    c50::predictor predictor_;
    std::mutex mutex_;
};

class python_model
{
public:
    explicit python_model(c50::model model) : model_(std::move(model)) {}

    python_model(const python_model &) = delete;
    python_model &operator=(const python_model &) = delete;
    python_model(python_model &&) noexcept = default;
    python_model &operator=(python_model &&) noexcept = default;

    static python_model train(const std::string &names,
                              const std::string &training_data,
                              c50::model_kind kind,
                              const c50::options &options,
                              const std::string &costs,
                              unsigned int split_workers)
    {
        c50::context context;
        context.split_workers(split_workers);
        py::gil_scoped_release release;
        return python_model(c50::model::train(
            context, kind, names, training_data, options, costs));
    }

    static python_model train_dense(const std::string &names,
                                    const dense_values<double> &values,
                                    const py::array &class_indices,
                                    c50::model_kind kind,
                                    const c50::options &options,
                                    const std::string &costs,
                                    unsigned int split_workers)
    {
        if ( values.ndim() != 2 )
        {
            throw py::value_error("values must be a two-dimensional array");
        }
        if ( class_indices.ndim() != 1 )
        {
            throw py::value_error("class_indices must be a one-dimensional array");
        }
        if ( values.shape(0) != class_indices.shape(0) )
        {
            throw py::value_error(
                "values and class_indices have inconsistent row counts");
        }

        const auto indices = checked_classes(class_indices);
        std::vector<double> value_copy(static_cast<std::size_t>(values.size()));
        std::copy_n(values.data(), value_copy.size(), value_copy.data());
        const c50::dense_dataset dataset(
            value_copy.data(), static_cast<std::size_t>(values.shape(0)),
            static_cast<std::size_t>(values.shape(1)), indices.data());
        c50::context context;
        context.split_workers(split_workers);
        py::gil_scoped_release release;
        return python_model(c50::model::train(
            context, kind, names, dataset, options, costs));
    }

    static python_model load(const std::string &names,
                             const std::string &serialized_data,
                             c50::model_kind kind,
                             const std::string &costs)
    {
        c50::context context;
        py::gil_scoped_release release;
        return python_model(c50::model::load(
            context, kind, names, serialized_data, costs));
    }

    c50::model_inspection inspect() const
    {
        c50::context context;
        py::gil_scoped_release release;
        return model_.inspect(context);
    }

    std::unique_ptr<python_predictor> prepare_predictor() const
    {
        py::gil_scoped_release release;
        return std::make_unique<python_predictor>(model_.prepare_predictor());
    }

    c50::model_kind kind() const noexcept { return model_.kind(); }
    std::string names_data() const { return model_.names_data(); }
    std::string serialized_data() const { return model_.serialized_data(); }
    std::string costs_data() const { return model_.costs_data(); }

    c50::predictions predict_details(const std::string &cases) const
    {
        c50::context context;
        py::gil_scoped_release release;
        return model_.predict(context, cases);
    }

    c50::predictions predict_details_dense(const dense_values<double> &values) const
    {
        if ( values.ndim() != 2 )
        {
            throw py::value_error("values must be a two-dimensional array");
        }

        std::vector<double> value_copy(static_cast<std::size_t>(values.size()));
        std::copy_n(values.data(), value_copy.size(), value_copy.data());
        const c50::dense_dataset dataset(
            value_copy.data(), static_cast<std::size_t>(values.shape(0)),
            static_cast<std::size_t>(values.shape(1)));
        c50::context context;
        py::gil_scoped_release release;
        return model_.predict(context, dataset);
    }

    std::vector<std::string> predict(const std::string &cases) const
    {
        c50::predictions predictions = predict_details(cases);
        std::vector<std::string> labels;

        labels.reserve(predictions.size());
        for ( std::size_t row = 0; row < predictions.size(); row++ )
        {
            labels.emplace_back(
                predictions.class_name(predictions.class_index(row)));
        }
        return labels;
    }

    std::vector<std::vector<double>> predict_proba(
        const std::string &cases) const
    {
        c50::predictions predictions = predict_details(cases);
        std::vector<std::vector<double>> scores(
            predictions.size(),
            std::vector<double>(predictions.class_count()));

        for ( std::size_t row = 0; row < predictions.size(); row++ )
        {
            for ( std::size_t class_index = 0;
                  class_index < predictions.class_count(); class_index++ )
            {
                scores[row][class_index] =
                    predictions.score(row, class_index);
            }
        }
        return scores;
    }

    std::vector<std::string> classes() const
    {
        c50::predictions predictions = predict_details("");
        std::vector<std::string> names;

        names.reserve(predictions.class_count());
        for ( std::size_t class_index = 0;
              class_index < predictions.class_count(); class_index++ )
        {
            names.emplace_back(predictions.class_name(class_index));
        }
        return names;
    }

private:
    c50::model model_;
};

std::vector<std::string> prediction_class_names(
    const c50::predictions &predictions)
{
    std::vector<std::string> names;

    names.reserve(predictions.class_count());
    for ( std::size_t class_index = 0;
          class_index < predictions.class_count(); class_index++ )
    {
        names.emplace_back(predictions.class_name(class_index));
    }
    return names;
}

std::vector<std::size_t> prediction_class_indices(
    const c50::predictions &predictions)
{
    std::vector<std::size_t> indices;

    indices.reserve(predictions.size());
    for ( std::size_t row = 0; row < predictions.size(); row++ )
    {
        indices.push_back(predictions.class_index(row));
    }
    return indices;
}

std::vector<std::string> prediction_labels(
    const c50::predictions &predictions)
{
    std::vector<std::string> labels;

    labels.reserve(predictions.size());
    for ( std::size_t row = 0; row < predictions.size(); row++ )
    {
        labels.emplace_back(
            predictions.class_name(predictions.class_index(row)));
    }
    return labels;
}

std::vector<double> prediction_confidences(
    const c50::predictions &predictions)
{
    std::vector<double> confidences;

    confidences.reserve(predictions.size());
    for ( std::size_t row = 0; row < predictions.size(); row++ )
    {
        confidences.push_back(predictions.confidence(row));
    }
    return confidences;
}

std::vector<std::vector<double>> prediction_scores(
    const c50::predictions &predictions)
{
    std::vector<std::vector<double>> scores(
        predictions.size(),
        std::vector<double>(predictions.class_count()));

    for ( std::size_t row = 0; row < predictions.size(); row++ )
    {
        for ( std::size_t class_index = 0;
              class_index < predictions.class_count(); class_index++ )
        {
            scores[row][class_index] =
                predictions.score(row, class_index);
        }
    }
    return scores;
}

void translate_c50_exception(std::exception_ptr pointer)
{
    if ( ! pointer ) return;
    try
    {
        std::rethrow_exception(pointer);
    }
    catch ( const c50::exception &exception )
    {
        PyObject *type = nullptr;
        switch ( exception.code() )
        {
            case c50::error_code::invalid_argument:
            case c50::error_code::parse_error:
                type = PyExc_ValueError;
                break;
            case c50::error_code::io_error:
                type = PyExc_OSError;
                break;
            default:
                throw;
        }
        if ( type )
        {
            PyErr_SetString(type, exception.what());
        }
    }
}

void bind_inspection(py::module_ &module)
{
    py::enum_<c50::node_kind>(module, "NodeKind")
        .value("LEAF", c50::node_kind::leaf)
        .value("DISCRETE", c50::node_kind::discrete)
        .value("THRESHOLD", c50::node_kind::threshold)
        .value("SUBSET", c50::node_kind::subset);
    py::enum_<c50::condition_kind>(module, "ConditionKind")
        .value("NOT_APPLICABLE", c50::condition_kind::not_applicable)
        .value("EQUALS", c50::condition_kind::equals)
        .value("LESS_EQUAL", c50::condition_kind::less_equal)
        .value("GREATER", c50::condition_kind::greater)
        .value("IN_SUBSET", c50::condition_kind::in_subset);
    py::class_<c50::split_condition>(module, "SplitCondition",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("feature", &c50::split_condition::feature)
        .def_readonly("kind", &c50::split_condition::kind)
        .def_readonly("cut", &c50::split_condition::cut)
        .def_readonly("values", &c50::split_condition::values)
        .def_readonly("includes_not_applicable", &c50::split_condition::includes_not_applicable);
    py::class_<c50::continuous_threshold>(module, "ContinuousThreshold",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("cut", &c50::continuous_threshold::cut)
        .def_readonly("lower", &c50::continuous_threshold::lower)
        .def_readonly("midpoint", &c50::continuous_threshold::midpoint)
        .def_readonly("upper", &c50::continuous_threshold::upper);
    py::class_<c50::tree_branch>(module, "TreeBranch",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("condition", &c50::tree_branch::condition)
        .def_readonly("child", &c50::tree_branch::child);
    py::class_<c50::tree_node>(module, "TreeNode",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("kind", &c50::tree_node::kind)
        .def_readonly("feature", &c50::tree_node::feature)
        .def_readonly("predicted_class", &c50::tree_node::predicted_class)
        .def_readonly("case_weight", &c50::tree_node::case_weight)
        .def_readonly("class_weights", &c50::tree_node::class_weights)
        .def_readonly("threshold", &c50::tree_node::threshold)
        .def_readonly("branches", &c50::tree_node::branches);
    py::class_<c50::tree_inspection>(module, "TreeInspection",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("nodes", &c50::tree_inspection::nodes)
        .def_readonly("leaf_count", &c50::tree_inspection::leaf_count)
        .def_readonly("supported_leaf_count", &c50::tree_inspection::supported_leaf_count)
        .def_readonly("depth", &c50::tree_inspection::depth)
        .def_readonly("supported_depth", &c50::tree_inspection::supported_depth)
        .def_readonly("feature_use", &c50::tree_inspection::feature_use);
    py::class_<c50::rule_inspection>(module, "RuleInspection",
        "Owned rule metadata in serialized order.")
        .def_readonly("conditions", &c50::rule_inspection::conditions)
        .def_readonly("predicted_class", &c50::rule_inspection::predicted_class)
        .def_readonly("cover", &c50::rule_inspection::cover)
        .def_readonly("correct", &c50::rule_inspection::correct)
        .def_readonly("prior", &c50::rule_inspection::prior)
        .def_readonly("vote", &c50::rule_inspection::vote);
    py::class_<c50::ruleset_inspection>(module, "RulesetInspection",
        "Owned ruleset with ordered rules and a fallback class.")
        .def_readonly("default_class", &c50::ruleset_inspection::default_class)
        .def_readonly("rules", &c50::ruleset_inspection::rules)
        .def_readonly("feature_use", &c50::ruleset_inspection::feature_use);
    py::class_<c50::model_inspection>(module, "ModelInspection",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("kind", &c50::model_inspection::kind)
        .def_readonly("class_names", &c50::model_inspection::class_names)
        .def_readonly("feature_names", &c50::model_inspection::feature_names)
        .def_readonly("trees", &c50::model_inspection::trees)
        .def_readonly("rulesets", &c50::model_inspection::rulesets);
}

} // namespace

PYBIND11_MODULE(_c50, module)
{
    module.doc() = R"doc(
Python bindings for the C5.0 GPL classifier core.

Native operations use independent C++ facade contexts and release the Python
GIL while training, loading, or predicting.
)doc";
    py::register_exception<c50::exception>(module, "C50Error",
                                           PyExc_RuntimeError);
    module.attr("C50Error").attr("__doc__") =
        "Base exception for native C5.0 failures not mapped to a built-in "
        "Python exception.";
    py::register_exception_translator(&translate_c50_exception);

    py::enum_<c50::model_kind>(
        module, "ModelKind",
        "Serialized classifier representations supported by C5.0.")
        .value("TREE", c50::model_kind::tree,
               "A decision tree or boosted tree ensemble.")
        .value("RULES", c50::model_kind::rules,
               "A ruleset or boosted ruleset ensemble.");

    bind_inspection(module);

    py::class_<c50::options>(
        module, "Options",
        "Mutable classifier-construction options. Values are validated when "
        "training begins.")
        .def(py::init<>(), "Construct deterministic single-tree defaults.")
        .def_readwrite(
            "trials", &c50::options::trials,
            "Number of classifiers to construct, from 1 through 1000.")
        .def_readwrite(
            "subset_splits", &c50::options::subset_splits,
            "Whether discrete attributes may use subset splits.")
        .def_readwrite(
            "winnow", &c50::options::winnow,
            "Whether to winnow attributes before training.")
        .def_readwrite(
            "global_pruning", &c50::options::global_pruning,
            "Whether to perform global tree pruning.")
        .def_readwrite(
            "probabilistic_thresholds", &c50::options::probabilistic_thresholds,
            "Whether to use probabilistic continuous-value thresholds.")
        .def_readwrite(
            "ignore_costs", &c50::options::ignore_costs,
            "Whether training ignores supplied misclassification costs.")
        .def_readwrite(
            "minimum_cases", &c50::options::minimum_cases,
            "Minimum cases represented by two branches, from 1 through "
            "1000000.")
        .def_readwrite(
            "confidence_factor", &c50::options::confidence_factor,
            "Pruning confidence factor expressed as a fraction in [0, 1].")
        .def_readwrite(
            "sample_fraction", &c50::options::sample_fraction,
            "Training sample fraction in [0, 0.999], or zero for no "
            "sampling.")
        .def_readwrite(
            "random_seed", &c50::options::random_seed,
            "Sampling seed in the range 0 through 4095.");

    py::class_<c50::predictions>(
        module, "Predictions",
        "Owned results for one prediction batch.")
        .def("__len__", &c50::predictions::size,
             "Return the number of predicted rows.")
        .def_property_readonly(
            "class_names", &prediction_class_names,
            "Class names in score-column order.")
        .def_property_readonly(
            "class_indices", &prediction_class_indices,
            "Zero-based predicted class index for each row.")
        .def_property_readonly(
            "labels", &prediction_labels,
            "Predicted class label for each row.")
        .def_property_readonly(
            "confidences", &prediction_confidences,
            "C5.0 confidence in the predicted class for each row.")
        .def_property_readonly(
            "scores", &prediction_scores,
            "Per-class scores as rows in class_names order.");

    py::class_<python_predictor>(module, "Predictor",
        "Owned parsed classifier for repeated prediction. Calls on one instance "
        "are serialized while the GIL is released. Create with Model.prepare_predictor().")
        .def("predict_details", &python_predictor::predict_details,
             predict_details_doc, py::arg("cases"))
        .def("predict_details_dense", &python_predictor::predict_details_dense,
             predict_details_dense_doc, py::arg("values"));

    py::class_<python_model>(
        module, "Model",
        "Immutable, pickleable C5.0 classifier with move-only native "
        "ownership.")
        .def_static("train", &python_model::train,
                    train_doc,
                    py::arg("names"), py::arg("training_data"),
                    py::arg("kind") = c50::model_kind::tree,
                    py::arg("options") = c50::options(),
                    py::arg("costs") = "",
                    py::arg("split_workers") = 1)
        .def_static("train_dense", &python_model::train_dense,
                    train_dense_doc,
                    py::arg("names"), py::arg("values"),
                    py::arg("class_indices"),
                    py::arg("kind") = c50::model_kind::tree,
                    py::arg("options") = c50::options(),
                    py::arg("costs") = "",
                    py::arg("split_workers") = 1)
        .def_static("load", &python_model::load,
                    load_doc,
                    py::arg("names"), py::arg("serialized_data"),
                    py::arg("kind") = c50::model_kind::tree,
                    py::arg("costs") = "")
        .def_property_readonly(
            "kind", &python_model::kind,
            "Serialized classifier representation.")
        .def_property_readonly(
            "names_data", &python_model::names_data,
            "Retained C5.0 names-file contents.")
        .def_property_readonly("serialized_data",
                               &python_model::serialized_data,
                               "Serialized tree or rules-file contents.")
        .def_property_readonly(
            "costs_data", &python_model::costs_data,
            "Retained costs-file contents, or an empty string.")
        .def_property_readonly(
            "classes_", &python_model::classes,
            "Class names in score-column order.")
        .def("prepare_predictor", &python_model::prepare_predictor,
             "Create an independent parsed classifier for repeated prediction calls.")
        .def("inspect", &python_model::inspect,
             "Return owned structural metadata parsed from the retained classifier.")
        .def("predict_details", &python_model::predict_details,
             predict_details_doc, py::arg("cases"))
        .def("predict_details_dense", &python_model::predict_details_dense,
             predict_details_dense_doc, py::arg("values"))
        .def("predict", &python_model::predict,
             predict_doc, py::arg("cases"))
        .def("predict_proba", &python_model::predict_proba,
             predict_proba_doc, py::arg("cases"))
        .def(py::pickle(
            [](const python_model &model) {
                return py::make_tuple(model.kind(), model.names_data(),
                                      model.serialized_data(),
                                      model.costs_data());
            },
            [](const py::tuple &state) {
                if ( state.size() != 4 )
                {
                    throw py::value_error("invalid C5.0 model state");
                }
                return python_model::load(
                    state[1].cast<std::string>(),
                    state[2].cast<std::string>(),
                    state[0].cast<c50::model_kind>(),
                    state[3].cast<std::string>());
            }));

    module.def("train", &python_model::train,
               train_doc,
               py::arg("names"), py::arg("training_data"),
               py::arg("kind") = c50::model_kind::tree,
               py::arg("options") = c50::options(),
               py::arg("costs") = "",
               py::arg("split_workers") = 1);
    module.def("train_dense", &python_model::train_dense,
               train_dense_doc,
               py::arg("names"), py::arg("values"),
               py::arg("class_indices"),
               py::arg("kind") = c50::model_kind::tree,
               py::arg("options") = c50::options(),
               py::arg("costs") = "",
               py::arg("split_workers") = 1);
    module.def("load", &python_model::load,
               load_doc,
               py::arg("names"), py::arg("serialized_data"),
               py::arg("kind") = c50::model_kind::tree,
               py::arg("costs") = "");
}
