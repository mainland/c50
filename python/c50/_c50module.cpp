/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <exception>
#include <cstdint>
#include <limits>
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
