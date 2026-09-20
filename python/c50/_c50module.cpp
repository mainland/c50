/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <exception>
#include <string>
#include <utility>
#include <vector>

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
)doc";

const char load_doc[] = R"doc(
Load and validate a serialized C5.0 classifier.

The returned model owns copies of the names, serialized classifier, and
optional costs strings.
)doc";

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
                              const std::string &costs)
    {
        c50::context context;
        py::gil_scoped_release release;
        return python_model(c50::model::train(
            context, kind, names, training_data, options, costs));
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
        switch ( exception.status() )
        {
            case C50_STATUS_INVALID_ARGUMENT:
            case C50_STATUS_PARSE_ERROR:
                type = PyExc_ValueError;
                break;
            case C50_STATUS_OUT_OF_MEMORY:
                type = PyExc_MemoryError;
                break;
            case C50_STATUS_IO_ERROR:
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
        .def_property(
            "trials",
            [](const c50::options &options) { return options.trials(); },
            [](c50::options &options, unsigned int value) {
                options.trials(value);
            },
            "Number of classifiers to construct, from 1 through 1000.")
        .def_property(
            "subset_splits",
            [](const c50::options &options) {
                return options.subset_splits();
            },
            [](c50::options &options, bool value) {
                options.subset_splits(value);
            },
            "Whether discrete attributes may use subset splits.")
        .def_property(
            "winnow",
            [](const c50::options &options) { return options.winnow(); },
            [](c50::options &options, bool value) { options.winnow(value); },
            "Whether to winnow attributes before training.")
        .def_property(
            "global_pruning",
            [](const c50::options &options) {
                return options.global_pruning();
            },
            [](c50::options &options, bool value) {
                options.global_pruning(value);
            },
            "Whether to perform global tree pruning.")
        .def_property(
            "probabilistic_thresholds",
            [](const c50::options &options) {
                return options.probabilistic_thresholds();
            },
            [](c50::options &options, bool value) {
                options.probabilistic_thresholds(value);
            },
            "Whether to use probabilistic continuous-value thresholds.")
        .def_property(
            "ignore_costs",
            [](const c50::options &options) {
                return options.ignore_costs();
            },
            [](c50::options &options, bool value) {
                options.ignore_costs(value);
            },
            "Whether training ignores supplied misclassification costs.")
        .def_property(
            "minimum_cases",
            [](const c50::options &options) {
                return options.minimum_cases();
            },
            [](c50::options &options, double value) {
                options.minimum_cases(value);
            },
            "Minimum cases represented by two branches, from 1 through "
            "1000000.")
        .def_property(
            "confidence_factor",
            [](const c50::options &options) {
                return options.confidence_factor();
            },
            [](c50::options &options, double value) {
                options.confidence_factor(value);
            },
            "Pruning confidence factor expressed as a fraction in [0, 1].")
        .def_property(
            "sample_fraction",
            [](const c50::options &options) {
                return options.sample_fraction();
            },
            [](c50::options &options, double value) {
                options.sample_fraction(value);
            },
            "Training sample fraction in [0, 0.999], or zero for no "
            "sampling.")
        .def_property(
            "random_seed",
            [](const c50::options &options) {
                return options.random_seed();
            },
            [](c50::options &options, unsigned int value) {
                options.random_seed(value);
            },
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
                    py::arg("costs") = "")
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
               py::arg("costs") = "");
    module.def("load", &python_model::load,
               load_doc,
               py::arg("names"), py::arg("serialized_data"),
               py::arg("kind") = c50::model_kind::tree,
               py::arg("costs") = "");
}
