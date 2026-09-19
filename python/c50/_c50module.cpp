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
    module.doc() = "Bindings for the C5.0 GPL core.";
    py::register_exception<c50::exception>(module, "C50Error",
                                           PyExc_RuntimeError);
    py::register_exception_translator(&translate_c50_exception);

    py::enum_<c50::model_kind>(module, "ModelKind")
        .value("TREE", c50::model_kind::tree)
        .value("RULES", c50::model_kind::rules);

    py::class_<c50::options>(module, "Options")
        .def(py::init<>())
        .def_property(
            "trials",
            [](const c50::options &options) { return options.trials(); },
            [](c50::options &options, unsigned int value) {
                options.trials(value);
            })
        .def_property(
            "subset_splits",
            [](const c50::options &options) {
                return options.subset_splits();
            },
            [](c50::options &options, bool value) {
                options.subset_splits(value);
            })
        .def_property(
            "winnow",
            [](const c50::options &options) { return options.winnow(); },
            [](c50::options &options, bool value) { options.winnow(value); })
        .def_property(
            "global_pruning",
            [](const c50::options &options) {
                return options.global_pruning();
            },
            [](c50::options &options, bool value) {
                options.global_pruning(value);
            })
        .def_property(
            "probabilistic_thresholds",
            [](const c50::options &options) {
                return options.probabilistic_thresholds();
            },
            [](c50::options &options, bool value) {
                options.probabilistic_thresholds(value);
            })
        .def_property(
            "ignore_costs",
            [](const c50::options &options) {
                return options.ignore_costs();
            },
            [](c50::options &options, bool value) {
                options.ignore_costs(value);
            })
        .def_property(
            "minimum_cases",
            [](const c50::options &options) {
                return options.minimum_cases();
            },
            [](c50::options &options, double value) {
                options.minimum_cases(value);
            })
        .def_property(
            "confidence_factor",
            [](const c50::options &options) {
                return options.confidence_factor();
            },
            [](c50::options &options, double value) {
                options.confidence_factor(value);
            })
        .def_property(
            "sample_fraction",
            [](const c50::options &options) {
                return options.sample_fraction();
            },
            [](c50::options &options, double value) {
                options.sample_fraction(value);
            })
        .def_property(
            "random_seed",
            [](const c50::options &options) {
                return options.random_seed();
            },
            [](c50::options &options, unsigned int value) {
                options.random_seed(value);
            });

    py::class_<c50::predictions>(module, "Predictions")
        .def("__len__", &c50::predictions::size)
        .def_property_readonly("class_names", &prediction_class_names)
        .def_property_readonly("class_indices", &prediction_class_indices)
        .def_property_readonly("labels", &prediction_labels)
        .def_property_readonly("confidences", &prediction_confidences)
        .def_property_readonly("scores", &prediction_scores);

    py::class_<python_model>(module, "Model")
        .def_static("train", &python_model::train,
                    py::arg("names"), py::arg("training_data"),
                    py::arg("kind") = c50::model_kind::tree,
                    py::arg("options") = c50::options(),
                    py::arg("costs") = "")
        .def_static("load", &python_model::load,
                    py::arg("names"), py::arg("serialized_data"),
                    py::arg("kind") = c50::model_kind::tree,
                    py::arg("costs") = "")
        .def_property_readonly("kind", &python_model::kind)
        .def_property_readonly("names_data", &python_model::names_data)
        .def_property_readonly("serialized_data",
                               &python_model::serialized_data)
        .def_property_readonly("costs_data", &python_model::costs_data)
        .def_property_readonly("classes_", &python_model::classes)
        .def("predict_details", &python_model::predict_details,
             py::arg("cases"))
        .def("predict", &python_model::predict, py::arg("cases"))
        .def("predict_proba", &python_model::predict_proba,
             py::arg("cases"))
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
               py::arg("names"), py::arg("training_data"),
               py::arg("kind") = c50::model_kind::tree,
               py::arg("options") = c50::options(),
               py::arg("costs") = "");
    module.def("load", &python_model::load,
               py::arg("names"), py::arg("serialized_data"),
               py::arg("kind") = c50::model_kind::tree,
               py::arg("costs") = "");
}
