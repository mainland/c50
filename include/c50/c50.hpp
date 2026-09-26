/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c50/c50.hpp
 * @brief Compiled C++17 interface to the C5.0 GPL classifier core.
 *
 * Link against C50::cpp. There is no stable binary ABI: consumers must use a
 * compatible C++ toolchain and rebuild when the library changes.
 *
 * Inputs are borrowed only for the duration of a call. Models and prediction
 * results own their data and may outlive the context that produced them.
 * Independent contexts may run concurrently. Callers must serialize operations
 * on each context. Immutable models may be shared by different contexts.
 * Destruction and move assignment require exclusive access to the object.
 *
 * Invalid arguments and input errors throw c50::exception. Allocation failures
 * throw std::bad_alloc. Invalid result indices throw std::out_of_range.
 * A failed operation leaves its context reusable. Moved-from objects may only
 * be destroyed or assigned another object.
 */
#ifndef C50_C50_HPP
#define C50_C50_HPP

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace c50 {
namespace detail {
struct context_state;
struct model_data;
struct prediction_data;
}

/** Classifier representation, including boosted ensembles. */
enum class model_kind { tree, /**< Decision tree or tree ensemble. */
                        rules /**< Ruleset or ruleset ensemble. */ };

/** Failure categories reported by the native library. */
enum class error_code {
    invalid_argument, /**< Invalid option or input descriptor. */
    io_error,         /**< Stream failure or missing required costs. */
    parse_error,      /**< Invalid schema, cases, costs, or model text. */
    unsupported,      /**< Unsupported schema or operation. */
    internal_error    /**< Operation-state or internal invariant failure. */
};

/** An input or operation failure with a diagnostic and category. */
class exception : public std::runtime_error {
public:
    /** Construct a diagnostic with the supplied category.
     * @param code Failure category.
     * @param message Human-readable diagnostic. */
    exception(error_code code, const std::string &message);
    /** @return The failure category. */
    error_code code() const noexcept;
private:
    error_code code_;
};

/** Training options. Values are checked at the start of training. */
struct options {
    unsigned trials = 1;                 /**< Classifiers to build, 1 to 1000. */
    bool subset_splits = false;          /**< Allow discrete value subsets. */
    bool winnow = false;                 /**< Remove unhelpful attributes. */
    bool global_pruning = true;         /**< Apply global tree pruning. */
    bool probabilistic_thresholds = false; /**< Use soft continuous thresholds. */
    bool ignore_costs = false;           /**< Ignore costs in training and prediction. */
    double minimum_cases = 2;            /**< Minimum branch cases, 1 to 1000000. */
    double confidence_factor = 0.25;     /**< Pruning confidence in [0, 1]. */
    double sample_fraction = 0;          /**< Training fraction in [0, 0.999], zero disables sampling. */
    unsigned random_seed = 0;            /**< Sampling seed, 0 to 4095. */
};

/** Move-only owner of mutable operation workspace. */
class context {
public:
    /** Create an independent workspace. */
    context();
    /** Release the workspace. Returned models and results remain valid. */
    ~context();
    /** Transfer workspace ownership.
     * @param other Source owner. */
    context(context &&other) noexcept;
    /** Replace this workspace by transferring ownership.
     * @param other Source owner.
     * @return This object. */
    context &operator=(context &&other) noexcept;
    /** Copying is disabled. */
    context(const context &) = delete;
    /** Copy assignment is disabled. */
    context &operator=(const context &) = delete;
private:
    friend class model;
    std::unique_ptr<detail::context_state> state_;
};

/** Move-only owner of a batch of predictions and class metadata. */
class predictions {
public:
    /** Release the batch. */
    ~predictions();
    /** Transfer batch ownership.
     * @param other Source owner. */
    predictions(predictions &&other) noexcept;
    /** Replace this batch by transferring ownership.
     * @param other Source owner.
     * @return This object. */
    predictions &operator=(predictions &&other) noexcept;
    /** Copying is disabled. */
    predictions(const predictions &) = delete;
    /** Copy assignment is disabled. */
    predictions &operator=(const predictions &) = delete;
    /** @return the number of rows. */
    std::size_t size() const noexcept;
    /** @return the number of classes, including for an empty batch. */
    std::size_t class_count() const noexcept;
    /** @param index Zero-based class index.
     * @return A class name reference, valid for the batch lifetime. */
    const std::string &class_name(std::size_t index) const;
    /** @param row Zero-based row index.
     * @return The zero-based predicted class. */
    std::size_t class_index(std::size_t row) const;
    /** @param row Zero-based row index.
     * @return The core's confidence for the predicted class. */
    double confidence(std::size_t row) const;
    /** @param row Zero-based row index.
     * @param index Zero-based class index.
     * @return A core score, not necessarily a calibrated probability. */
    double score(std::size_t row, std::size_t index) const;
private:
    friend class model;
    explicit predictions(std::unique_ptr<detail::prediction_data> data);
    std::unique_ptr<detail::prediction_data> data_;
};

/** Move-only owner of an immutable classifier and its schema and costs. */
class model {
public:
    /** Release the model. Previously returned predictions remain valid. */
    ~model();
    /** Transfer model ownership.
     * @param other Source owner. */
    model(model &&other) noexcept;
    /** Replace this model by transferring ownership.
     * @param other Source owner.
     * @return This object. */
    model &operator=(model &&other) noexcept;
    /** Copying is disabled. */
    model(const model &) = delete;
    /** Copy assignment is disabled. */
    model &operator=(const model &) = delete;

    /** Train from names/data/costs file contents in memory.
     * @param workspace Exclusive operation workspace.
     * @param kind Classifier representation.
     * @param names Names-file contents.
     * @param training Data-file contents.
     * @param settings Training options.
     * @param costs Optional costs-file contents.
     * @return An independently owned model. */
    static model train(context &workspace, model_kind kind,
                       std::string_view names, std::string_view training,
                       const options &settings = {}, std::string_view costs = {});
    /** Validate and copy a serialized legacy classifier, schema, and costs.
     * @param workspace Exclusive operation workspace.
     * @param kind Classifier representation.
     * @param names Names-file contents.
     * @param serialized Tree or rules file contents.
     * @param costs Optional costs-file contents, required by cost-bearing models.
     * @return An independently owned model. */
    static model load(context &workspace, model_kind kind,
                      std::string_view names, std::string_view serialized,
                      std::string_view costs = {});
    /** Predict from data-file contents. Empty input returns class metadata.
     * @param workspace Exclusive operation workspace.
     * @param cases Data-file contents, possibly empty.
     * @return An independently owned prediction batch. */
    predictions predict(context &workspace, std::string_view cases) const;
    /** @return the classifier representation. */
    model_kind kind() const noexcept;
    /** @return owned names-file contents. The reference follows model lifetime. */
    const std::string &names_data() const noexcept;
    /** @return owned legacy classifier bytes. The reference follows model lifetime. */
    const std::string &serialized_data() const noexcept;
    /** @return retained costs, empty when ignored. The reference follows model lifetime. */
    const std::string &costs_data() const noexcept;
private:
    explicit model(std::unique_ptr<detail::model_data> data);
    std::unique_ptr<detail::model_data> data_;
};
}
#endif
