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
 * Every operation reads and writes numbers in the C locale on the calling
 * thread and on its split workers, so results do not depend on the host
 * locale. The caller's thread locale is restored when the call returns or
 * throws, and the process locale is never changed.
 * Destruction and move assignment require exclusive access to the object.
 *
 * Invalid arguments and input errors throw c50::exception. Allocation failures
 * throw std::bad_alloc. Invalid result indices throw std::out_of_range.
 * Training with more than one split worker throws std::system_error, which
 * is not a c50::exception, when the system cannot start a worker thread.
 * A failed operation leaves its context reusable. Moved-from objects may only
 * be destroyed or assigned another object.
 *
 * Training, parsing, and classification recurse once per tree level on the
 * calling thread's stack, which the caller must size. model::load accepts
 * trees of at most load_options::max_tree_depth levels, 1,024 by default.
 * Models that train() returned, and their moves, parse without a limit. See
 * docs/ownership-concurrency.md for the stack that each level needs.
 */
#ifndef C50_C50_HPP
#define C50_C50_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace c50 {
namespace detail {
struct context_state;
struct model_access;
struct model_data;
struct prediction_data;
struct predictor_data;
}

/** Classifier representation, including boosted ensembles. */
enum class model_kind { tree, /**< Decision tree or tree ensemble. */
                        rules /**< Ruleset or ruleset ensemble. */ };

/** Failure categories reported by the native library. */
enum class error_code {
    invalid_argument, /**< Invalid option or input, or missing required costs. */
    io_error,         /**< Stream failure. */
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

/** Order of equal continuous values when evaluating splits. */
enum class tie_order {
    reference, /**< Reproduce the tie order of the reference C5.0 learner. */
    stable     /**< Keep equal values in case order with a faster sort.
                    Classifiers are deterministic on every platform but may
                    differ from the reference learner. */
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
    tie_order ties = tie_order::reference; /**< Split-evaluation tie order. */
};

/**
 * Borrowed row-major dense data. Arrays must remain valid throughout the call.
 * Discrete values are zero-based schema indices. NaN denotes missing values.
 * The not_applicable mask denotes not-applicable values, which are distinct
 * from missing values. A schema with an implicit, ignored, label, date, time,
 * or timestamp attribute, a class attribute, or a discrete attribute without
 * explicit values requires the text interface. Each dense operation rejects
 * such a schema with error_code::unsupported, whatever the feature count.
 * The caller must provide storage for all addressed elements.
 */
struct dense_dataset {
    const double *values = nullptr;      /**< Feature data, null only for zero rows. */
    std::size_t row_count = 0;            /**< Training requires at least one row. */
    std::size_t feature_count = 0;        /**< Must match the explicit schema. */
    const std::size_t *class_indices = nullptr; /**< Zero-based labels, required for training. */
    std::size_t row_stride = 0;          /**< Elements between rows, zero means feature_count. */
    /** Borrowed row-major not-applicable mask, or null for none.
     *
     * The mask has row_count * feature_count bytes with no padding, whatever
     * row_stride is, and must remain valid and unmodified throughout each
     * call that reads the dataset. The dataset does not own it, and the
     * library never writes it. A nonzero byte makes the value of that row
     * and feature not applicable, exactly as "N/A" does in the text
     * interface. The corresponding element of values is then ignored,
     * whatever it holds, including NaN and values that are not finite or
     * are outside the schema. A zero byte leaves the value to values, where
     * NaN is missing. The mask has no entry for the class, which is never
     * not applicable. Reading the mask never modifies the dataset, so
     * concurrent operations may share one mask if none writes it. */
    const unsigned char *not_applicable = nullptr;

    /** Construct an empty borrowed dataset. */
    dense_dataset() = default;
    /** Construct a borrowed view.
     * @param values Feature storage, borrowed for each operation.
     * @param rows Row count.
     * @param features Feature count.
     * @param classes Zero-based training labels, or null for prediction.
     * @param stride Elements between rows, zero means features.
     * @param mask Borrowed not-applicable mask of rows * features bytes, or
     *        null for none. See not_applicable. */
    dense_dataset(const double *values, std::size_t rows, std::size_t features,
                  const std::size_t *classes = nullptr, std::size_t stride = 0,
                  const unsigned char *mask = nullptr);
};

/** Move-only owner of mutable operation workspace. */
class context {
public:
    /** Create an independent workspace using one split worker.
     * @throws exception with error_code::internal_error when the platform
     *         cannot create the C locale that the workspace owns.
     * @throws std::bad_alloc when memory is exhausted. */
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
    /** Set the maximum split worker count.
     * @param count Maximum workers, from 1 to 8. */
    void split_workers(unsigned count);
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
    friend class predictor;
    explicit predictions(std::unique_ptr<detail::prediction_data> data);
    std::unique_ptr<detail::prediction_data> data_;
};

/**
 * Move-only owner of a parsed classifier and mutable prediction workspace.
 * It may outlive its source model. Callers must serialize operations on each
 * predictor. Independent predictors may run concurrently. Results own their
 * data. Failed predictions leave this object reusable, reparsing on demand.
 * The first prediction after a failure can throw exception with
 * error_code::internal_error when the platform cannot create the C locale
 * that the new workspace owns.
 */
class predictor {
public:
    /** Release retained classifier and workspace storage. */
    ~predictor();
    /** Transfer ownership. @param other Source owner. */
    predictor(predictor &&other) noexcept;
    /** Transfer ownership. @param other Source owner. @return This object. */
    predictor &operator=(predictor &&other) noexcept;
    /** Copying is disabled. */
    predictor(const predictor &) = delete;
    /** Copy assignment is disabled. */
    predictor &operator=(const predictor &) = delete;
    /** Predict from borrowed data-file contents, possibly empty.
     *
     * The call ends at the first bad value and throws exception with
     * error_code::parse_error and the line number. The c5.0 program instead
     * reports each bad value, replaces it with "?", and continues until ten
     * errors.
     *
     * @param cases Data-file contents, borrowed for this call.
     * @return An independently owned prediction batch. */
    predictions predict(std::string_view cases);
    /** Predict from borrowed dense features. Class indices are ignored.
     * @param cases Dense features, borrowed for this call.
     * @return An independently owned prediction batch. */
    predictions predict(const dense_dataset &cases);
private:
    friend class model;
    explicit predictor(std::unique_ptr<detail::predictor_data> data);
    std::unique_ptr<detail::predictor_data> data_;
};

/** Tree node representation in an owned inspection snapshot. */
enum class node_kind {
    leaf,      /**< Terminal class distribution. */
    discrete,  /**< One branch per categorical value, including N/A. */
    threshold, /**< Continuous cutoff, possibly with soft thresholds. */
    subset     /**< Branches containing subsets of categorical values. */
};

/** A tree branch or rule condition, independent of private node encodings. */
enum class condition_kind {
    not_applicable, /**< The feature has the distinct N/A value. */
    equals,        /**< The feature equals the one named category. */
    less_equal,    /**< The feature is at or below the cutoff. */
    greater,       /**< The feature is above the cutoff. */
    in_subset      /**< The feature belongs to the named subset. */
};

/** Owned condition metadata. Missing values follow native prediction policy. */
struct split_condition {
    std::size_t feature = 0; /**< Zero-based schema attribute index. */
    condition_kind kind = condition_kind::not_applicable; /**< Comparison kind. */
    double cut = 0; /**< Cutoff for less_equal and greater, otherwise zero. */
    std::vector<std::string> values; /**< Ordinary categories for equals/in_subset. */
    bool includes_not_applicable = false; /**< N/A membership for in_subset only. */
};

/** Continuous threshold points as read from the retained classifier. */
struct continuous_threshold {
    double cut = 0;      /**< Printed decision cutoff. */
    double lower = 0;    /**< Lower soft-threshold point. */
    double midpoint = 0; /**< Middle soft-threshold point. */
    double upper = 0;    /**< Upper soft-threshold point. */
};

/** Owned outgoing branch with an index into the containing tree's nodes. */
struct tree_branch {
    split_condition condition; /**< Branch condition in native branch order. */
    std::size_t child = 0; /**< Zero-based child node index. */
};

/** Owned node metadata, not an alternate prediction implementation. */
struct tree_node {
    node_kind kind = node_kind::leaf; /**< Leaf or split representation. */
    std::optional<std::size_t> feature; /**< Tested attribute, absent at a leaf. */
    std::size_t predicted_class = 0; /**< Stored class, not a cost-adjusted prediction. */
    double case_weight = 0; /**< Sum of serialized class weights, possibly zero. */
    std::vector<double> class_weights; /**< Weighted support in class-name order. */
    std::optional<continuous_threshold> threshold; /**< Present only for thresholds. */
    std::vector<tree_branch> branches; /**< Ordered branches, empty at a leaf. */
};

/** Owned tree with root zero and nodes in preorder, including empty leaves. */
struct tree_inspection {
    std::vector<tree_node> nodes; /**< Independently owned node records. */
    std::size_t leaf_count = 0; /**< All leaves, including zero-support leaves. */
    std::size_t supported_leaf_count = 0; /**< Leaves with positive case weight. */
    std::size_t depth = 0; /**< Maximum node depth in edges from the root. */
    std::size_t supported_depth = 0; /**< Maximum positive-support leaf depth. */
    std::vector<std::size_t> feature_use; /**< Split counts in attribute-name order. */
};

/** Owned rule in serialized order, with all its condition metadata. */
struct rule_inspection {
    std::vector<split_condition> conditions; /**< Conjunction in serialized order. */
    std::size_t predicted_class = 0; /**< Zero-based rule conclusion. */
    double cover = 0; /**< Weighted cases covered by the rule. */
    double correct = 0; /**< Weighted covered cases with the concluded class. */
    double prior = 0; /**< Prior reconstructed from serialized cover, correct, and lift. */
    int vote = 0; /**< Native confidence vote in integer thousandths. */
};

/** Owned ruleset, including the fallback class for cases matching no rule. */
struct ruleset_inspection {
    std::size_t default_class = 0; /**< Zero-based fallback class. */
    std::vector<rule_inspection> rules; /**< Rules in serialized order. */
    std::vector<std::size_t> feature_use; /**< Condition counts per schema attribute. */
};

/**
 * Copyable snapshot owning all model-inspection data. Mutations affect only
 * this value. It may outlive its model and context. Concurrent reads are safe
 * when no thread mutates the snapshot. Counts reflect serialization precision.
 */
struct model_inspection {
    model_kind kind = model_kind::tree; /**< Source classifier representation. */
    std::vector<std::string> class_names; /**< Classes in prediction-score order. */
    std::vector<std::string> feature_names; /**< All schema attributes in source order. */
    std::vector<tree_inspection> trees; /**< Tree components, empty for rules models. */
    std::vector<ruleset_inspection> rulesets; /**< Ruleset components in serialized order. */
};

/** The default for load_options::max_tree_depth, in tree levels.
 *
 * A later 1.x release may raise the value but never lowers it, so a model
 * that loads with the default keeps loading. Use this constant, not 1,024, to
 * name the default. */
inline constexpr std::size_t default_max_tree_depth = 1024;

/** Options for model::load. */
struct load_options {
    /** The most levels that a serialized tree may have, or no limit.
     *
     * The root is level 1, so a tree of this many levels is accepted and a
     * deeper one throws exception with error_code::parse_error. The limit
     * bounds the stack that parsing needs when the text is untrusted. It
     * applies to each tree of an ensemble, and a rules model ignores it.
     * A value of zero is invalid and throws exception with
     * error_code::invalid_argument. std::nullopt removes the limit, so that
     * parsing and prediction need stack in proportion to the tree's depth.
     * The model keeps this value and applies it each time that it parses its
     * text again, in predict(), attribute_usage(), inspect(), and
     * prepare_predictor(). */
    std::optional<std::size_t> max_tree_depth = default_max_tree_depth;
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
     *
     * The call ends at the first bad value, such as a discrete value that
     * the names file does not declare or a nonfinite number, and throws
     * exception with error_code::parse_error and the line number. The c5.0
     * program instead reports each bad value, replaces it with "?", and
     * continues until ten errors.
     *
     * @param workspace Exclusive operation workspace.
     * @param kind Classifier representation.
     * @param names Names-file contents.
     * @param training Data-file contents.
     * @param settings Training options.
     * @param costs Optional costs-file contents.
     * @return An independently owned model. It parses its own text without a
     *         tree depth limit, because the library wrote that text. */
    static model train(context &workspace, model_kind kind,
                       std::string_view names, std::string_view training,
                       const options &settings = {}, std::string_view costs = {});
    /** Train from borrowed dense arrays and names/costs file contents.
     * @param workspace Exclusive operation workspace.
     * @param kind Classifier representation.
     * @param names Names-file contents.
     * @param training Borrowed dense training data.
     * @param settings Training options.
     * @param costs Optional costs-file contents.
     * @return An independently owned model. It parses its own text without a
     *         tree depth limit, because the library wrote that text. */
    static model train(context &workspace, model_kind kind,
                       std::string_view names, const dense_dataset &training,
                       const options &settings = {}, std::string_view costs = {});
    /** Validate and copy a serialized legacy classifier, schema, and costs.
     * @param workspace Exclusive operation workspace.
     * @param kind Classifier representation.
     * @param names Names-file contents.
     * @param serialized Tree or rules file contents.
     * @param costs Optional costs-file contents, required by cost-bearing models.
     *        A model without costs ignores them. Missing required costs throw
     *        `error_code::invalid_argument`. Costs text that sets no cost
     *        other than the default of 1 counts as missing. That includes
     *        empty text, text with only comments, and text whose entries
     *        all have cost 1 or name one class twice.
     * @param settings Loading options. The call reads a copy. A zero
     *        `max_tree_depth` throws `error_code::invalid_argument`, and a
     *        tree deeper than a nonzero limit throws
     *        `error_code::parse_error`.
     * @return An independently owned model, which applies the same tree
     *         depth limit whenever it parses its text again. */
    static model load(context &workspace, model_kind kind,
                      std::string_view names, std::string_view serialized,
                      std::string_view costs = {},
                      const load_options &settings = {});
    /** Predict from data-file contents. Empty input returns class metadata.
     *
     * The call ends at the first bad value and throws exception with
     * error_code::parse_error and the line number, as train() does. The c5.0
     * program instead reports each bad value, replaces it with "?", and
     * continues until ten errors.
     *
     * @param workspace Exclusive operation workspace.
     * @param cases Data-file contents, possibly empty.
     * @return An independently owned prediction batch. */
    predictions predict(context &workspace, std::string_view cases) const;
    /** Predict from borrowed dense arrays. Class indices are ignored.
     * @param workspace Exclusive operation workspace.
     * @param cases Borrowed dense features, possibly empty.
     * @return An independently owned prediction batch. */
    predictions predict(context &workspace, const dense_dataset &cases) const;
    /** Count the cases whose classification tests each attribute.
     *
     * A case counts for an attribute when classifying it tests that
     * attribute and the case's value is known. For implicit attributes, the
     * attributes used by the definition also count. These are the counts
     * behind C5.0's "Attribute usage" report, which divides them by the
     * number of cases. The call ends at the first bad value and throws
     * exception with error_code::parse_error, as predict() does.
     *
     * @param workspace Exclusive operation workspace.
     * @param cases Data-file contents, possibly empty.
     * @return One count per attribute of the names data, in declaration
     *         order. */
    std::vector<std::size_t> attribute_usage(context &workspace,
                                             std::string_view cases) const;
    /** Count the cases whose classification tests each attribute.
     * @param workspace Exclusive operation workspace.
     * @param cases Borrowed dense features, possibly empty. Class indices are
     *        ignored.
     * @return One count per attribute of the names data, in declaration
     *         order, as for the data-file overload. */
    std::vector<std::size_t> attribute_usage(context &workspace,
                                             const dense_dataset &cases) const;
    /** Inspect retained classifiers through the validated native loader.
     * @param workspace Exclusive operation workspace.
     * @return A copyable snapshot independent of this model and workspace. */
    model_inspection inspect(context &workspace) const;
    /** Parse an independently owned classifier for repeated predictions.
     * @return A predictor owning both parsed and serialized model data.
     * @throws exception with error_code::internal_error when the platform
     *         cannot create the C locale that the predictor owns, or with
     *         the code of the parse failure. */
    predictor prepare_predictor() const;
    /** @return the classifier representation. */
    model_kind kind() const noexcept;
    /** @return owned names-file contents. The reference follows model lifetime. */
    const std::string &names_data() const noexcept;
    /** @return owned legacy classifier bytes. The reference follows model lifetime. */
    const std::string &serialized_data() const noexcept;
    /** @return the costs that the model uses, empty when it uses none. A
     *         loaded model keeps costs only when its classifier text declares
     *         `costs="1"`, and ignores any others. The reference follows model
     *         lifetime. */
    const std::string &costs_data() const noexcept;
private:
    friend struct detail::model_access;
    explicit model(std::unique_ptr<detail::model_data> data);
    std::unique_ptr<detail::model_data> data_;
};
}
#endif
