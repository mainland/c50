/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c50/c50.hpp
 * @brief Header-only C++11 facade over the compiled C5.0 C library.
 *
 * The facade supplies RAII ownership, move-only model and result types, string
 * overloads, and exception-based error reporting. It contains no learning or
 * prediction logic and must be linked with the C5.0 C core.
 */

#ifndef C50_C50_HPP
#define C50_C50_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <c50/c50.h>

/** @brief Header-only C++ facade for the C5.0 C API. */
namespace c50
{

/** @brief Serialized classifier representations supported by C5.0. */
enum class model_kind
{
    /** A decision tree or boosted ensemble of decision trees. */
    tree = C50_MODEL_TREE,
    /** A ruleset or boosted ensemble of rulesets. */
    rules = C50_MODEL_RULES
};

/**
 * @brief Convert a C++ model kind to the corresponding C API value.
 * @param[in] kind C++ model kind.
 * @return The equivalent c50_model_kind value.
 */
inline c50_model_kind native_model_kind(model_kind kind) noexcept
{
    return static_cast<c50_model_kind>(kind);
}

/** @brief Error reported by a failed C5.0 C API operation. */
class exception : public std::runtime_error
{
public:
    /**
     * @brief Construct an exception using the status's standard message.
     * @param[in] status Native failure status.
     */
    explicit exception(c50_status status)
        : std::runtime_error(c50_status_message(status)), status_(status)
    {
    }

    /**
     * @brief Construct an exception with operation-specific diagnostic text.
     * @param[in] status Native failure status.
     * @param[in] message Diagnostic text, or null to use the standard message.
     */
    exception(c50_status status, const char *message)
        : std::runtime_error(message && message[0] ?
                             message : c50_status_message(status)),
          status_(status)
    {
    }

    /**
     * @brief Return the native status that caused the exception.
     * @return The status value supplied at construction.
     */
    c50_status status() const noexcept
    {
        return status_;
    }

private:
    c50_status status_;
};

/**
 * @brief Move-only owner of a native C5.0 operation context.
 *
 * Independent contexts may be used concurrently. Calls that use one context
 * must be serialized. A successful operation clears any earlier error recorded
 * by the context.
 */
class context
{
public:
    /**
     * @brief Allocate an independent native context.
     * @throws c50::exception if allocation fails.
     */
    context() : handle_(nullptr)
    {
        const c50_status status = c50_context_create(&handle_);
        if ( status != C50_STATUS_OK ) throw exception(status);
    }

    /** @brief Release the owned native context. */
    ~context()
    {
        c50_context_destroy(handle_);
    }

    /** @brief Contexts cannot be copied. */
    context(const context &) = delete;
    /** @brief Contexts cannot be copy-assigned. */
    context &operator=(const context &) = delete;

    /**
     * @brief Transfer ownership from another context.
     * @param[in,out] other Context whose handle is transferred.
     */
    context(context &&other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    /**
     * @brief Replace this context by taking ownership from another context.
     * @param[in,out] other Context whose handle is transferred.
     * @return This context.
     */
    context &operator=(context &&other) noexcept
    {
        if ( this != &other )
        {
            c50_context_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Return the mutable native handle without transferring ownership.
     * @return The owned handle.
     * @warning Calls made through this handle must preserve the C API's
     *          ownership and one-active-operation rules.
     */
    c50_context *native_handle() noexcept
    {
        return handle_;
    }

    /**
     * @brief Return the const native handle without transferring ownership.
     * @return The owned handle.
     */
    const c50_context *native_handle() const noexcept
    {
        return handle_;
    }

    /**
     * @brief Return the status of the most recent native operation.
     * @return The context's current native status.
     */
    c50_status last_status() const noexcept
    {
        return c50_context_last_status(handle_);
    }

    /**
     * @brief Return diagnostic detail for the most recent native operation.
     * @return A borrowed, non-null string that remains valid until the next
     *         operation or destruction of this context.
     */
    const char *error_message() const noexcept
    {
        return c50_context_error_message(handle_);
    }

private:
    c50_context *handle_;
};

/**
 * @brief Value object containing classifier-construction options.
 *
 * Construction selects the defaults returned by c50_options_init(). Setters do
 * not validate their arguments. model::train() validates the complete option
 * set before starting native training.
 */
class options
{
public:
    /** @brief Construct the deterministic single-tree defaults. */
    options()
    {
        c50_options_init(&options_);
    }

    /**
     * @brief Return the number of classifiers to construct.
     * @return The configured number of trials.
     */
    unsigned int trials() const noexcept { return options_.trials; }
    /**
     * @brief Set the number of classifiers to construct.
     * @param[in] value Number of trials, from 1 through 1000.
     * @return This options object for method chaining.
     */
    options &trials(unsigned int value) noexcept
    {
        options_.trials = value;
        return *this;
    }

    /**
     * @brief Return whether discrete attributes may use subset splits.
     * @return Whether subset splits are enabled.
     */
    bool subset_splits() const noexcept { return options_.subset_splits != 0; }
    /**
     * @brief Enable or disable subset splits for discrete attributes.
     * @param[in] value Whether subset splits are enabled.
     * @return This options object for method chaining.
     */
    options &subset_splits(bool value) noexcept
    {
        options_.subset_splits = value;
        return *this;
    }

    /**
     * @brief Return whether attribute winnowing is enabled.
     * @return Whether winnowing is enabled.
     */
    bool winnow() const noexcept { return options_.winnow != 0; }
    /**
     * @brief Enable or disable attribute winnowing.
     * @param[in] value Whether winnowing is enabled.
     * @return This options object for method chaining.
     */
    options &winnow(bool value) noexcept
    {
        options_.winnow = value;
        return *this;
    }

    /**
     * @brief Return whether global tree pruning is enabled.
     * @return Whether global pruning is enabled.
     */
    bool global_pruning() const noexcept
    {
        return options_.global_pruning != 0;
    }
    /**
     * @brief Enable or disable global tree pruning.
     * @param[in] value Whether global pruning is enabled.
     * @return This options object for method chaining.
     */
    options &global_pruning(bool value) noexcept
    {
        options_.global_pruning = value;
        return *this;
    }

    /**
     * @brief Return whether probabilistic thresholds are enabled.
     * @return Whether probabilistic thresholds are enabled.
     */
    bool probabilistic_thresholds() const noexcept
    {
        return options_.probabilistic_thresholds != 0;
    }
    /**
     * @brief Enable or disable probabilistic continuous-value thresholds.
     * @param[in] value Whether probabilistic thresholds are enabled.
     * @return This options object for method chaining.
     */
    options &probabilistic_thresholds(bool value) noexcept
    {
        options_.probabilistic_thresholds = value;
        return *this;
    }

    /**
     * @brief Return whether supplied misclassification costs are ignored.
     * @return Whether supplied costs are ignored.
     */
    bool ignore_costs() const noexcept { return options_.ignore_costs != 0; }
    /**
     * @brief Select whether training ignores supplied costs.
     * @param[in] value Whether supplied costs are ignored.
     * @return This options object for method chaining.
     */
    options &ignore_costs(bool value) noexcept
    {
        options_.ignore_costs = value;
        return *this;
    }

    /**
     * @brief Return the minimum cases represented by two branches.
     * @return The configured minimum case count.
     */
    double minimum_cases() const noexcept { return options_.minimum_cases; }
    /**
     * @brief Set the minimum cases represented by two branches.
     * @param[in] value Minimum cases, from 1 through 1000000.
     * @return This options object for method chaining.
     */
    options &minimum_cases(double value) noexcept
    {
        options_.minimum_cases = value;
        return *this;
    }

    /**
     * @brief Return the pruning confidence factor as a fraction.
     * @return The configured confidence factor.
     */
    double confidence_factor() const noexcept
    {
        return options_.confidence_factor;
    }
    /**
     * @brief Set the pruning confidence factor.
     * @param[in] value Fraction in the closed interval [0, 1].
     * @return This options object for method chaining.
     */
    options &confidence_factor(double value) noexcept
    {
        options_.confidence_factor = value;
        return *this;
    }

    /**
     * @brief Return the training sample fraction.
     * @return The configured sample fraction.
     */
    double sample_fraction() const noexcept
    {
        return options_.sample_fraction;
    }
    /**
     * @brief Set the training sample fraction.
     * @param[in] value Fraction in [0, 0.999], or zero for no sampling.
     * @return This options object for method chaining.
     */
    options &sample_fraction(double value) noexcept
    {
        options_.sample_fraction = value;
        return *this;
    }

    /**
     * @brief Return the sampling seed.
     * @return The configured random seed.
     */
    unsigned int random_seed() const noexcept { return options_.random_seed; }
    /**
     * @brief Set the sampling seed.
     * @param[in] value Seed in the range 0 through 4095.
     * @return This options object for method chaining.
     */
    options &random_seed(unsigned int value) noexcept
    {
        options_.random_seed = value;
        return *this;
    }

    /**
     * @brief Return the mutable native options without transferring ownership.
     * @return A pointer to the native options owned by this object.
     * @warning Preserve `struct_size` and keep reserved fields zero.
     */
    c50_options *native_handle() noexcept { return &options_; }
    /**
     * @brief Return the const native options without transferring ownership.
     * @return A pointer to the native options owned by this object.
     */
    const c50_options *native_handle() const noexcept { return &options_; }

private:
    c50_options options_;
};

/**
 * @brief Move-only owner of one batch of prediction results.
 *
 * Instances are returned by model::predict(). They retain class metadata after
 * the originating context or model is destroyed. Concurrent const access is
 * safe while the object remains alive.
 */
class predictions
{
public:
    /** @brief Release the owned native prediction results. */
    ~predictions()
    {
        c50_predictions_destroy(handle_);
    }

    /** @brief Prediction results cannot be copied. */
    predictions(const predictions &) = delete;
    /** @brief Prediction results cannot be copy-assigned. */
    predictions &operator=(const predictions &) = delete;

    /**
     * @brief Transfer ownership from another result object.
     * @param[in,out] other Results whose handle is transferred.
     */
    predictions(predictions &&other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    /**
     * @brief Replace these results by taking ownership from another object.
     * @param[in,out] other Results whose handle is transferred.
     * @return This result object.
     */
    predictions &operator=(predictions &&other) noexcept
    {
        if ( this != &other )
        {
            c50_predictions_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Return the number of predicted rows.
     * @return The row count.
     */
    std::size_t size() const noexcept
    {
        return c50_predictions_row_count(handle_);
    }

    /**
     * @brief Return the number of classes represented by each row.
     * @return The class count.
     */
    std::size_t class_count() const noexcept
    {
        return c50_predictions_class_count(handle_);
    }

    /**
     * @brief Borrow a class name from the result metadata.
     * @param[in] class_index Zero-based class index.
     * @return A null-terminated name owned by this object, or null when the
     *         index is out of range.
     */
    const char *class_name(std::size_t class_index) const noexcept
    {
        return c50_predictions_class_name(handle_, class_index);
    }

    /**
     * @brief Return the predicted class for a row.
     * @param[in] row Zero-based row index.
     * @return A zero-based class index, or `std::size_t(-1)` when out of range.
     */
    std::size_t class_index(std::size_t row) const noexcept
    {
        return c50_predictions_class_index(handle_, row);
    }

    /**
     * @brief Return C5.0's confidence in the predicted class for a row.
     * @param[in] row Zero-based row index.
     * @return The confidence, or zero when @p row is out of range.
     */
    double confidence(std::size_t row) const noexcept
    {
        return c50_predictions_confidence(handle_, row);
    }

    /**
     * @brief Return C5.0's score for one row and class.
     * @param[in] row Zero-based row index.
     * @param[in] class_index Zero-based class index.
     * @return The score, or zero when either index is out of range.
     */
    double score(std::size_t row, std::size_t class_index) const noexcept
    {
        return c50_predictions_score(handle_, row, class_index);
    }

    /**
     * @brief Return the mutable native handle without transferring ownership.
     * @return The native handle owned by this object.
     * @warning Do not destroy the returned handle.
     */
    c50_predictions *native_handle() noexcept { return handle_; }
    /**
     * @brief Return the const native handle without transferring ownership.
     * @return The native handle owned by this object.
     */
    const c50_predictions *native_handle() const noexcept { return handle_; }

private:
    friend class model;

    explicit predictions(c50_predictions *handle) noexcept : handle_(handle) {}

    c50_predictions *handle_;
};

/**
 * @brief Move-only owner of an immutable C5.0 classifier.
 *
 * A model owns its names data, serialized classifier, and applicable costs
 * data. It is independent of the context used to create it. The same model may
 * be used by concurrent predictions when each operation has a different
 * context and the model remains alive until all operations finish.
 */
class model
{
public:
    /** @brief Release the owned native model. */
    ~model()
    {
        c50_model_destroy(handle_);
    }

    /** @brief Models cannot be copied. */
    model(const model &) = delete;
    /** @brief Models cannot be copy-assigned. */
    model &operator=(const model &) = delete;

    /**
     * @brief Transfer ownership from another model.
     * @param[in,out] other Model whose handle is transferred.
     */
    model(model &&other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    /**
     * @brief Replace this model by taking ownership from another model.
     * @param[in,out] other Model whose handle is transferred.
     * @return This model.
     */
    model &operator=(model &&other) noexcept
    {
        if ( this != &other )
        {
            c50_model_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Train a classifier from length-delimited C5.0 text buffers.
     *
     * Required buffers must be non-null and nonempty. Text buffers must not
     * contain embedded null bytes. The returned model owns copies of retained
     * input data and is independent of @p ctx.
     *
     * @param[in,out] ctx Context used for training and diagnostics.
     * @param[in] kind Representation to construct.
     * @param[in] names_data C5.0 names-file contents.
     * @param[in] names_size Number of bytes in @p names_data.
     * @param[in] training_data C5.0 data-file contents.
     * @param[in] training_size Number of bytes in @p training_data.
     * @param[in] training_options Classifier-construction options.
     * @param[in] costs_data Optional C5.0 costs-file contents.
     * @param[in] costs_size Number of bytes in @p costs_data.
     * @return An owned, immutable model.
     * @throws c50::exception if validation, allocation, parsing, or training
     *         fails.
     */
    static model train(context &ctx, model_kind kind,
                       const char *names_data, std::size_t names_size,
                       const char *training_data, std::size_t training_size,
                       const options &training_options = options(),
                       const char *costs_data = nullptr,
                       std::size_t costs_size = 0)
    {
        c50_model *handle = nullptr;
        const c50_status status = c50_model_train(
            ctx.native_handle(), native_model_kind(kind),
            training_options.native_handle(), names_data, names_size,
            training_data, training_size, costs_data, costs_size, &handle);
        if ( status != C50_STATUS_OK )
        {
            throw exception(status, ctx.error_message());
        }
        return model(handle);
    }

    /**
     * @brief Train a classifier from C5.0 text stored in strings.
     * @param[in,out] ctx Context used for training and diagnostics.
     * @param[in] kind Representation to construct.
     * @param[in] names_data C5.0 names-file contents.
     * @param[in] training_data C5.0 data-file contents.
     * @param[in] training_options Classifier-construction options.
     * @param[in] costs_data Optional C5.0 costs-file contents.
     * @return An owned, immutable model.
     * @throws c50::exception if validation, allocation, parsing, or training
     *         fails, including when a string contains an embedded null byte.
     */
    static model train(context &ctx, model_kind kind,
                       const std::string &names_data,
                       const std::string &training_data,
                       const options &training_options = options(),
                       const std::string &costs_data = std::string())
    {
        return train(ctx, kind, names_data.data(), names_data.size(),
                     training_data.data(), training_data.size(),
                     training_options,
                     costs_data.empty() ? nullptr : costs_data.data(),
                     costs_data.size());
    }

    /**
     * @brief Load a classifier from length-delimited C5.0 text buffers.
     *
     * Required buffers must be non-null and nonempty. Text buffers must not
     * contain embedded null bytes. The returned model owns copies of all input
     * data and is independent of @p ctx.
     *
     * @param[in,out] ctx Context used for parsing and diagnostics.
     * @param[in] kind Representation contained in @p model_data.
     * @param[in] names_data C5.0 names-file contents.
     * @param[in] names_size Number of bytes in @p names_data.
     * @param[in] model_data Serialized tree or rules-file contents.
     * @param[in] model_size Number of bytes in @p model_data.
     * @param[in] costs_data Optional C5.0 costs-file contents.
     * @param[in] costs_size Number of bytes in @p costs_data.
     * @return An owned, immutable model.
     * @throws c50::exception if validation, allocation, or parsing fails.
     */
    static model load(context &ctx, model_kind kind,
                      const char *names_data, std::size_t names_size,
                      const char *model_data, std::size_t model_size,
                      const char *costs_data = nullptr,
                      std::size_t costs_size = 0)
    {
        c50_model *handle = nullptr;
        const c50_status status = c50_model_load(
            ctx.native_handle(), native_model_kind(kind),
            names_data, names_size, model_data, model_size,
            costs_data, costs_size, &handle);
        if ( status != C50_STATUS_OK )
        {
            throw exception(status, ctx.error_message());
        }
        return model(handle);
    }

    /**
     * @brief Load a classifier from C5.0 text stored in strings.
     * @param[in,out] ctx Context used for parsing and diagnostics.
     * @param[in] kind Representation contained in @p model_data.
     * @param[in] names_data C5.0 names-file contents.
     * @param[in] model_data Serialized tree or rules-file contents.
     * @param[in] costs_data Optional C5.0 costs-file contents.
     * @return An owned, immutable model.
     * @throws c50::exception if validation, allocation, or parsing fails,
     *         including when a string contains an embedded null byte.
     */
    static model load(context &ctx, model_kind kind,
                      const std::string &names_data,
                      const std::string &model_data,
                      const std::string &costs_data = std::string())
    {
        return load(ctx, kind, names_data.data(), names_data.size(),
                    model_data.data(), model_data.size(),
                    costs_data.empty() ? nullptr : costs_data.data(),
                    costs_data.size());
    }

    /**
     * @brief Return the serialized representation kind.
     * @return The model kind.
     */
    model_kind kind() const noexcept
    {
        return static_cast<model_kind>(c50_model_get_kind(handle_));
    }

    /**
     * @brief Return a copy of the retained C5.0 names-file contents.
     * @return The names-file contents.
     */
    std::string names_data() const
    {
        std::size_t size = 0;
        const char *data = c50_model_names_data(handle_, &size);
        return data ? std::string(data, size) : std::string();
    }

    /**
     * @brief Return a copy of the serialized tree or rules-file contents.
     * @return The serialized classifier.
     */
    std::string serialized_data() const
    {
        std::size_t size = 0;
        const char *data = c50_model_serialized_data(handle_, &size);
        return data ? std::string(data, size) : std::string();
    }

    /**
     * @brief Return a copy of the retained costs-file contents.
     * @return An empty string when the model has no retained costs.
     */
    std::string costs_data() const
    {
        std::size_t size = 0;
        const char *data = c50_model_costs_data(handle_, &size);
        return data ? std::string(data, size) : std::string();
    }

    /**
     * @brief Predict cases from a length-delimited C5.0 text buffer.
     *
     * Each nonempty row must contain all input fields followed by a class
     * field, which may be `?`. A null pointer is allowed only for an empty
     * batch. The returned results are independent of this model and @p ctx.
     *
     * @param[in,out] ctx Context used for parsing and prediction.
     * @param[in] cases_data C5.0 case data, or null for an empty batch.
     * @param[in] cases_size Number of bytes in @p cases_data.
     * @return Owned prediction results.
     * @throws c50::exception if validation, allocation, parsing, or prediction
     *         fails.
     */
    predictions predict(context &ctx, const char *cases_data,
                        std::size_t cases_size) const
    {
        c50_predictions *result = nullptr;
        const c50_status status = c50_model_predict(
            ctx.native_handle(), handle_, cases_data, cases_size, &result);
        if ( status != C50_STATUS_OK )
        {
            throw exception(status, ctx.error_message());
        }
        return predictions(result);
    }

    /**
     * @brief Predict cases from C5.0 text stored in a string.
     * @param[in,out] ctx Context used for parsing and prediction.
     * @param[in] cases_data C5.0 case data.
     * @return Owned prediction results.
     * @throws c50::exception if validation, allocation, parsing, or prediction
     *         fails, including when @p cases_data contains an embedded null.
     */
    predictions predict(context &ctx, const std::string &cases_data) const
    {
        return predict(ctx, cases_data.data(), cases_data.size());
    }

    /**
     * @brief Return the mutable native handle without transferring ownership.
     * @return The native handle owned by this object.
     * @warning Do not mutate or destroy the returned handle.
     */
    c50_model *native_handle() noexcept { return handle_; }
    /**
     * @brief Return the const native handle without transferring ownership.
     * @return The native handle owned by this object.
     */
    const c50_model *native_handle() const noexcept { return handle_; }

private:
    explicit model(c50_model *handle) noexcept : handle_(handle) {}

    c50_model *handle_;
};

} // namespace c50

#endif
