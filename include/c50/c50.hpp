/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef C50_C50_HPP
#define C50_C50_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <c50/c50.h>

namespace c50
{

enum class model_kind
{
    tree = C50_MODEL_TREE,
    rules = C50_MODEL_RULES
};

inline c50_model_kind native_model_kind(model_kind kind) noexcept
{
    return static_cast<c50_model_kind>(kind);
}

class exception : public std::runtime_error
{
public:
    explicit exception(c50_status status)
        : std::runtime_error(c50_status_message(status)), status_(status)
    {
    }

    exception(c50_status status, const char *message)
        : std::runtime_error(message && message[0] ?
                             message : c50_status_message(status)),
          status_(status)
    {
    }

    c50_status status() const noexcept
    {
        return status_;
    }

private:
    c50_status status_;
};

class context
{
public:
    context() : handle_(nullptr)
    {
        const c50_status status = c50_context_create(&handle_);
        if ( status != C50_STATUS_OK ) throw exception(status);
    }

    ~context()
    {
        c50_context_destroy(handle_);
    }

    context(const context &) = delete;
    context &operator=(const context &) = delete;

    context(context &&other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

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

    c50_context *native_handle() noexcept
    {
        return handle_;
    }

    const c50_context *native_handle() const noexcept
    {
        return handle_;
    }

    c50_status last_status() const noexcept
    {
        return c50_context_last_status(handle_);
    }

    const char *error_message() const noexcept
    {
        return c50_context_error_message(handle_);
    }

private:
    c50_context *handle_;
};

class options
{
public:
    options()
    {
        c50_options_init(&options_);
    }

    unsigned int trials() const noexcept { return options_.trials; }
    options &trials(unsigned int value) noexcept
    {
        options_.trials = value;
        return *this;
    }

    bool subset_splits() const noexcept { return options_.subset_splits != 0; }
    options &subset_splits(bool value) noexcept
    {
        options_.subset_splits = value;
        return *this;
    }

    bool winnow() const noexcept { return options_.winnow != 0; }
    options &winnow(bool value) noexcept
    {
        options_.winnow = value;
        return *this;
    }

    bool global_pruning() const noexcept
    {
        return options_.global_pruning != 0;
    }
    options &global_pruning(bool value) noexcept
    {
        options_.global_pruning = value;
        return *this;
    }

    bool probabilistic_thresholds() const noexcept
    {
        return options_.probabilistic_thresholds != 0;
    }
    options &probabilistic_thresholds(bool value) noexcept
    {
        options_.probabilistic_thresholds = value;
        return *this;
    }

    bool ignore_costs() const noexcept { return options_.ignore_costs != 0; }
    options &ignore_costs(bool value) noexcept
    {
        options_.ignore_costs = value;
        return *this;
    }

    double minimum_cases() const noexcept { return options_.minimum_cases; }
    options &minimum_cases(double value) noexcept
    {
        options_.minimum_cases = value;
        return *this;
    }

    double confidence_factor() const noexcept
    {
        return options_.confidence_factor;
    }
    options &confidence_factor(double value) noexcept
    {
        options_.confidence_factor = value;
        return *this;
    }

    double sample_fraction() const noexcept
    {
        return options_.sample_fraction;
    }
    options &sample_fraction(double value) noexcept
    {
        options_.sample_fraction = value;
        return *this;
    }

    unsigned int random_seed() const noexcept { return options_.random_seed; }
    options &random_seed(unsigned int value) noexcept
    {
        options_.random_seed = value;
        return *this;
    }

    c50_options *native_handle() noexcept { return &options_; }
    const c50_options *native_handle() const noexcept { return &options_; }

private:
    c50_options options_;
};

class predictions
{
public:
    ~predictions()
    {
        c50_predictions_destroy(handle_);
    }

    predictions(const predictions &) = delete;
    predictions &operator=(const predictions &) = delete;

    predictions(predictions &&other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

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

    std::size_t size() const noexcept
    {
        return c50_predictions_row_count(handle_);
    }

    std::size_t class_count() const noexcept
    {
        return c50_predictions_class_count(handle_);
    }

    const char *class_name(std::size_t class_index) const noexcept
    {
        return c50_predictions_class_name(handle_, class_index);
    }

    std::size_t class_index(std::size_t row) const noexcept
    {
        return c50_predictions_class_index(handle_, row);
    }

    double confidence(std::size_t row) const noexcept
    {
        return c50_predictions_confidence(handle_, row);
    }

    double score(std::size_t row, std::size_t class_index) const noexcept
    {
        return c50_predictions_score(handle_, row, class_index);
    }

    c50_predictions *native_handle() noexcept { return handle_; }
    const c50_predictions *native_handle() const noexcept { return handle_; }

private:
    friend class model;

    explicit predictions(c50_predictions *handle) noexcept : handle_(handle) {}

    c50_predictions *handle_;
};

class model
{
public:
    ~model()
    {
        c50_model_destroy(handle_);
    }

    model(const model &) = delete;
    model &operator=(const model &) = delete;

    model(model &&other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

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

    model_kind kind() const noexcept
    {
        return static_cast<model_kind>(c50_model_get_kind(handle_));
    }

    std::string names_data() const
    {
        std::size_t size = 0;
        const char *data = c50_model_names_data(handle_, &size);
        return data ? std::string(data, size) : std::string();
    }

    std::string serialized_data() const
    {
        std::size_t size = 0;
        const char *data = c50_model_serialized_data(handle_, &size);
        return data ? std::string(data, size) : std::string();
    }

    std::string costs_data() const
    {
        std::size_t size = 0;
        const char *data = c50_model_costs_data(handle_, &size);
        return data ? std::string(data, size) : std::string();
    }

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

    predictions predict(context &ctx, const std::string &cases_data) const
    {
        return predict(ctx, cases_data.data(), cases_data.size());
    }

    c50_model *native_handle() noexcept { return handle_; }
    const c50_model *native_handle() const noexcept { return handle_; }

private:
    explicit model(c50_model *handle) noexcept : handle_(handle) {}

    c50_model *handle_;
};

} // namespace c50

#endif
