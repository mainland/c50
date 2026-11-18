/* Copyright 2026 Geoffrey Mainland. */
/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <exception>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <c50/c50.hpp>

namespace py = pybind11;

namespace
{

// Text that the native library reads after the binding releases the GIL. A
// str or bytes argument is immutable, and the call keeps it alive, so the text
// borrows its buffer. A data file can be larger than a gigabyte, and a copy
// would double that memory. A bytearray's contents are copied when the
// argument is converted, because keeping a bytearray alive does not keep its
// storage valid. Python code that runs while pybind11 converts a later
// argument, such as an __index__ method, can resize it, and so can another
// thread once the GIL is released. pybind11 documents this limitation of its
// std::string_view caster, which borrows a bytearray's storage.
class text_argument
{
public:
    text_argument() = default;
    explicit text_argument(std::string_view borrowed) noexcept
        : borrowed_(borrowed)
    {
    }
    explicit text_argument(std::string copy) : copy_(std::move(copy)) {}

    std::string_view view() const noexcept
    {
        return copy_ ? std::string_view(*copy_) : borrowed_;
    }

private:
    std::string_view borrowed_;
    std::optional<std::string> copy_;
};

// The feature matrix of a training call. Its conversion to a C-contiguous
// float64 array is the one that every dense function performs, but an object
// that NumPy cannot convert, such as an array of words or a ragged list, raises
// ValueError that names the argument. pybind11 would raise an unnamed
// TypeError for it.
class training_matrix
{
public:
    using array_type =
        py::array_t<double, py::array::c_style | py::array::forcecast>;

    training_matrix() = default;
    explicit training_matrix(array_type array) : array_(std::move(array)) {}

    const array_type &array() const noexcept { return array_; }

private:
    array_type array_;
};

// The tree depth limit of a load call, where std::nullopt is no limit. The
// converter accepts None and a count, as the c50 package documents. It rejects
// anything else with pybind11's TypeError for an unsuitable argument. Zero, a
// negative number, and a number beyond the native size type raise ValueError.
class tree_depth_argument
{
public:
    tree_depth_argument() = default;
    explicit tree_depth_argument(std::optional<std::size_t> limit) noexcept
        : limit_(limit)
    {
    }

    std::optional<std::size_t> limit() const noexcept { return limit_; }

private:
    std::optional<std::size_t> limit_;
};

// The worker count of a training call. The converter accepts a count, as the
// c50 package documents, and rejects anything else with TypeError. A number
// that does not fit the native type raises a ValueError that names the
// argument. The native context checks the supported range.
class worker_count_argument
{
public:
    worker_count_argument() = default;
    explicit worker_count_argument(unsigned count) noexcept : count_(count) {}

    unsigned count() const noexcept { return count_; }

private:
    unsigned count_ = 1;
};

// Whether the object is a bool or a NumPy boolean, which no count argument
// accepts. NumPy 1.26 to 2.2 define np.bool_.__index__ with a deprecation, so
// the type decides, not the missing method. An int that is not a bool cannot
// be a numpy.bool_, so it returns before the lookup, which would run the
// deferred import of a lazily loaded NumPy. For any other object, the lookup
// does not import NumPy by itself: an object of numpy.bool_ exists only when
// the module is already loaded. A missing module or attribute means that no
// such object exists. Any other error, such as one that the object's __class__
// raises, propagates.
bool is_bool_like(const py::handle &source)
{
    if ( PyBool_Check(source.ptr()) )
    {
        return true;
    }
    if ( PyLong_Check(source.ptr()) )
    {
        return false;
    }
    const py::object numpy = py::reinterpret_steal<py::object>(
        PyImport_GetModule(py::str("numpy").ptr()));
    if ( !numpy )
    {
        if ( PyErr_Occurred() )
        {
            throw py::error_already_set();
        }
        return false;
    }
    const py::object bool_type =
        py::reinterpret_steal<py::object>(
            PyObject_GetAttrString(numpy.ptr(), "bool_"));
    if ( !bool_type )
    {
        if ( PyErr_ExceptionMatches(PyExc_AttributeError) )
        {
            PyErr_Clear();
            return false;
        }
        throw py::error_already_set();
    }
    const int found = PyObject_IsInstance(source.ptr(), bool_type.ptr());
    if ( found < 0 )
    {
        throw py::error_already_set();
    }
    return found == 1;
}

// Why a count argument is not an integer.
enum class count_problem
{
    none,
    boolean,
    not_an_index,
};

// The one place that decides which objects every count argument accepts: the
// objects with __index__, except a bool or a NumPy boolean. The rule is in the
// documentation of the c50 package. Return the object as a Python int, or a
// null object and the reason when the argument is unsuitable. An exception
// that the checks or __index__ raise propagates.
py::object count_index(const py::handle &source, count_problem &problem)
{
    problem = count_problem::none;
    if ( is_bool_like(source) )
    {
        problem = count_problem::boolean;
        return py::object();
    }
    if ( !PyIndex_Check(source.ptr()) )
    {
        problem = count_problem::not_an_index;
        return py::object();
    }
    py::object number =
        py::reinterpret_steal<py::object>(PyNumber_Index(source.ptr()));
    if ( !number )
    {
        throw py::error_already_set();
    }
    return number;
}

// The TypeError for an unsuitable count argument.
[[noreturn]] void reject_count(
    const py::handle &source, const char *field, count_problem problem)
{
    if ( problem == count_problem::boolean )
    {
        throw py::type_error(std::string(field) + " must not be a bool");
    }
    throw py::type_error(
        std::string(field) + " must be an integer, not " +
        std::string(py::str(py::type::of(source).attr("__name__"))));
}

// Convert an int that count_index returned to unsigned, or raise ValueError
// that names the field when the value does not fit. Python raises ValueError
// or OverflowError for such a value, never pybind11's TypeError for an
// unsuitable argument.
unsigned fit_unsigned(const py::object &number, const char *field)
{
    const unsigned long long value = PyLong_AsUnsignedLongLong(number.ptr());
    if ( value == static_cast<unsigned long long>(-1) && PyErr_Occurred() )
    {
        PyErr_Clear();
        throw py::value_error(
            std::string(field) + " does not fit an unsigned integer");
    }
    if ( value > std::numeric_limits<unsigned>::max() )
    {
        throw py::value_error(
            std::string(field) + " does not fit an unsigned integer");
    }
    return static_cast<unsigned>(value);
}

// Convert a count to unsigned, or raise TypeError for an unsuitable argument
// and ValueError when the value does not fit.
unsigned checked_unsigned(const py::handle &source, const char *field)
{
    count_problem problem;
    const py::object number = count_index(source, problem);
    if ( !number )
    {
        reject_count(source, field, problem);
    }
    return fit_unsigned(number, field);
}

// Convert an int or float to double, or raise ValueError that names the field
// when an int is too large for a double.
double checked_double(const py::handle &source, const char *field)
{
    const double value = PyFloat_AsDouble(source.ptr());
    if ( value == -1.0 && PyErr_Occurred() )
    {
        if ( PyErr_ExceptionMatches(PyExc_OverflowError) )
        {
            PyErr_Clear();
            throw py::value_error(
                std::string(field) + " is too large for a float");
        }
        throw py::error_already_set();
    }
    return value;
}

// Convert with NumPy, as the converting pass of an array_t parameter does, but
// keep the Python error. pybind11 clears the error of a failed conversion, so
// the caller could neither say why it failed nor tell a MemoryError or a
// KeyboardInterrupt from an object that NumPy cannot read. An Exception other
// than MemoryError becomes a ValueError chained to it. MemoryError and any
// exception that is not an Exception propagate as they are.
py::object convert_to_float64(py::handle source)
{
    try
    {
        return py::module_::import("numpy").attr("asarray")(
            source, py::arg("dtype") = "float64", py::arg("order") = "C");
    }
    catch ( py::error_already_set &error )
    {
        if ( !error.matches(PyExc_Exception) ||
             error.matches(PyExc_MemoryError) )
        {
            throw;
        }
        py::raise_from(
            error, PyExc_ValueError,
            "values cannot be converted to a float64 array");
        throw py::error_already_set();
    }
}

} // namespace

namespace pybind11::detail
{

// Accept the same objects as a std::string parameter: str, encoded as UTF-8,
// bytes, and bytearray.
template <>
struct type_caster<text_argument>
{
    PYBIND11_TYPE_CASTER(text_argument, const_name("str"));

    bool load(handle source, bool convert)
    {
        if ( PyByteArray_Check(source.ptr()) )
        {
            value = text_argument(std::string(
                PyByteArray_AS_STRING(source.ptr()),
                static_cast<std::size_t>(
                    PyByteArray_GET_SIZE(source.ptr()))));
            return true;
        }

        // pybind11 reports a str that UTF-8 cannot encode, such as a lone
        // surrogate, as an unsuitable argument. Python raises
        // UnicodeEncodeError, a ValueError, so let that error through.
        if ( PyUnicode_Check(source.ptr()) &&
             PyUnicode_AsUTF8AndSize(source.ptr(), nullptr) == nullptr )
        {
            throw error_already_set();
        }
        make_caster<std::string_view> text;
        if ( !text.load(source, convert) )
        {
            return false;
        }
        value = text_argument(cast_op<std::string_view>(text));
        return true;
    }
};

// Accept what a float64 array parameter accepts. A float64 C-contiguous array
// is read directly. In the converting pass, NumPy converts anything else, and
// a failure raises ValueError that chains its cause instead of rejecting the
// argument.
template <>
struct type_caster<training_matrix>
{
    PYBIND11_TYPE_CASTER(
        training_matrix, make_caster<training_matrix::array_type>::name);

    bool load(handle source, bool convert)
    {
        make_caster<training_matrix::array_type> array;
        if ( !array.load(source, false) )
        {
            if ( !convert )
            {
                return false;
            }
            const object converted = convert_to_float64(source);
            if ( !array.load(converted, false) )
            {
                throw py::value_error(
                    "values cannot be converted to a float64 array");
            }
        }
        value = training_matrix(
            cast_op<training_matrix::array_type &&>(std::move(array)));
        return true;
    }
};

// Accept None, or a count (see count_index) between 1 and the native size
// type's maximum.
template <>
struct type_caster<tree_depth_argument>
{
    PYBIND11_TYPE_CASTER(
        tree_depth_argument, const_name("typing.SupportsIndex | None"));

    bool load(handle source, bool)
    {
        if ( source.is_none() )
        {
            value = tree_depth_argument();
            return true;
        }
        count_problem problem;
        const object number = count_index(source, problem);
        if ( !number )
        {
            return false;
        }
        if ( PyObject_RichCompareBool(number.ptr(), int_(1).ptr(), Py_LT) )
        {
            throw value_error("max_tree_depth must be positive or None");
        }
        const unsigned long long levels =
            PyLong_AsUnsignedLongLong(number.ptr());
        if ( levels == static_cast<unsigned long long>(-1) &&
             PyErr_Occurred() )
        {
            PyErr_Clear();
            throw value_error("max_tree_depth is too large");
        }
        if ( levels > std::numeric_limits<std::size_t>::max() )
        {
            throw value_error("max_tree_depth is too large");
        }
        value = tree_depth_argument(static_cast<std::size_t>(levels));
        return true;
    }
};

// Accept a count (see count_index), and raise ValueError when the value does
// not fit unsigned. Reject any other object with pybind11's TypeError for an
// unsuitable argument, as the max_tree_depth caster does.
template <>
struct type_caster<worker_count_argument>
{
    PYBIND11_TYPE_CASTER(
        worker_count_argument, const_name("typing.SupportsIndex"));

    bool load(handle source, bool)
    {
        count_problem problem;
        const object number = count_index(source, problem);
        if ( !number )
        {
            return false;
        }
        value = worker_count_argument(fit_unsigned(number, "split_workers"));
        return true;
    }
};

} // namespace pybind11::detail

namespace
{

const char train_doc[] = R"doc(
Train a C5.0 classifier from text held in memory.

The names, training, and optional costs strings use the corresponding C5.0
file formats. The returned model owns its serialized representation and is
independent of the native context used during training. The model parses that
text without a tree depth limit, because the library wrote it.

The call stops at the first bad value, such as a discrete value that the
names file does not declare or a nonfinite number, and raises ``ValueError``
with the line number. ``c5.0`` instead reports each bad value, replaces it
with ``?``, and continues until ten errors.

Args:
    names: Contents of a C5.0 names file.
    training_data: Contents of a C5.0 data file.
    kind: Whether to build a tree or a ruleset.
    options: Training options. The call reads a copy, so later changes to
        the object do not affect it.
    costs: Contents of a C5.0 costs file, or an empty string.
    split_workers: Number of native split-evaluation workers, from 1
        through 8.

Returns:
    The trained model.

Raises:
    ValueError: If an option, the schema, the data, or the costs are
        invalid.
    C50Error: If the schema uses a feature that the library does not
        support.
    MemoryError: If memory is exhausted.
)doc";

const char load_doc[] = R"doc(
Load and validate a serialized C5.0 classifier.

The returned model owns copies of the names, serialized classifier, and
optional costs strings.

Args:
    names: Contents of the C5.0 names file that the classifier was trained
        with.
    serialized_data: Contents of a ``.tree`` or ``.rules`` file.
    kind: Whether ``serialized_data`` holds a tree or a ruleset.
    costs: Contents of the C5.0 costs file, which a classifier trained with
        costs requires. Otherwise it is ignored and the model retains none.
        Text that sets no cost other than the default of 1, such as empty
        text, only comments, only entries of cost 1, or only entries that name
        one class twice, counts as missing and raises ``ValueError`` for a
        cost model.
    max_tree_depth: The most levels that a serialized tree may have, where the
        root is level 1, or ``None`` for no limit. The default is 1024. A later
        1.x release may raise it but never lowers it. The limit bounds the
        stack that parsing needs for untrusted text. A tree needs stack in
        proportion to its depth to parse and to predict, so remove the limit
        only for text that you trust. The model applies the same limit whenever
        it parses its text again. A rules model ignores it.

Returns:
    The loaded model.

Raises:
    TypeError: If ``max_tree_depth`` is neither ``None`` nor a count, as the
        ``c50`` package documentation defines it.
    ValueError: If the names, the classifier, or the costs are invalid, a
        required costs file is missing, the tree is deeper than
        ``max_tree_depth``, or ``max_tree_depth`` is not positive.
    C50Error: If the schema uses a feature that the library does not
        support.
    MemoryError: If memory is exhausted.
)doc";

const char train_dense_doc[] = R"doc(
Train a C5.0 classifier from a dense NumPy feature matrix.

Continuous columns contain their numeric values. Categorical columns contain
zero-based indices into the explicit values declared by ``names``. NaN denotes
a missing feature. A true element of ``not_applicable`` marks the value of that
case and feature as not applicable, which C5.0 treats differently from missing
(the data-file ``N/A`` as opposed to ``?``). The matching element of ``values``
is then ignored, whatever it holds. The returned model parses its serialized
text without a tree depth limit, because the library wrote it.

Training runs without the global interpreter lock, so by default it copies
``values`` and ``not_applicable`` first. With ``copy=False``, training reads a
float64, C-contiguous ``values`` array and a C-contiguous bool
``not_applicable`` array directly, which avoids holding a second copy of the
matrices. The caller must then ensure that no thread modifies either array
until the call returns. Arrays with another dtype or layout are always
converted into a temporary array that no other code can modify.

Args:
    names: Contents of a C5.0 names file.
    values: A two-dimensional matrix with one column per feature and one row
        per case.
    class_indices: A one-dimensional integer array with one nonnegative
        zero-based class index per row.
    kind: Whether to build a tree or a ruleset.
    options: Training options. The call reads a copy, so later changes to
        the object do not affect it.
    costs: Contents of a C5.0 costs file, or an empty string.
    split_workers: Number of native split-evaluation workers, from 1
        through 8.
    copy: Whether to copy ``values`` and ``not_applicable`` before training.
    not_applicable: Keyword-only. None, or a bool array with the same shape
        as ``values`` whose true elements mark not-applicable values. A
        value under a true element is never checked, so it may be NaN or out
        of range. The class has no not-applicable value.

Returns:
    The trained model.

Raises:
    ValueError: If an array has the wrong shape or dtype, ``values`` cannot
        be converted to a float64 array (the error chains the cause), the
        row counts differ, a class
        index or a value is out of range, ``not_applicable``
        does not have a bool dtype or the shape of ``values``, or an option,
        the schema, or the costs are invalid.
    C50Error: If the schema has an implicit, ignored, label, date, time, or
        timestamp attribute, a class attribute, or a discrete attribute
        without explicit values, which dense data does not support. The
        number of columns does not matter.
    MemoryError: If memory is exhausted, including while converting
        ``values``. A ``KeyboardInterrupt`` or any other exception that is not
        an ``Exception`` raised by that conversion also propagates.
)doc";

const char predict_details_dense_doc[] = R"doc(
Predict a dense NumPy feature matrix.

The matrix uses the same continuous, categorical, missing-value, and
not-applicable encoding as ``train_dense``. The call always copies its
arguments before it releases the global interpreter lock.

Args:
    values: A two-dimensional matrix with one column per feature and one row
        per case.
    not_applicable: Keyword-only. None, or a bool array with the same shape
        as ``values`` whose true elements mark not-applicable values.

Returns:
    The labels, confidences, and per-class scores of the rows.

Raises:
    ValueError: If the array is not two-dimensional, has the wrong number of
        columns, or holds a value that is out of range for its feature, or
        if ``not_applicable`` does not have a bool dtype or the shape of
        ``values``.
    C50Error: If the model's schema has an implicit, ignored, label, date,
        time, or timestamp attribute, a class attribute, or a discrete
        attribute without explicit values, which the dense interface does not
        support. The number of columns does not matter. Use the text methods
        for such a model.
    MemoryError: If memory is exhausted.
)doc";

const char attribute_usage_doc[] = R"doc(
Count the cases whose classification tests each attribute.

A case counts for an attribute when classifying it tests that attribute and
the case's value is known. The result has one count per attribute of
``names_data``, in declaration order. These are the counts behind C5.0's
"Attribute usage" report, which divides them by the number of cases.

The call stops at the first bad value, such as a discrete value that the
names file does not declare or a nonfinite number, and raises ``ValueError``
with the line number. ``c5.0`` instead reports each bad value, replaces it
with ``?``, and continues until ten errors.

Args:
    cases: Case rows in C5.0 data-file syntax. Each row includes a final
        class field, which may be ``?``.

Returns:
    One count per attribute.

Raises:
    ValueError: If a case row is malformed.
    MemoryError: If memory is exhausted.
)doc";

const char attribute_usage_dense_doc[] = R"doc(
Count attribute usage for a dense NumPy feature matrix.

The matrix uses the same encoding as ``predict_details_dense``. The result
matches ``attribute_usage`` for the equivalent data-file contents.

Args:
    values: A two-dimensional matrix with one column per feature and one row
        per case.
    not_applicable: Keyword-only. None, or a bool array with the same shape
        as ``values`` whose true elements mark not-applicable values.

Returns:
    One count per attribute.

Raises:
    ValueError: If the array is not two-dimensional, has the wrong number of
        columns, or holds a value that is out of range for its feature, or
        if ``not_applicable`` does not have a bool dtype or the shape of
        ``values``.
    C50Error: If the model's schema has an implicit, ignored, label, date,
        time, or timestamp attribute, a class attribute, or a discrete
        attribute without explicit values, which the dense interface does not
        support. The number of columns does not matter. Use the text methods
        for such a model.
    MemoryError: If memory is exhausted.
)doc";

template<class T>
using dense_values = py::array_t<
    T, py::array::c_style | py::array::forcecast>;

// A validated not-applicable mask for one dense call. It owns whatever keeps
// the bytes alive while the GIL is released: a private copy, or a reference to
// the caller's C-contiguous bool array when the caller promised not to modify
// it. Without a mask, data() is null.
class mask_argument
{
public:
    mask_argument() = default;

    mask_argument(const std::optional<py::array> &mask,
                  const py::array &values, bool copy)
    {
        if ( ! mask )
        {
            return;
        }
        if ( mask->dtype().kind() != 'b' )
        {
            throw py::value_error("not_applicable must have a bool dtype");
        }
        if ( mask->ndim() != values.ndim() )
        {
            throw py::value_error(
                "not_applicable must have the same shape as values");
        }
        for ( py::ssize_t axis = 0; axis < values.ndim(); ++axis )
        {
            if ( mask->shape(axis) != values.shape(axis) )
            {
                throw py::value_error(
                    "not_applicable must have the same shape as values");
            }
        }
        // Another layout is converted into a temporary that no other code
        // can modify.
        array_ = py::array_t<bool, py::array::c_style | py::array::forcecast>(
            *mask);
        data_ = reinterpret_cast<const unsigned char *>(array_.data());
        if ( copy )
        {
            bytes_.assign(data_, data_ + array_.size());
            data_ = bytes_.data();
            array_ = py::array_t<bool, py::array::c_style>();
        }
    }

    const unsigned char *data() const noexcept { return data_; }

private:
    py::array_t<bool, py::array::c_style> array_;
    std::vector<unsigned char> bytes_;
    const unsigned char *data_ = nullptr;
};

template<class T>
std::vector<std::size_t> copy_checked_classes(const dense_values<T> &classes)
{
    std::vector<std::size_t> result(static_cast<std::size_t>(classes.shape(0)));
    const auto values = classes.template unchecked<1>();
    for ( py::ssize_t row = 0; row < values.shape(0); ++row )
    {
        const auto value = values(row);
        if constexpr ( std::is_signed_v<T> )
            if ( value < 0 )
                throw py::value_error("class_indices must be nonnegative");
        if ( value >= std::numeric_limits<int>::max() )
            throw py::value_error("class_indices exceed the native class range");
        result[static_cast<std::size_t>(row)] = static_cast<std::size_t>(value);
    }
    return result;
}

std::vector<std::size_t> checked_classes(const py::array &classes)
{
    const char kind = classes.dtype().kind();
    if ( kind != 'i' && kind != 'u' )
        throw py::value_error("class_indices must have an integer dtype");
    if ( kind == 'i' )
        return copy_checked_classes(dense_values<std::int64_t>(classes));
    return copy_checked_classes(dense_values<std::uint64_t>(classes));
}

const char predict_details_doc[] = R"doc(
Predict cases and return labels, confidences, and per-class scores.

Each case row uses C5.0 data-file syntax and includes a final class field,
which may be ``?`` when unknown.

The call stops at the first bad value, such as a discrete value that the
names file does not declare or a nonfinite number, and raises ``ValueError``
with the line number. ``c5.0`` instead reports each bad value, replaces it
with ``?``, and continues until ten errors.

Args:
    cases: Case rows in C5.0 data-file syntax.

Returns:
    The labels, confidences, and per-class scores of the rows.

Raises:
    ValueError: If a case row is malformed.
    MemoryError: If memory is exhausted.
)doc";

const char predict_doc[] = R"doc(
Predict the class label for each case row.

Each case row uses C5.0 data-file syntax and includes a final class field,
which may be ``?`` when unknown.

The call stops at the first bad value, such as a discrete value that the
names file does not declare or a nonfinite number, and raises ``ValueError``
with the line number. ``c5.0`` instead reports each bad value, replaces it
with ``?``, and continues until ten errors.

Args:
    cases: Case rows in C5.0 data-file syntax.

Returns:
    One class label per row.

Raises:
    ValueError: If a case row is malformed.
    MemoryError: If memory is exhausted.
)doc";

const char predict_scores_doc[] = R"doc(
Return C5.0's per-class scores for each case row.

The scores are C5.0's native class scores, not normalized probabilities: a
ruleset's scores, for example, sum to less than one. Columns follow
``class_names`` order. Each case row uses C5.0 data-file syntax and includes
a final class field, which may be ``?`` when unknown.

The call stops at the first bad value, such as a discrete value that the
names file does not declare or a nonfinite number, and raises ``ValueError``
with the line number. ``c5.0`` instead reports each bad value, replaces it
with ``?``, and continues until ten errors.

Args:
    cases: Case rows in C5.0 data-file syntax.

Returns:
    One list of class scores per row.

Raises:
    ValueError: If a case row is malformed.
    MemoryError: If memory is exhausted.
)doc";

class python_predictor
{
public:
    explicit python_predictor(c50::predictor predictor) : predictor_(std::move(predictor)) {}

    c50::predictions predict_details(const text_argument &cases)
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return predictor_.predict(cases.view());
    }

    c50::predictions predict_details_dense(
        const dense_values<double> &values,
        const std::optional<py::array> &not_applicable)
    {
        if ( values.ndim() != 2 )
            throw py::value_error("values must be a two-dimensional array");
        const mask_argument mask(not_applicable, values, true);
        std::vector<double> copy(static_cast<std::size_t>(values.size()));
        std::copy_n(values.data(), copy.size(), copy.data());
        const c50::dense_dataset dataset(copy.data(), static_cast<std::size_t>(values.shape(0)),
                                        static_cast<std::size_t>(values.shape(1)),
                                        nullptr, 0, mask.data());
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        return predictor_.predict(dataset);
    }

private:
    c50::predictor predictor_;
    std::mutex mutex_;
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

class python_model
{
public:
    explicit python_model(c50::model model) : model_(std::move(model)) {}

    python_model(const python_model &) = delete;
    python_model &operator=(const python_model &) = delete;
    python_model(python_model &&) noexcept = default;
    python_model &operator=(python_model &&) noexcept = default;

    static python_model train(const text_argument &names,
                              const text_argument &training_data,
                              c50::model_kind kind,
                              c50::options options,
                              const text_argument &costs,
                              const worker_count_argument &split_workers)
    {
        // The options are a copy. A reference would point into the Python
        // Options object, which another thread can change while the GIL is
        // released.
        c50::context context;
        context.split_workers(split_workers.count());
        py::gil_scoped_release release;
        return python_model(c50::model::train(
            context, kind, names.view(), training_data.view(), options,
            costs.view()));
    }

    static python_model train_dense(const text_argument &names,
                                    const training_matrix &matrix,
                                    const py::array &class_indices,
                                    c50::model_kind kind,
                                    c50::options options,
                                    const text_argument &costs,
                                    const worker_count_argument &split_workers,
                                    bool copy,
                                    const std::optional<py::array> &not_applicable)
    {
        const dense_values<double> &values = matrix.array();
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
        // The mask follows values: a private copy, or the caller's array
        // when copy is false.
        const mask_argument mask(not_applicable, values, copy);
        // The argument keeps values alive while the GIL is released. Without
        // a copy, the caller guarantees that no thread modifies it.
        std::vector<double> value_copy;
        const double *data = values.data();
        if ( copy )
        {
            value_copy.assign(data, data + values.size());
            data = value_copy.data();
        }
        const c50::dense_dataset dataset(
            data, static_cast<std::size_t>(values.shape(0)),
            static_cast<std::size_t>(values.shape(1)), indices.data(), 0,
            mask.data());
        c50::context context;
        context.split_workers(split_workers.count());
        py::gil_scoped_release release;
        return python_model(c50::model::train(
            context, kind, names.view(), dataset, options, costs.view()));
    }

    static python_model load(const text_argument &names,
                             const text_argument &serialized_data,
                             c50::model_kind kind,
                             const text_argument &costs,
                             const tree_depth_argument &max_tree_depth)
    {
        c50::load_options settings;
        settings.max_tree_depth = max_tree_depth.limit();
        c50::context context;
        py::gil_scoped_release release;
        return python_model(c50::model::load(
            context, kind, names.view(), serialized_data.view(),
            costs.view(), settings));
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

    c50::predictions predict_details(const text_argument &cases) const
    {
        c50::context context;
        py::gil_scoped_release release;
        return model_.predict(context, cases.view());
    }

    c50::predictions predict_details_dense(
        const dense_values<double> &values,
        const std::optional<py::array> &not_applicable) const
    {
        if ( values.ndim() != 2 )
        {
            throw py::value_error("values must be a two-dimensional array");
        }
        const mask_argument mask(not_applicable, values, true);

        std::vector<double> value_copy(static_cast<std::size_t>(values.size()));
        std::copy_n(values.data(), value_copy.size(), value_copy.data());
        const c50::dense_dataset dataset(
            value_copy.data(), static_cast<std::size_t>(values.shape(0)),
            static_cast<std::size_t>(values.shape(1)), nullptr, 0,
            mask.data());
        c50::context context;
        py::gil_scoped_release release;
        return model_.predict(context, dataset);
    }

    std::vector<std::size_t> attribute_usage(const text_argument &cases) const
    {
        c50::context context;
        py::gil_scoped_release release;
        return model_.attribute_usage(context, cases.view());
    }

    std::vector<std::size_t> attribute_usage_dense(
        const dense_values<double> &values,
        const std::optional<py::array> &not_applicable) const
    {
        if ( values.ndim() != 2 )
        {
            throw py::value_error("values must be a two-dimensional array");
        }
        const mask_argument mask(not_applicable, values, true);

        std::vector<double> value_copy(static_cast<std::size_t>(values.size()));
        std::copy_n(values.data(), value_copy.size(), value_copy.data());
        const c50::dense_dataset dataset(
            value_copy.data(), static_cast<std::size_t>(values.shape(0)),
            static_cast<std::size_t>(values.shape(1)), nullptr, 0,
            mask.data());
        c50::context context;
        py::gil_scoped_release release;
        return model_.attribute_usage(context, dataset);
    }

    std::vector<std::string> predict(const text_argument &cases) const
    {
        return prediction_labels(predict_details(cases));
    }

    std::vector<std::vector<double>> predict_scores(
        const text_argument &cases) const
    {
        return prediction_scores(predict_details(cases));
    }

    std::vector<std::string> classes() const
    {
        return prediction_class_names(predict_details(text_argument()));
    }

private:
    c50::model model_;
};

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
    py::native_enum<c50::node_kind>(
        module, "NodeKind", "enum.IntEnum",
        "Tree node representation in an inspection snapshot.")
        .value("LEAF", c50::node_kind::leaf,
               "Terminal class distribution.")
        .value("DISCRETE", c50::node_kind::discrete,
               "One branch per categorical value, including N/A.")
        .value("THRESHOLD", c50::node_kind::threshold,
               "Continuous cutoff, possibly with soft thresholds.")
        .value("SUBSET", c50::node_kind::subset,
               "Branches containing subsets of categorical values.")
        .finalize();
    py::native_enum<c50::condition_kind>(
        module, "ConditionKind", "enum.IntEnum",
        "A tree branch or rule condition, independent of node encodings.")
        .value("NOT_APPLICABLE", c50::condition_kind::not_applicable,
               "The feature has the distinct N/A value.")
        .value("EQUALS", c50::condition_kind::equals,
               "The feature equals the one named category.")
        .value("LESS_EQUAL", c50::condition_kind::less_equal,
               "The feature is at or below the cutoff.")
        .value("GREATER", c50::condition_kind::greater,
               "The feature is above the cutoff.")
        .value("IN_SUBSET", c50::condition_kind::in_subset,
               "The feature belongs to the named subset.")
        .finalize();
    py::class_<c50::split_condition>(module, "SplitCondition",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("feature", &c50::split_condition::feature,
                      "Zero-based schema attribute index.")
        .def_readonly("kind", &c50::split_condition::kind,
                      "Comparison kind.")
        .def_readonly("cut", &c50::split_condition::cut,
                      "Cutoff for LESS_EQUAL and GREATER, otherwise zero.")
        .def_readonly("values", &c50::split_condition::values,
                      "Ordinary categories for EQUALS and IN_SUBSET.")
        .def_readonly("includes_not_applicable", &c50::split_condition::includes_not_applicable,
                      "N/A membership, for IN_SUBSET only.");
    py::class_<c50::continuous_threshold>(module, "ContinuousThreshold",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("cut", &c50::continuous_threshold::cut,
                      "Printed decision cutoff.")
        .def_readonly("lower", &c50::continuous_threshold::lower,
                      "Lower soft-threshold point.")
        .def_readonly("midpoint", &c50::continuous_threshold::midpoint,
                      "Middle soft-threshold point.")
        .def_readonly("upper", &c50::continuous_threshold::upper,
                      "Upper soft-threshold point.");
    py::class_<c50::tree_branch>(module, "TreeBranch",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("condition", &c50::tree_branch::condition,
                      "Branch condition in native branch order.")
        .def_readonly("child", &c50::tree_branch::child,
                      "Zero-based child node index.");
    py::class_<c50::tree_node>(module, "TreeNode",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("kind", &c50::tree_node::kind,
                      "Leaf or split representation.")
        .def_readonly("feature", &c50::tree_node::feature,
                      "Tested attribute, None at a leaf.")
        .def_readonly("predicted_class", &c50::tree_node::predicted_class,
                      "Stored class, not a cost-adjusted prediction.")
        .def_readonly("case_weight", &c50::tree_node::case_weight,
                      "Sum of serialized class weights, possibly zero.")
        .def_readonly("class_weights", &c50::tree_node::class_weights,
                      "Weighted support in class-name order.")
        .def_readonly("threshold", &c50::tree_node::threshold,
                      "Present only for thresholds.")
        .def_readonly("branches", &c50::tree_node::branches,
                      "Ordered branches, empty at a leaf.");
    py::class_<c50::tree_inspection>(module, "TreeInspection",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("nodes", &c50::tree_inspection::nodes,
                      "Independently owned node records.")
        .def_readonly("leaf_count", &c50::tree_inspection::leaf_count,
                      "All leaves, including zero-support leaves.")
        .def_readonly("supported_leaf_count", &c50::tree_inspection::supported_leaf_count,
                      "Leaves with positive case weight.")
        .def_readonly("depth", &c50::tree_inspection::depth,
                      "Maximum node depth in edges from the root.")
        .def_readonly("supported_depth", &c50::tree_inspection::supported_depth,
                      "Maximum positive-support leaf depth.")
        .def_readonly("feature_use", &c50::tree_inspection::feature_use,
                      "Split counts in attribute-name order.");
    py::class_<c50::rule_inspection>(module, "RuleInspection",
        "Owned rule metadata in serialized order.")
        .def_readonly("conditions", &c50::rule_inspection::conditions,
                      "Conjunction in serialized order.")
        .def_readonly("predicted_class", &c50::rule_inspection::predicted_class,
                      "Zero-based rule conclusion.")
        .def_readonly("cover", &c50::rule_inspection::cover,
                      "Weighted cases covered by the rule.")
        .def_readonly("correct", &c50::rule_inspection::correct,
                      "Weighted covered cases with the concluded class.")
        .def_readonly("prior", &c50::rule_inspection::prior,
                      "Prior reconstructed from serialized cover, correct, and lift.")
        .def_readonly("vote", &c50::rule_inspection::vote,
                      "Native confidence vote in integer thousandths.");
    py::class_<c50::ruleset_inspection>(module, "RulesetInspection",
        "Owned ruleset with ordered rules and a fallback class.")
        .def_readonly("default_class", &c50::ruleset_inspection::default_class,
                      "Zero-based fallback class.")
        .def_readonly("rules", &c50::ruleset_inspection::rules,
                      "Rules in serialized order.")
        .def_readonly("feature_use", &c50::ruleset_inspection::feature_use,
                      "Condition counts per schema attribute.");
    py::class_<c50::model_inspection>(module, "ModelInspection",
        "Owned inspection metadata. Fields are read-only and lists are copies.")
        .def_readonly("kind", &c50::model_inspection::kind,
                      "Source classifier representation.")
        .def_readonly("class_names", &c50::model_inspection::class_names,
                      "Classes in prediction-score order.")
        .def_readonly("feature_names", &c50::model_inspection::feature_names,
                      "All schema attributes in source order.")
        .def_readonly("trees", &c50::model_inspection::trees,
                      "Tree components, empty for rules models.")
        .def_readonly("rulesets", &c50::model_inspection::rulesets,
                      "Ruleset components in serialized order.");
}

// pybind11 objects have no pickle support unless a class defines it. Without a
// __reduce__, protocols 0 and 1 reach copyreg._reduce_ex, which allocates the
// bare pybind11 base type, and the uncaught C++ exception aborts the process.
// Give every class that the module defines and that does not define its own
// __reduce__ one that fails with TypeError, and give every enumeration one
// that pickles the member by type and member name through the public package
// at every protocol, like a Model, so that the pickle does not name this
// module or depend on the integer values. An enumeration is an enum.IntEnum,
// whose own __reduce_ex__ takes precedence over __reduce__ when pickle and copy
// look for either, so the enumerations get both.
//
// An enum.IntEnum prints as its integer from Python 3.11 on and as its
// qualified name before. Give every enumeration the qualified name at every
// version, which is also what the earlier pybind11 enumerations printed. The
// formatted text must not depend on the version either. The format function of
// enum.IntEnum formats the integer from Python 3.11 on, so give every
// enumeration a format function that formats the string form.
void bind_pickle_support(py::module_ &module)
{
    const py::object reject_pickle = py::cpp_function(
        [](const py::object &self) -> py::tuple {
            const std::string name =
                py::cast<std::string>(py::type::of(self).attr("__name__"));
            throw py::type_error("cannot pickle '" + name + "' objects");
        },
        py::name("__reduce__"), py::is_method(py::none()),
        py::doc("Raise TypeError: this type does not support pickling."));
    const py::object reduce_enum = py::cpp_function(
        [](const py::object &self) {
            const py::module_ package = py::module_::import("c50");
            return py::make_tuple(
                package.attr("_restore_enum"),
                py::make_tuple(py::type::of(self).attr("__name__"),
                               self.attr("name")));
        },
        py::name("__reduce__"), py::is_method(py::none()),
        py::doc("Return the pickle state: the type name and the member "
                "name."));
    const py::object reduce_enum_ex = py::cpp_function(
        [reduce_enum](const py::object &self, const py::object &) {
            return reduce_enum(self);
        },
        py::name("__reduce_ex__"), py::is_method(py::none()),
        py::doc("Return the pickle state at any protocol."));
    const py::object enum_format = py::cpp_function(
        [](const py::object &self, const py::object &specification) {
            return py::str(self).attr("__format__")(specification);
        },
        py::name("__format__"), py::is_method(py::none()),
        py::doc("Format the type name and the member name."));
    const py::object enum_to_string = py::cpp_function(
        [](const py::object &self) {
            return py::str(py::type::of(self).attr("__name__"))
                + py::str(".") + py::str(self.attr("name"));
        },
        py::name("__str__"), py::is_method(py::none()),
        py::doc("Return the type name and the member name."));

    const std::string module_name = module.attr("__name__").cast<std::string>();
    for ( const auto item : py::dict(module.attr("__dict__")) )
    {
        if ( !PyType_Check(item.second.ptr()) )
        {
            continue;
        }
        const py::object type = py::reinterpret_borrow<py::object>(item.second);
        if ( type.attr("__module__").cast<std::string>() != module_name
             || PyType_IsSubtype(reinterpret_cast<PyTypeObject *>(type.ptr()),
                                 reinterpret_cast<PyTypeObject *>(
                                     PyExc_BaseException)) )
        {
            continue;
        }
        if ( py::hasattr(type, "__members__") )
        {
            type.attr("__reduce__") = reduce_enum;
            type.attr("__reduce_ex__") = reduce_enum_ex;
            type.attr("__str__") = enum_to_string;
            type.attr("__format__") = enum_format;
        }
        else if ( !py::dict(type.attr("__dict__")).contains("__reduce__") )
        {
            type.attr("__reduce__") = reject_pickle;
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

    py::native_enum<c50::model_kind>(
        module, "ModelKind", "enum.IntEnum",
        "Serialized classifier representations supported by C5.0.")
        .value("TREE", c50::model_kind::tree,
               "A decision tree or boosted tree ensemble.")
        .value("RULES", c50::model_kind::rules,
               "A ruleset or boosted ruleset ensemble.")
        .finalize();

    py::native_enum<c50::tie_order>(
        module, "TieOrder", "enum.IntEnum",
        "Order of equal continuous values when evaluating splits.")
        .value("REFERENCE", c50::tie_order::reference,
               "Reproduce the tie order of the reference C5.0 learner.")
        .value("STABLE", c50::tie_order::stable,
               "Keep equal values in case order with a faster sort. "
               "Classifiers are deterministic on every platform but may "
               "differ from the reference learner.")
        .finalize();

    module.def(
        "_count_index",
        [](const py::object &value, const char *name) {
            count_problem problem;
            py::object number = count_index(value, problem);
            if ( !number )
            {
                reject_count(value, name, problem);
            }
            return number;
        },
        py::arg("value"), py::arg("name"),
        "Private. Convert a count argument to an int under the rule that "
        "every count argument follows, or raise TypeError.");

    bind_inspection(module);

    py::class_<c50::options>(
        module, "Options",
        "Mutable classifier-construction options. Assigning a number that the "
        "native type cannot hold raises ValueError. Training checks the "
        "documented ranges.")
        .def(py::init<>(), "Construct deterministic single-tree defaults.")
        .def_property(
            "trials",
            [](const c50::options &options) { return options.trials; },
            [](c50::options &options, const py::object &value) {
                options.trials = checked_unsigned(value, "options.trials");
            },
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
        .def_property(
            "minimum_cases",
            [](const c50::options &options) {
                return options.minimum_cases;
            },
            [](c50::options &options, const py::object &value) {
                options.minimum_cases =
                    checked_double(value, "options.minimum_cases");
            },
            "Minimum cases represented by two branches, from 1 through "
            "1000000.")
        .def_property(
            "confidence_factor",
            [](const c50::options &options) {
                return options.confidence_factor;
            },
            [](c50::options &options, const py::object &value) {
                options.confidence_factor =
                    checked_double(value, "options.confidence_factor");
            },
            "Pruning confidence factor expressed as a fraction in [0, 1].")
        .def_property(
            "sample_fraction",
            [](const c50::options &options) {
                return options.sample_fraction;
            },
            [](c50::options &options, const py::object &value) {
                options.sample_fraction =
                    checked_double(value, "options.sample_fraction");
            },
            "Training sample fraction in [0, 0.999], or zero for no "
            "sampling.")
        .def_property(
            "random_seed",
            [](const c50::options &options) { return options.random_seed; },
            [](c50::options &options, const py::object &value) {
                options.random_seed = checked_unsigned(value, "options.random_seed");
            },
            "Sampling seed in the range 0 through 4095.")
        .def_readwrite(
            "ties", &c50::options::ties,
            "Order of equal continuous values when evaluating splits.");

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
             predict_details_dense_doc, py::arg("values"), py::kw_only(),
             py::arg("not_applicable") = py::none());

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
                    py::arg("split_workers") = 1,
                    py::arg("copy") = true, py::kw_only(),
                    py::arg("not_applicable") = py::none())
        .def_static("load", &python_model::load,
                    load_doc,
                    py::arg("names"), py::arg("serialized_data"),
                    py::arg("kind") = c50::model_kind::tree,
                    py::arg("costs") = "", py::kw_only(),
                    py::arg("max_tree_depth") =
                        c50::default_max_tree_depth)
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
            "class_names", &python_model::classes,
            "Class names in score-column order.")
        .def("prepare_predictor", &python_model::prepare_predictor,
             "Create an independent parsed classifier for repeated prediction calls.")
        .def("inspect", &python_model::inspect,
             "Return owned structural metadata parsed from the retained classifier.")
        .def("predict_details", &python_model::predict_details,
             predict_details_doc, py::arg("cases"))
        .def("predict_details_dense", &python_model::predict_details_dense,
             predict_details_dense_doc, py::arg("values"), py::kw_only(),
             py::arg("not_applicable") = py::none())
        .def("attribute_usage", &python_model::attribute_usage,
             attribute_usage_doc, py::arg("cases"))
        .def("attribute_usage_dense", &python_model::attribute_usage_dense,
             attribute_usage_dense_doc, py::arg("values"), py::kw_only(),
             py::arg("not_applicable") = py::none())
        .def("predict", &python_model::predict,
             predict_doc, py::arg("cases"))
        .def("predict_scores", &python_model::predict_scores,
             predict_scores_doc, py::arg("cases"))
        .def("__reduce__",
             [](const python_model &model) {
                 // Rebuild through a public Python function, so that a pickle
                 // names neither this module nor the pybind11 enum protocol.
                 const char *kind =
                     model.kind() == c50::model_kind::rules ? "rules" : "tree";
                 const py::module_ package = py::module_::import("c50");
                 return py::make_tuple(
                     package.attr("_restore_model"),
                     py::make_tuple(package.attr("_MODEL_PICKLE_VERSION"), kind,
                                    model.names_data(),
                                    model.serialized_data(),
                                    model.costs_data()));
             },
             "Return the pickle state: the serialized model text and a format "
             "version.");

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
               py::arg("split_workers") = 1,
               py::arg("copy") = true, py::kw_only(),
               py::arg("not_applicable") = py::none());
    module.def("load", &python_model::load,
               load_doc,
               py::arg("names"), py::arg("serialized_data"),
               py::arg("kind") = c50::model_kind::tree,
               py::arg("costs") = "", py::kw_only(),
               py::arg("max_tree_depth") = c50::default_max_tree_depth);

    bind_pickle_support(module);
}
