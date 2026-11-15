..
   Copyright 2026 Geoffrey Mainland.
   SPDX-License-Identifier: GPL-3.0-or-later

Python API reference
====================

The extension for Python 3.10 or newer binds the compiled C++ API with pybind11. It
provides the compatible C5.0 text formats and a typed dense NumPy boundary.

.. automodule:: c50
   :members:
   :show-inheritance:

.. py:data:: c50.NOT_APPLICABLE

   Marks a value of ``X`` that is not applicable to its case.

   C5.0 distinguishes a missing value, which exists but is unknown, from a
   value that does not apply to the case. The estimator passes a cell that
   holds this object to C5.0 as ``N/A``. The object equals only itself, is not
   missing to ``pandas.isna``, and is not a number. See :doc:`../estimator`.
