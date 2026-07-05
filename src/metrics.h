//
// Copyright (C) 2024-2026 sitsfeats.py.
//
// sitsfeats.py is free software; you can redistribute it and/or modify it
// under the terms of the MIT License; see LICENSE file for more details.
//
// Shared types and the metric-descriptor registry interface. Each metric
// family (basic, polar, ...) contributes descriptors through a single
// `register_*` function, so the Python layer can discover metrics
// automatically instead of mirroring a hand-maintained list.
//

#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

// Row-major matrix matching NumPy's default (C-contiguous) layout. Inputs are
// taken as strided, read-only references so any layout works zero-copy and the
// caller's array is never mutated.
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Mat = nb::DRef<const RowMatrix>;

// A metric reduces each row (time series) to a single value.
using MetricFn = Eigen::VectorXd (*)(const Mat&);

struct MetricDescriptor {
  std::string name;         // e.g., "mean"
  std::string group;        // e.g., "basic"
  std::string description;  // one-line summary, e.g. "Arithmetic mean"
  MetricFn fn;             // function pointer to the metric implementation
};

// Family registration hooks. Add new families here and in module.cpp.
void register_basic_metrics(std::vector<MetricDescriptor>& reg);
void register_polar_metrics(std::vector<MetricDescriptor>& reg);
