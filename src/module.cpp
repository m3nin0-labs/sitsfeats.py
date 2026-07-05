//
// Copyright (C) 2024-2026 sitsfeats.py.
//
// sitsfeats.py is free software; you can redistribute it and/or modify it
// under the terms of the MIT License; see LICENSE file for more details.
//

#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "metrics.h"

namespace {

// Single source of truth for all available metrics, built once from every
// family's registration hook.
const std::vector<MetricDescriptor>& registry() {
  // initialize the registry
  static const std::vector<MetricDescriptor> reg = [] {
    // create the registry
    std::vector<MetricDescriptor> r;

    // register the basic metrics
    register_basic_metrics(r);

    // register the polar metrics
    register_polar_metrics(r);

    // return
    return r;
  }();

  // return the registry
  return reg;
}

// Lookup the metric descriptor by name
const MetricDescriptor& lookup(const std::string& name) {
  static const std::unordered_map<std::string, const MetricDescriptor*> index =
      [] {
        // create the index
        std::unordered_map<std::string, const MetricDescriptor*> m;

        // iterate over the registry and emplace communities
        for (const auto& d : registry()) {
          m.emplace(d.name, &d);
        }

        // return the index
        return m;
      }();

  // find the metric descriptor by name
  auto it = index.find(name);

  // if the metric descriptor is not found, throw an error
  if (it == index.end()) {
    throw std::invalid_argument("Unknown metric: " + name);
  }

  // return!
  return *it->second;
}

// Compute the requested metrics in a single pass and pack
// them column-wise into one `(n_series, n_metrics)` matrix. Column `j` holds
// the metric named `names[j]`.
RowMatrix compute(const std::vector<std::string>& names, const Mat& data) {
  // create the output matrix
  RowMatrix out(data.rows(), static_cast<Eigen::Index>(names.size()));

  // iterate over the names and compute the metrics
  for (std::size_t j = 0; j < names.size(); ++j) {
    out.col(static_cast<Eigen::Index>(j)) = lookup(names[j]).fn(data);
  }

  // return!
  return out;
}

// `(name, group, description)` triples for every available metric, so Python can
// reflect the registry instead of mirroring a list.
std::vector<std::tuple<std::string, std::string, std::string>> list_metrics() {
  // vector
  std::vector<std::tuple<std::string, std::string, std::string>> out;

  // reserve the size
  out.reserve(registry().size());

  // iterate over the registry and emplace the triples
  for (const auto& d : registry()) {
    out.emplace_back(d.name, d.group, d.description);
  }

  // return!
  return out;
}

}  // namespace

NB_MODULE(_sitsfeats, m) {
  m.doc() = "(Internal) Time-series features";

  // Discovery + batch computation.
  m.def("list_metrics", &list_metrics,
        "List available metrics as (name, group, description) triples.");
  
  m.def("compute", &compute, nb::arg("names"), nb::arg("data"),
        "Compute the named metrics, stacked column-wise into one matrix.");

  // Per-metric kernels, generated from the registry
  for (const auto& d : registry()) {
    m.def(("C_ts_" + d.name).c_str(), d.fn, "Time-series metric");
  }
}
