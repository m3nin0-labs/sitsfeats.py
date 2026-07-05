//
// Copyright (C) 2024-2026 sitsfeats.py.
//
// sitsfeats.py is free software; you can redistribute it and/or modify it
// under the terms of the MIT License; see LICENSE file for more details.
//
// Basic time-series metrics.
//
// Convention: each row of the input matrix is a time series; reductions run
// along the columns (time), producing one value per row.
//

#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Per-row centered deviations (x - row mean) as an array expression source.
Eigen::ArrayXXd centered(const Mat& m) {
  return m.array().colwise() - m.rowwise().mean().array();
}

// p-quantile of every row using the Hazen plotting position, preserved from
// the original Armadillo implementation so numeric output is unchanged.
// numpy equivalent: ``np.quantile(x, p, axis=1, method="hazen")``.
Eigen::VectorXd row_quantile(const Mat& m, double p) {
  // get the number of rows and columns
  const Eigen::Index n = m.rows();
  const Eigen::Index c = m.cols();

  // initialize the output vector
  Eigen::VectorXd out(n);

  // iterate over the rows
  std::vector<double> row(static_cast<std::size_t>(c));

  for (Eigen::Index i = 0; i < n; ++i) {

    // in this row, we will store the values
    for (Eigen::Index j = 0; j < c; ++j) {
      row[static_cast<std::size_t>(j)] = m(i, j);
    }

    // sort the row
    std::sort(row.begin(), row.end());

    // Hazen: 1-based fractional rank h = p * c + 0.5, clamped to [1, c].
    const double h = std::clamp(
      p * static_cast<double>(c) + 0.5, 1.0, static_cast<double>(c)
    );

    // get the lower index
    const double lo = std::floor(h);

    // get the 0-based lower index
    const auto k = static_cast<std::size_t>(lo) - 1;

    // get the interpolation factor
    const double g = h - lo;

    // compute the quantile
    out(i) = (g == 0.0 || k + 1 >= static_cast<std::size_t>(c))
              ? row[k]
              : row[k] + g * (row[k + 1] - row[k]);
  }

  // return!
  return out;
}

// Max
Eigen::VectorXd ts_max(const Mat& m) { 
  return m.rowwise().maxCoeff();
 }

// Min
Eigen::VectorXd ts_min(const Mat& m) { 
  return m.rowwise().minCoeff();
}

// Mean
Eigen::VectorXd ts_mean(const Mat& m) { 
  return m.rowwise().mean();
}

// Median (Hazen 0.5-quantile equals the standard median)
Eigen::VectorXd ts_median(const Mat& m) { 
  return row_quantile(m, 0.50);
}

// Sum
Eigen::VectorXd ts_sum(const Mat& m) { 
  return m.rowwise().sum();
}

// Standard deviation (sample, ddof = 1)
Eigen::VectorXd ts_std(const Mat& m) {
  // get the number of columns
  const double c = static_cast<double>(m.cols());

  // compute the squared centered deviations
  Eigen::ArrayXd m2 = centered(m).square().rowwise().sum();

  // compute the standard deviation
  return (m2 / (c - 1.0)).sqrt().matrix();
}

// Skewness: adjusted Fisher-Pearson standardized moment coefficient (G1)
Eigen::VectorXd ts_skew(const Mat& m) {
  // get the number of columns
  const double c = static_cast<double>(m.cols());
  
  // compute the centered deviations
  Eigen::ArrayXXd cen = centered(m);

  // compute the squared centered deviations
  Eigen::ArrayXd m2 = cen.square().rowwise().sum();
  Eigen::ArrayXd m3 = cen.cube().rowwise().sum();

  // compute the factor
  const double factor = std::sqrt(c * (c - 1.0)) / (c - 2.0);

  // compute the skewness
  Eigen::ArrayXd g1 = (m3 / c) / (m2 / c).pow(1.5);

  // return!
  return (factor * g1).matrix();
}

// Kurtosis (Pearson's definition, non-excess; normal distribution == 3.0)
Eigen::VectorXd ts_kurt(const Mat& m) {
  // get the number of columns
  const double c = static_cast<double>(m.cols());

  // compute the centered deviations
  Eigen::ArrayXXd cen = centered(m);
  Eigen::ArrayXd m2 = cen.square().rowwise().sum();
  Eigen::ArrayXd m4 = cen.square().square().rowwise().sum();

  // compute the kurtosis
  return (c * m4 / (m2 * m2)).matrix();
}

// Amplitude (max - min)
Eigen::VectorXd ts_amplitude(const Mat& m) {
  return m.rowwise().maxCoeff() - m.rowwise().minCoeff();
}

// F-Slope: maximum absolute first difference.
Eigen::VectorXd ts_fslope(const Mat& m) {
  // if there are less than 2 columns, return zeros
  if (m.cols() < 2) {
    return Eigen::VectorXd::Zero(m.rows());
  }

  // get the number of columns
  Eigen::Index c = m.cols();

  // compute the absolute first differences
  return (m.rightCols(c - 1) - m.leftCols(c - 1))
          .cwiseAbs()
          .rowwise()
          .maxCoeff();
}

// Absolute sum.
Eigen::VectorXd ts_abs_sum(const Mat& m) { 
  return m.cwiseAbs().rowwise().sum();
 }

// AMD: mean absolute first difference.
Eigen::VectorXd ts_amd(const Mat& m) {
  // if there are less than 2 columns, return zeros
  if (m.cols() < 2) {
    return Eigen::VectorXd::Zero(m.rows());
  }

  // get the number of columns
  Eigen::Index c = m.cols();

  // compute the absolute first differences
  return (m.rightCols(c - 1) - m.leftCols(c - 1)).cwiseAbs().rowwise().mean();
}

// MSE: mean power spectrum. By Parseval's theorem the mean of |FFT|^2 over all
// frequency bins equals the sum of squared samples, so no FFT is required.
Eigen::VectorXd ts_mse(const Mat& m) {
  return m.array().square().rowwise().sum().matrix();
}

// First quartile.
Eigen::VectorXd ts_fqr(const Mat& m) { 
  return row_quantile(m, 0.25);
 }

// Third quartile.
Eigen::VectorXd ts_tqr(const Mat& m) { 
  return row_quantile(m, 0.75);
}

// Interquartile range.
Eigen::VectorXd ts_iqr(const Mat& m) {
  return row_quantile(m, 0.75) - row_quantile(m, 0.25);
}
}  // namespace

void register_basic_metrics(std::vector<MetricDescriptor>& reg) {
  reg.push_back({"max", "basic", "Maximum value", &ts_max});
  reg.push_back({"min", "basic", "Minimum value", &ts_min});
  reg.push_back({"mean", "basic", "Arithmetic mean", &ts_mean});
  reg.push_back({"median", "basic", "Median (second quartile)", &ts_median});
  reg.push_back({"sum", "basic", "Sum of values", &ts_sum});
  reg.push_back({"std", "basic", "Sample standard deviation (ddof=1)", &ts_std});
  reg.push_back({"skew", "basic", "Adjusted Fisher-Pearson skewness", &ts_skew});
  reg.push_back({"kurt", "basic", "Kurtosis (Pearson, non-excess)", &ts_kurt});
  reg.push_back({"amplitude", "basic", "Range (max - min)", &ts_amplitude});
  reg.push_back({"fslope", "basic", "Maximum absolute first difference", &ts_fslope});
  reg.push_back({"abs_sum", "basic", "Sum of absolute values", &ts_abs_sum});
  reg.push_back({"amd", "basic", "Mean absolute first difference", &ts_amd});
  reg.push_back({"mse", "basic", "Mean power spectrum (sum of squares)", &ts_mse});
  reg.push_back({"fqr", "basic", "First quartile (Q1)", &ts_fqr});
  reg.push_back({"tqr", "basic", "Third quartile (Q3)", &ts_tqr});
  reg.push_back({"iqr", "basic", "Interquartile range (Q3 - Q1)", &ts_iqr});
}
