#include <armadillo>
#include <carma/carma.h>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// global variables
arma::vec P_025 = {0.25};
arma::vec P_050 = {0.50};
arma::vec P_075 = {0.75};

// Max
arma::vec C_ts_max(const arma::mat& mtx) {
  return arma::max(mtx, 1);
}

// Min
arma::vec C_ts_min(const arma::mat& mtx) {
  return arma::min(mtx, 1);
}

// Mean
arma::vec C_ts_mean(const arma::mat& mtx) {

  return arma::mean(mtx, 1);
}

// Median
arma::vec C_ts_median(const arma::mat& mtx) {

  return arma::median(mtx, 1);
}

// Sum
arma::vec C_ts_sum(const arma::mat& mtx) {

  return arma::sum(mtx, 1);
}

// Standard deviation
arma::vec C_ts_std(const arma::mat& mtx) {

  return arma::stddev(mtx, 0, 1);
}

// Skew
arma::vec C_ts_skew(const arma::mat& mtx) {
  // skewness based on adjusted Fisher-Pearson coefficient
  const int n = mtx.n_cols;
  const double expS = 1.5;

  // adjusted factor
  double adj_factor = sqrt((n*(n-1)))/n-2;

  arma::vec m3 = arma::sum(arma::pow(mtx.each_col()- arma::mean(mtx, 1), 3), 1)/n;
  arma::vec s = arma::pow(arma::sum(arma::pow(mtx.each_col()- arma::mean(mtx, 1), 2), 1)/n, expS);

  return (m3/s)*adj_factor;
}

// Kurt
arma::vec C_ts_kurt(const arma::mat& mtx) {
  // kurtosis based on pearson’s definition is used (normal ==> 3.0)
  const int n = mtx.n_cols;

  arma::vec m4 = arma::sum(arma::pow(mtx.each_col()- arma::mean(mtx, 1), 4), 1);
  arma::vec m2 = arma::pow(arma::sum(arma::pow(mtx.each_col()- arma::mean(mtx, 1), 2), 1), 2);

  return n*m4/m2;
}

// Amplitude
arma::vec C_ts_amplitude(const arma::mat& mtx) {
  return arma::max(mtx, 1) - arma::min(mtx, 1);
}

// F-Slope
arma::vec C_ts_fslope(const arma::mat& mtx) {

  return arma::max(arma::abs(arma::diff(mtx, 1, 1)), 1);
}

// Absolute sum
arma::vec C_ts_abs_sum(const arma::mat& mtx) {

  return arma::sum(arma::abs(mtx), 1);
}

// AMD
arma::vec C_ts_amd(const arma::mat& mtx) {

  return arma::mean(arma::abs(arma::diff(mtx, 1, 1)), 1);
}

// MSE
arma::vec C_ts_mse(const arma::mat& mtx) {
  arma::mat metrics = mtx.t();

  return arma::mean(arma::pow(arma::abs(arma::trans(arma::fft(metrics))), 2), 1);
}

// FQR
arma::vec C_ts_fqr(const arma::mat& mtx) {

  return arma::quantile(mtx, P_025, 1);
}

// SQR
arma::vec C_ts_sqr(const arma::mat& mtx) {

  return arma::quantile(mtx, P_050, 1);
}

// TQR
arma::vec C_ts_tqr(const arma::mat& mtx) {

  return arma::quantile(mtx, P_075, 1);
}

// IQR
arma::vec C_ts_iqr(const arma::mat& mtx) {

  arma::vec res = C_ts_tqr(mtx) - C_ts_fqr(mtx);

  return (res);
}

void init_basic_features(py::module &m) {
    m.def("C_ts_max", &C_ts_max, "Time-series max");
    m.def("C_ts_min", &C_ts_min, "Time-series min");
    m.def("C_ts_mean", &C_ts_mean, "Time-series mean");    
    m.def("C_ts_median", &C_ts_median, "Time-series median");
    m.def("C_ts_sum", &C_ts_sum, "Time-series values sum");
    m.def("C_ts_std", &C_ts_std, "Time-series standard deviation");
    m.def("C_ts_skew", &C_ts_skew, "Time-series skew");
    m.def("C_ts_kurt", &C_ts_kurt, "Time-series kurt");
    m.def("C_ts_amplitude", &C_ts_amplitude, "Time-series amplitude");
    m.def("C_ts_fslope", &C_ts_fslope, "Time-series F-Slope");
    m.def("C_ts_abs_sum", &C_ts_abs_sum, "Time-series absolute sum");
    m.def("C_ts_amd", &C_ts_amd, "Time-series AMD");
    m.def("C_ts_mse", &C_ts_mse, "Time-series MSE");
    m.def("C_ts_fqr", &C_ts_fqr, "Time-series FQR");
    m.def("C_ts_sqr", &C_ts_sqr, "Time-series SQR");
    m.def("C_ts_tqr", &C_ts_tqr, "Time-series TQR");
    m.def("C_ts_iqr", &C_ts_iqr, "Time-series IQR");
}
