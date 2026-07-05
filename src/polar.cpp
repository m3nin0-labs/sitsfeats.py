//
// Copyright (C) 2024-2026 sitsfeats.py.
//
// sitsfeats.py is free software; you can redistribute it and/or modify it
// under the terms of the MIT License; see LICENSE file for more details.
//
// Polar time-series metrics (Körting polar-plot representation, after stmetrics).
// Each row is mapped to a closed polygon whose vertex `i` sits at angle `2*pi*i/N` with 
// radius `|x_i|`. Since radii are non-negative and vertices are ordered by angle, 
// the polygon is star-shaped about the origin and hence simple, so these closed forms 
// reproduce the shapely-based `stmetrics` values.
//

#include "metrics.h"

#include <array>
#include <cmath>
#include <vector>

namespace {

// PI constant
constexpr double PI = 3.14159265358979323846;

// Point type
using Pt = std::array<double, 2>;

// Cartesian polygon coordinates for one row: vertex `i` = `(|x_i| cos t, |x_i| sin t)`.
void polar_xy(const Mat& m, Eigen::Index row, std::vector<double>& xs,
              std::vector<double>& ys) {
  // get the number of columns
  const Eigen::Index N = m.cols();

  // resize the vectors
  xs.resize(static_cast<std::size_t>(N));
  ys.resize(static_cast<std::size_t>(N));

  // iterate over the columns and compute the coordinates
  for (Eigen::Index i = 0; i < N; ++i) {
    const double r = std::abs(m(row, i));
    const double t = 2.0 * PI * static_cast<double>(i) / static_cast<double>(N);

    // compute the coordinates
    xs[static_cast<std::size_t>(i)] = r * std::cos(t);
    ys[static_cast<std::size_t>(i)] = r * std::sin(t);
  }
}

// Sutherland-Hodgman clip of a polygon to the half-plane sign * coord[axis] >= 0.
std::vector<Pt> clip_halfplane(const std::vector<Pt>& poly, int axis, double sign) {
  // create the output vector
  std::vector<Pt> out;

  // get the number of vertices
  const std::size_t n = poly.size();

  for (std::size_t i = 0; i < n; ++i) {
    const Pt& cur = poly[i];
    const Pt& nxt = poly[(i + 1) % n];
    
    const bool ci = cur[axis] * sign >= 0.0;
    const bool ni = nxt[axis] * sign >= 0.0;
    
    if (ci) {
      out.push_back(cur);
    }

    if (ci != ni) {
      const double t = cur[axis] / (cur[axis] - nxt[axis]);

      out.push_back({cur[0] + t * (nxt[0] - cur[0]),
                     cur[1] + t * (nxt[1] - cur[1])});
    }
  }

  // return!
  return out;
}

double shoelace_abs(const std::vector<Pt>& poly) {
  // get the number of vertices
  const std::size_t n = poly.size();

  // if the number of vertices is less than 3, return 0.0
  if (n < 3) {
    return 0.0;
  }

  // initialize the area
  double a = 0.0;

  // iterate over the vertices and compute the area
  for (std::size_t i = 0; i < n; ++i) {
    const Pt& p = poly[i];
    const Pt& q = poly[(i + 1) % n];

    a += p[0] * q[1] - q[0] * p[1];
  }

  // return!
  return std::abs(a) * 0.5;
}

// Areas of the polygon in the four seasonal quadrants (q1..q4).
std::array<double, 4> quadrant_areas(const Mat& m, Eigen::Index row) {
  // create the output array
  std::array<double, 4> out;

  // create the vectors
  std::vector<double> xs, ys;

  // compute the polar coordinates
  polar_xy(m, row, xs, ys);

  // create the polygon
  std::vector<Pt> poly(xs.size());

  // iterate over the vertices and create the polygon
  for (std::size_t i = 0; i < xs.size(); ++i) {
    poly[i] = {xs[i], ys[i]};
  }

  // create the signs
  const std::array<std::array<double, 2>, 4> signs = {
    {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}
  };

  // create the areas
  std::array<double, 4> areas{};

  // iterate over the quadrants and compute the areas
  for (int q = 0; q < 4; ++q) {
    auto clipped = clip_halfplane(
        clip_halfplane(poly, 0, signs[static_cast<std::size_t>(q)][0]), 1,
        signs[static_cast<std::size_t>(q)][1]
    );
    
    // compute the area
    areas[static_cast<std::size_t>(q)] = shoelace_abs(clipped);
  }

  // return
  return areas;
}

// Area of the polar polygon (shoelace closed form).
Eigen::VectorXd ts_area_ts(const Mat& m) {
  // get the number of rows
  const Eigen::Index n = m.rows();
  const Eigen::Index N = m.cols();

  // compute the sine of 2*PI/N
  const double s = std::sin(2.0 * PI / static_cast<double>(N));

  // create the output vector
  Eigen::VectorXd out(n);

  // iterate over the rows and compute the area
  for (Eigen::Index r = 0; r < n; ++r) {
    // initialize the accumulator
    double acc = 0.0;

    // calculate sum of products
    for (Eigen::Index i = 0; i < N; ++i) {
      acc += std::abs(m(r, i)) * std::abs(m(r, (i + 1) % N));
    }

    // compute the area
    out(r) = 0.5 * s * acc;
  }

  // return!
  return out;
}

// Angle of the maximum value, using linspace(0, 2*pi, N) (stmetrics).
Eigen::VectorXd ts_angle(const Mat& m) {
  const Eigen::Index n = m.rows();
  const Eigen::Index N = m.cols();

  // compute the step size
  const double step =
      (N > 1) ? (2.0 * PI / static_cast<double>(N - 1)) : 0.0;

  // create the output vector
  Eigen::VectorXd out(n);

  // iterate over the rows and compute the angle
  for (Eigen::Index r = 0; r < n; ++r) {
    // initialize the maximum index and value
    Eigen::Index amax = 0;
    double best = std::abs(m(r, 0));

    // iterate over the columns and find the maximum value
    for (Eigen::Index i = 1; i < N; ++i) {
      // get the value
      const double v = std::abs(m(r, i));

      // update the maximum value and index
      if (v > best) {
        best = v;
        amax = i;
      }
    }

    // compute the angle
    out(r) = step * static_cast<double>(amax);
  }

  // return!
  return out;
}

// Mean distance from polygon vertices to its area-centroid. Matches stmetrics,
// including Shapely's exterior ring repeating the first vertex (averaged over
// N + 1 coordinates).
Eigen::VectorXd ts_gyration_radius(const Mat& m) {
  // get the number of rows
  const Eigen::Index n = m.rows();
  const Eigen::Index N = m.cols();

  // create the output vector
  Eigen::VectorXd out(n);

  // coordinate vectors
  std::vector<double> xs, ys;

  // iterate over the rows and compute the gyration radius
  for (Eigen::Index r = 0; r < n; ++r) {
    // compute the polar coordinates
    polar_xy(m, r, xs, ys);

    // initialize the accumulators
    double cross_sum = 0.0, cxnum = 0.0, cynum = 0.0;

    // iterate over the vertices and compute the cross products
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
      // get the next vertex
      const std::size_t j = (i + 1) % static_cast<std::size_t>(N);

      // compute the cross product
      const double cr = xs[i] * ys[j] - xs[j] * ys[i];

      // update the accumulators
      cross_sum += cr;
      cxnum += (xs[i] + xs[j]) * cr;
      cynum += (ys[i] + ys[j]) * cr;
    }

    // compute the area
    const double area = 0.5 * cross_sum;
    const double cx = cxnum / (6.0 * area);
    const double cy = cynum / (6.0 * area);

    double dsum = 0.0, d0 = 0.0;

    // iterate over the vertices and compute the distances
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
      // compute the distance
      const double d = std::hypot(xs[i] - cx, ys[i] - cy);

      // update the accumulators
      dsum += d;

      // update the first distance
      if (i == 0) {
        d0 = d;
      }
    }

    // compute the gyration radius
    out(r) = (dsum + d0) / (static_cast<double>(N) + 1.0);
  }

  // return!
  return out;
}

// Cell shape index: perimeter^2 / (4 * pi * area).
Eigen::VectorXd ts_csi(const Mat& m) {
  // get the number of rows
  const Eigen::Index n = m.rows();

  // get the number of columns
  const Eigen::Index N = m.cols();

  // compute the area
  Eigen::VectorXd area = ts_area_ts(m);
  Eigen::VectorXd out(n);

  // create the coordinate vectors
  std::vector<double> xs, ys;

  // iterate over the rows and compute the cell shape index
  for (Eigen::Index r = 0; r < n; ++r) {
    // compute the polar coordinates
    polar_xy(m, r, xs, ys);

    // initialize the perimeter
    double per = 0.0;

    // iterate over the vertices and compute the perimeter
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
      // get the next vertex
      const std::size_t j = (i + 1) % static_cast<std::size_t>(N);

      // compute the distance
      per += std::hypot(xs[j] - xs[i], ys[j] - ys[i]);
    }

    // compute the cell shape index
    out(r) = per * per / (4.0 * PI * area(r));
  }

  // return!
  return out;
}

Eigen::VectorXd ts_area_q(const Mat& m, int q) {
  // get the number of rows
  const Eigen::Index n = m.rows();

  // create the output vector
  Eigen::VectorXd out(n);

  // iterate over the rows and compute the area
  for (Eigen::Index r = 0; r < n; ++r) {
    // compute the area
    out(r) = quadrant_areas(m, r)[static_cast<std::size_t>(q)];
  }

  // return!
  return out;
}

// Quadrant 1 area
Eigen::VectorXd ts_area_q1(const Mat& m) { 
  return ts_area_q(m, 0);
}

// Quadrant 2 area
Eigen::VectorXd ts_area_q2(const Mat& m) { 
  return ts_area_q(m, 1);
}

// Quadrant 3 area
Eigen::VectorXd ts_area_q3(const Mat& m) { 
  return ts_area_q(m, 2);
}

// Quadrant 4 area
Eigen::VectorXd ts_area_q4(const Mat& m) { 
  return ts_area_q(m, 3);
}

// Standard deviation of the four seasonal quadrant areas (population, ddof=0).
Eigen::VectorXd ts_polar_balance(const Mat& m) {
  // get the number of rows
  const Eigen::Index n = m.rows();

  // create the output vector
  Eigen::VectorXd out(n);

  // iterate over the rows and compute the polar balance
  for (Eigen::Index r = 0; r < n; ++r) {
    // compute the quadrant areas
    const auto a = quadrant_areas(m, r);

    // compute the mean
    const double mean = (a[0] + a[1] + a[2] + a[3]) / 4.0;

    // compute the variance
    double var = 0.0;

    // iterate over the quadrants and compute the variance
    for (int q = 0; q < 4; ++q) {
      const double d = a[static_cast<std::size_t>(q)] - mean;
      var += d * d;
    }

    // compute the polar balance
    out(r) = std::sqrt(var / 4.0);
  }

  // return!
  return out;
}

}  // namespace

void register_polar_metrics(std::vector<MetricDescriptor>& reg) {
  reg.push_back({"area_ts", "polar", "Area of the polar-plot polygon", &ts_area_ts});
  reg.push_back({"angle", "polar", "Phase angle of the maximum", &ts_angle});
  reg.push_back({"gyration_radius", "polar", "Mean vertex distance to the centroid", &ts_gyration_radius});
  reg.push_back({"csi", "polar", "Cell shape index (perimeter^2 / 4*pi*area)", &ts_csi});

  reg.push_back({"area_q1", "polar", "Polygon area in quadrant 1 (upper-right)", &ts_area_q1});
  reg.push_back({"area_q2", "polar", "Polygon area in quadrant 2 (upper-left)", &ts_area_q2});
  reg.push_back({"area_q3", "polar", "Polygon area in quadrant 3 (lower-left)", &ts_area_q3});
  reg.push_back({"area_q4", "polar", "Polygon area in quadrant 4 (lower-right)", &ts_area_q4});

  reg.push_back({"polar_balance", "polar", "Std-dev of the four quadrant areas", &ts_polar_balance});
}
