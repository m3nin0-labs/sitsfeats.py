#include <armadillo>
#include <carma/carma.h>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_basic_features(py::module &m);
// void init_polar_features(py::module &m);

PYBIND11_MODULE(_sitsfeats, m)
{
    // Module metadata
    m.doc() = "(Internal) Time-series features";

    // Basic metrics
   init_basic_features(m);
}
