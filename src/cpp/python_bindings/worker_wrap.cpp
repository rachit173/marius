//
// Created by Jason Mohoney on 4/9/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "worker.h"

namespace py = pybind11;

void init_worker(py::module &m) {
    m.def("worker", [](int argc, std::vector<std::string> argv) {

            argv[0] = "worker";
            std::vector<char *> c_strs;
            c_strs.reserve(argv.size());
            for (auto &s : argv) c_strs.push_back(const_cast<char *>(s.c_str()));

            worker_main(argc, c_strs.data());

        }, py::arg("argc"), py::arg("argv"), py::return_value_policy::reference);
}