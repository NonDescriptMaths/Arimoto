#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

double calculateCapacity(const std::vector<double>& p, const std::vector<std::vector<double>>& P, const std::vector<std::vector<double>>& W) {
    double C = 0;
    for (int i = 0; i < P.size(); ++i) {
        for (int j = 0; j < P[0].size(); ++j) {
            if (p[j] > 0) {
                C += p[j] * P[i][j] * log(W[j][i] / p[j]);
            }
        }
    }
    return C / log(2);
}

std::tuple<std::vector<double>, double, std::vector<std::vector<double>>> arimito(const std::vector<std::vector<double>>& P, const std::vector<double>& prior, int max_iterations = 1000, double thresh = 1e-12) {
    
    int n = P.size();
    int m = P[0].size();
    std::vector<std::vector<double>> W(m, std::vector<double>(n));
    std::vector<double> p = prior;
    std::vector<double> q(m);
    std::vector<std::vector<double>> p_route;
    p_route.push_back(p);

    bool converged = false;
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Maximise I(p,W;P) over W
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                double dot_product = std::inner_product(P[i].begin(), P[i].end(), p.begin(), 0.0);
                W[j][i] = (P[i][j] * p[j]) / dot_product;
            }
        }
        
        // Maximise I(p,W;P) over p
        std::vector<double> r(m);
        for (int j = 0; j < m; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += P[i][j] * log(W[j][i]);
            }
        r[j] = exp(sum);
        }

        double r_sum = std::accumulate(r.begin(), r.end(), 0.0);
        for (int i = 0; i < m; ++i) {
            q[i] = r[i] / r_sum;
        }
        
        p_route.push_back(q);

        double norm = std::inner_product(p.begin(), p.end(), q.begin(), 0.0, std::plus<double>(), [](double a, double b) { return pow(a - b, 2); });
        norm = sqrt(norm);

        if (norm < thresh) {
            converged = true;
            break;
        }
        
        p = q;
    }
    
    double C = calculateCapacity(p, P, W);
    if (!converged) {
            std::fill(p.begin(), p.end(), 0.0);
            double C = 0.0;
        }
    return std::make_tuple(p, C, p_route);
}

PYBIND11_MODULE(CArimito, m) {
    m.def("arimito", &arimito, "Arimoto algorithm");
}