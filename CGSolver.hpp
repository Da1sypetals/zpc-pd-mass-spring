#pragma once

#include <eigen3/Eigen/Sparse>
#include <iostream>
#include <zensim/container/Vector.hpp>
#include <zensim/omp/execution/ExecutionPolicy.hpp>
#include <zensim/math/matrix/SparseMatrixOperations.hpp>


// works, frozen
class ConjugateGradient {

    zs::OmpExecutionPolicy &ompPolicy;
    zs::SparseMatrix<float> &A;
    zs::Vector<float> &b;
    size_t n;

  private:
    void copy(zs::Vector<float> &from, zs::Vector<float> &to) {
        if (from.size() != to.size()) {
            throw std::runtime_error("vector dimensions does not match");
        }
        ompPolicy(zs::range(from.size()),
                  [from = zs::view<zs::execspace_e::openmp>(from),
                   to = zs::view<zs::execspace_e::openmp>(to)](auto i) mutable {
                      to[i] = from[i];
                  });
    }
    void axpy(float alpha, zs::Vector<float> &q, float beta,
              zs::Vector<float> &w, zs::Vector<float> &result) {

        if (q.size() != w.size()) {
            throw std::runtime_error("vector dimensions does not match");
        }

        ompPolicy(
            zs::range(q.size()),
            [alpha, beta, q = zs::view<zs::execspace_e::openmp>(q),
             w = zs::view<zs::execspace_e::openmp>(w),
             result = zs::view<zs::execspace_e::openmp>(result)](
                auto i) mutable { result[i] = alpha * q[i] + beta * w[i]; });
    }
    float square(zs::Vector<float> &q) {
        zs::Vector<float> res(1);
        zs::reduce(ompPolicy, q.begin(), q.end(), res.begin(), 0,
                   [](const float &prev, const float &cur) {
                       return prev + cur * cur;
                   });
        return *(res.begin());
    }
    float dot(zs::Vector<float> &q, zs::Vector<float> &w) {

        if (q.size() != w.size()) {
            throw std::runtime_error("vector dimensions does not match");
        }

        float res = 0;
        // ??? sequential here?
        for (int i = 0; i < q.size(); ++i) {
            res += q[i] * w[i];
        }
        return res;
    }

    void scale(float scalar, zs::Vector<float> &q) {
        zs::transform(ompPolicy, q.begin(), q.end(),
                      [scalar](const float &a) { return scalar * a; });
    }

    void _Solve_Impl(zs::Vector<float> &x, float tol = 1e-6f,
                     size_t max_iter = UINT32_MAX) {

        std::fill_n(x.begin(), x.size(), 0);

        float alpha, beta;
        zs::Vector<float> r(n);
        zs::Vector<float> r_next(n);
        zs::Vector<float> p(n);
        zs::Vector<float> Ap(n);

        // r = b - Ax
        zs::Vector<float> Ax(n);

        zs::spmv(ompPolicy, A, x, Ax);
        axpy(1, b, -1, Ax, r);


        // p = r
        copy(r, p);

        // iterations?
        for (size_t k = 0; k < max_iter; k++) {

            // Ap = A * p
            zs::spmv(ompPolicy, A, p, Ap);
            float r_sqr = square(r);
            float pt_a_p = dot(p, Ap);

            // alpha = ((r.dot(r)) / (p.dot(A * p)));
            alpha = r_sqr / pt_a_p;

            // x += alpha * p;
            axpy(1, x, alpha, p, x);

            // r_next = r - alpha * A * p;
            axpy(1, r, -alpha, Ap, r_next);

            float rnext_sqr = square(r_next);

            if (zs::sqrt(rnext_sqr) < tol) {
                return;
            }

            beta = rnext_sqr / r_sqr;

            // p = beta * p + r_next;
            axpy(beta, p, 1, r_next, p);

            // r = r_next;
            copy(r_next, r);


        }
    }

  public:
    ConjugateGradient(zs::OmpExecutionPolicy &_ompPolicy,
                      zs::SparseMatrix<float> &_A, zs::Vector<float> &_b)
        : A(_A), b(_b), ompPolicy(_ompPolicy) {

        // require matrix to be SPD.

        if (not(A.cols() == A.cols() and A.cols() == b.size())) {
            std::cerr << "Dimensions does not match!";
            std::exit(1);
        }
        n = b.size();
    }

    void Solve(zs::Vector<float> &x) {
        if (x.size() != n) {
            std::cerr << "Dimensions does not match!";
            std::exit(1);
        }

        _Solve_Impl(x);
    }

    zs::Vector<float> Solve() {
        zs::Vector<float> x(n);
        _Solve_Impl(x);
        return x;
    }
};
