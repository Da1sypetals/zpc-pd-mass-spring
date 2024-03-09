#include "Config.hpp"
#include "SolverPreprocessor.hpp"
#include <iostream>
#include <memory>
#include <zensim/container/TileVector.hpp>
#include <zensim/container/Vector.hpp>

#include <zensim/execution/ExecutionPolicy.hpp>
#include <zensim/math/matrix/SparseMatrix.hpp>
#include <zensim/math/matrix/SparseMatrixOperations.hpp>
#include <zensim/omp/execution/ExecutionPolicy.hpp>

#include "CGSolver.hpp"

class Solver {

    friend class SolverPreprocessor;

  private:
    bool initialized;

  public:
    int n_iter;

    std::shared_ptr<Cloth> cloth;

    // matrices
    zs::SparseMatrix<float> Y;
    zs::SparseMatrix<float> J;

    zs::Vector<float> x;
    zs::Vector<float> x_prev;
    zs::Vector<float> d;
    zs::Vector<float> b;
    zs::Vector<float> y;
    zs::Vector<float> f_external;

    std::vector<std::pair<int, eg::Vector3f>> fixed;

    // policies
    zs::OmpExecutionPolicy ompPolicy;

    std::unique_ptr<ConjugateGradient> cg;

    // Utility {

    __host__ __device__ int index(int irow, int icol) {
        return irow * cloth->nside + icol;
    }

    // }

    // Inner {

    void LocalStep() {
        ompPolicy(
            zs::range(cloth->numConstraint),
            [this, x = zs::view<zs::execspace_e::openmp>(x),
             d = zs::view<zs::execspace_e::openmp>(d),
             numConstraint = cloth->numConstraint](auto i) mutable {
                auto &con =
                    this->cloth
                        ->constraints[i]; // range of i = constraints.size();

                float d0 = x[con.iend * 3 + 0] - x[con.istart * 3 + 0];
                float d1 = x[con.iend * 3 + 1] - x[con.istart * 3 + 1];
                float d2 = x[con.iend * 3 + 2] - x[con.istart * 3 + 2];

                eg::Vector3f dir;
                dir << d0, d1, d2;
                dir.normalize();

                d[i * 3 + 0] = dir[0] * con.restLength;
                d[i * 3 + 1] = dir[1] * con.restLength;
                d[i * 3 + 2] = dir[2] * con.restLength;
            });
    }

    void GlobalStep() {

        // b = dt2 * J * d + f_ext * dt2 + y

        // b = J * d;
        zs::spmv(ompPolicy, J, d, b);
        // b = dt2 * b + f_ext * dt2 + y
        ompPolicy(zs::range(b.size()),
                  [b = zs::view<zs::execspace_e::openmp>(b),
                   f_external = zs::view<zs::execspace_e::openmp>(f_external),
                   y = zs::view<zs::execspace_e::openmp>(y)](auto i) mutable {
                      b[i] = dt2 * b[i] + f_external[i] * dt2 + y[i];
                  });

        // x = arg(Ax = b)
        cg->Solve(x);
    }

    // }

    // API {

    Solver(std::shared_ptr<Cloth> _cloth, int _n_iter)
        : cloth(_cloth), n_iter(_n_iter) {
        initialized = false;

        ompPolicy = zs::omp_exec();
    }

    void Step() {

        //        y = ((2 - preservation) * x - (1 - preservation) * x_prev);

        ompPolicy(zs::range(cloth->numVertex * 3),
                  [y = zs::view<zs::execspace_e::openmp>(y),
                   x = zs::view<zs::execspace_e::openmp>(x),
                   x_prev = zs::view<zs::execspace_e::openmp>(x_prev)](auto i) mutable {
                      y[i] = (2 - preservation) * x[i] -
                             (1 - preservation) * x_prev[i];
                      x_prev[i] = x[i];
                  });


        for (int iter = 0; iter < n_iter; iter++) {
            LocalStep();

            GlobalStep();
        }

        for (auto &&[ifixed, fixpos] : fixed) {
            x[3 * ifixed] = fixpos(0);
            x[3 * ifixed + 1] = fixpos(1);
            x[3 * ifixed + 2] = fixpos(2);
        }
    }

    void AddFixed(int irow, int icol) {

        if (not initialized) {
            std::cerr << "Fatal: Not initialized, cannot add fixed vertex!"
                      << std::endl;
            exit(1);
        }

        int idx = index(irow, icol);
        fixed.push_back(std::make_pair(
            index(irow, icol),
            eg::Vector3f(x[3 * idx], x[3 * idx + 1], x[3 * idx + 2])));
    }

    // }
};
