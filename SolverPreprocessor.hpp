#ifndef SOLVER_PREP_HPP
#define SOLVER_PREP_HPP

#include "CGSolver.hpp"
#include "Cloth.hpp"
#include "Config.hpp"
#include "Solver.hpp"
#include "Types.hpp"
#include <Eigen/Sparse>
#include <memory>
#include <utility>
#include <vector>

std::vector<Eigen::Triplet<float>> ToTriplets(Eigen::SparseMatrix<float> &M) {
    std::vector<Eigen::Triplet<float>> v;
    for (int i = 0; i < M.outerSize(); i++) {
        for (typename Eigen::SparseMatrix<float>::InnerIterator it(M, i); it;
             ++it) {
            v.emplace_back(it.row(), it.col(), it.value());
        }
    }

    return v;
}

class SolverPreprocessor {

    eg::SparseMatrix<float> Y;
    std::vector<eg::Triplet<float>> JTriplets;

  public:
    std::shared_ptr<Solver> solver;

    explicit SolverPreprocessor(std::shared_ptr<Solver> _solver)
        : solver(_solver){};

    void Init_Y() {
        // M
        std::vector<eg::Triplet<float>> MTriplets;
        for (int ivertex = 0; ivertex < solver->cloth->numVertex; ivertex++) {
            for (int j = 0; j < 3; j++) {
                MTriplets.push_back(eg::Triplet<float>(
                    3 * ivertex + j, 3 * ivertex + j, solver->cloth->mass));
            }
        }
        eg::SparseMatrix<float> M(3 * solver->cloth->numVertex,
                                  3 * solver->cloth->numVertex);
        M.setFromTriplets(MTriplets.begin(), MTriplets.end());

        // L
        std::vector<eg::Triplet<float>> LTriplets;
        for (auto &&con : solver->cloth->constraints) {
            for (int j = 0; j < 3; j++) {
                LTriplets.push_back(eg::Triplet<float>(
                    3 * con.istart + j, 3 * con.istart + j, solver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(
                    3 * con.istart + j, 3 * con.iend + j, -solver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(
                    3 * con.iend + j, 3 * con.iend + j, solver->cloth->k));
                LTriplets.push_back(eg::Triplet<float>(
                    3 * con.iend + j, 3 * con.istart + j, -solver->cloth->k));
            }
        }
        eg::SparseMatrix<float> L(3 * solver->cloth->numVertex,
                                  3 * solver->cloth->numVertex);
        L.setFromTriplets(LTriplets.begin(), LTriplets.end());

        // Y
        Y = M + dt2 * L;
    }

    void Init_J() {
        for (int iconstraint = 0; iconstraint < solver->cloth->numConstraint;
             iconstraint++) {

            auto &con = solver->cloth->constraints[iconstraint];

            for (int j = 0; j < 3; j++) {
                JTriplets.push_back(eg::Triplet<float>(3 * con.istart + j,
                                                       3 * iconstraint + j,
                                                       -solver->cloth->k));
                JTriplets.push_back(eg::Triplet<float>(
                    3 * con.iend + j, 3 * iconstraint + j, solver->cloth->k));
            }
        }
    }

    void Init_zs_mat() {

        auto YTriplets = ToTriplets(Y);

        std::vector<int> rowY, colY, valY;
        rowY.reserve(YTriplets.size());
        colY.reserve(YTriplets.size());
        valY.reserve(YTriplets.size());

        std::vector<int> rowJ, colJ, valJ;
        rowJ.reserve(JTriplets.size());
        colJ.reserve(JTriplets.size());
        valJ.reserve(JTriplets.size());

        for (auto &&ytrip : YTriplets) {
            rowY.push_back(ytrip.row());
            colY.push_back(ytrip.col());
            valY.push_back(ytrip.value());
        }

        for (auto &&jtrip : JTriplets) {
            rowJ.push_back(jtrip.row());
            colJ.push_back(jtrip.col());
            valJ.push_back(jtrip.value());
        }

        solver->J.build(solver->ompPolicy, 3 * solver->cloth->numVertex,
                        3 * solver->cloth->numConstraint, rowJ, colJ, valJ);

        solver->Y.build(solver->ompPolicy, 3 * solver->cloth->numVertex,
                        3 * solver->cloth->numVertex, rowY, colY, valY);
    }

    void CreateVec() {
        solver->d =
            zs::Vector<float>(solver->cloth->numConstraint * 3, zs::memsrc_e::host);

        solver->x =
            zs::Vector<float>(solver->cloth->numVertex * 3, zs::memsrc_e::host);

        solver->x_prev =
            zs::Vector<float>(solver->cloth->numVertex * 3, zs::memsrc_e::host);

        solver->y =
            zs::Vector<float>(solver->cloth->numVertex * 3, zs::memsrc_e::host);
        solver->f_external =
            zs::Vector<float>(solver->cloth->numVertex * 3, zs::memsrc_e::host);

        solver->b =
            zs::Vector<float>(solver->cloth->numVertex * 3, zs::memsrc_e::host);
    }

    void InitVec() {
        for (int irow = 0; irow < solver->cloth->nside; irow++) {
            for (int icol = 0; icol < solver->cloth->nside; icol++) {

                int idx = solver->index(irow, icol);

                // init pos {
                solver->x[3 * idx] =
                    solver->cloth->size * static_cast<float>(irow) /
                    static_cast<float>(solver->cloth->nside - 1);
                solver->x[3 * idx + 1] = 0;

                solver->x[3 * idx + 2] =
                    solver->cloth->size * static_cast<float>(icol) /
                    static_cast<float>(solver->cloth->nside - 1);
                // }

                // init force {
                solver->f_external[3 * idx] = 0;
                solver->f_external[3 * idx + 1] = gravity;
                solver->f_external[3 * idx + 2] = 0;
                // }
            }
        }

        // init host, copy from host to device x and x_prev
        solver->x_prev = solver->x.clone({zs::memsrc_e::host, 0});

    }

    void Init() {

        solver->cloth->InitConstraints();

        CreateVec();
        InitVec();

        Init_J();
        Init_Y();
        Init_zs_mat();

        solver->cg = std::make_unique<ConjugateGradient>(solver->ompPolicy,
                                                         solver->Y, solver->b);

        solver->initialized = true;
    }
};

#endif