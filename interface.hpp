#include "memory"
#include "vector"
#include <glm/vec3.hpp>

#include "SolverPreprocessor.hpp"

class Interface {

    std::shared_ptr<Solver> solver;

  public:
    void init(std::vector<glm::vec3> &vertices,
              std::vector<glm::ivec3> &indices) {

        // ---

        int nside;
        float size;
        float k;
        int n_iter;

        std::cout << "n of vertex each side, cloth size, stiffness k, number "
                     "of iterations / frame"
                  << std::endl;
        std::cin >> nside >> size >> k >> n_iter;

        // ---

        std::shared_ptr<Cloth> cloth = std::make_shared<Cloth>(nside, size, k);

        solver = std::make_shared<Solver>(cloth, n_iter);

        std::shared_ptr<SolverPreprocessor> pre =
            std::make_shared<SolverPreprocessor>(solver);
        pre->Init();
        solver->AddFixed(0, 0);
        solver->AddFixed(0, nside - 1);

        // init vertices and indices
        vertices.resize(cloth->numVertex);
        indices.resize(0);
        indices.reserve((cloth->nside - 1) * (cloth->nside - 1) * 3);

        for (int irow = 0; irow < cloth->nside - 1; irow++) {
            for (int icol = 0; icol < cloth->nside - 1; icol++) {

                int i1 = cloth->index(irow, icol);
                int i2 = cloth->index(irow, icol + 1);
                int i3 = cloth->index(irow + 1, icol);
                int i4 = cloth->index(irow + 1, icol + 1);

                indices.push_back({i1, i2, i3});
                indices.push_back({i1, i3, i4});
                indices.push_back({i1, i2, i4});
            }
        }

        fmt::print("init done\n");

        for (int i = 0; i < solver->cloth->numVertex; i++) {
            vertices[i] = {solver->x[3 * i], solver->x[3 * i + 1],
                           solver->x[3 * i + 2]};
        }

    }

    void update(std::vector<glm::vec3> &vertices) {

        solver->Step();

        // copy data

        for (int i = 0; i < solver->cloth->numVertex; i++) {
            vertices[i] = {solver->x[3 * i], solver->x[3 * i + 1],
                           solver->x[3 * i + 2]};
        }
    }
};
