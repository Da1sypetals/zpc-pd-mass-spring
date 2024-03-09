#pragma once

#include <eigen3/Eigen/Sparse>

namespace eg = Eigen;

#include <zensim/container/Vector.hpp>
#include <zensim/container/TileVector.hpp>

using vec3 = zs::vec<float, 3>;

struct Constraint {
    int istart, iend;
    float restLength;

    Constraint() = default;

    Constraint(int _istart, int _iend, float _restLength) : istart(_istart), iend(_iend), restLength(_restLength) {}

};