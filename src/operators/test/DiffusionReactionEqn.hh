#ifndef DIFFUSION_REACTION_EQN_
#define DIFFUSION_REACTION_EQN_

#include <functional>
#include "Tensor.hh"
#include "Point.hh"

using Node = Amanzi::AmanziGeometry::Point;
using Tensor = Amanzi::WhetStone::Tensor;
using ScalarFunc = std::function<double(Node const &, double)>;
using VectorFunc = std::function<Node(Node const &, double)>;
using TensorFunc = std::function<Tensor(Node const &, double)>;

struct DiffusionReactionEqn {
    ScalarFunc p; // soln
    VectorFunc pGrad; // and its gradient
    TensorFunc pHess; // and its hessian
    double c; // reaction coef
    Tensor K; // diffusion coef
    VectorFunc u() { // flux
        return [=](Node const & x, double t = 0.) {
            return -(K * pGrad(x, t));
        };
    }
    ScalarFunc f() {
        return [=](Node const & x, double t = 0.) {
            auto pHess0 = pHess(x, t);
            return -(K(0, 0) * pHess0(0, 0) + K(1, 1) * pHess0(1, 1) + K(2, 2) * pHess0(2, 2)) + c * p(x, t);
        };
    }
};

#endif