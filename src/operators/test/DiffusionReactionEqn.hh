
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
    TensorFunc K; // diffusion coef
    VectorFunc u() { // flux
        return [=](Node const & x, double t = 0.) {
            return -(K(x, t) * pGrad(x, t));
        };
    }
    ScalarFunc f() {
        return [=](Node const & x, double t = 0.) {
            auto K0 = K(x, t);
            auto pHess0 = pHess(x, t);
            return -(K0(0, 0) * pHess0(0, 0) + K0(1, 1) * pHess0(1, 1) + K0(2, 2) * pHess0(2, 2)) + c * p(x, t);
        };
    }
};

#endif