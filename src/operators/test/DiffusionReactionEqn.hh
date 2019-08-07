#ifndef DIFFUSION_REACTION_EQN_
#define DIFFUSION_REACTION_EQN_

#include <unordered_map>
#include "Tensor.hh"
#include "Point.hh"

using Node = Amanzi::AmanziGeometry::Point;
using Tensor = Amanzi::WhetStone::Tensor;

class DiffusionReactionEqn {
public:
    virtual double p(Node const &) = 0; // pressure
    virtual Node   pGrad(Node const &) = 0; // its gradient
    virtual Tensor K(Node const &) = 0;
    virtual double f(Node const &) = 0;
    virtual double c() = 0;
    Node u(Node const & x) { // flux
        return -(K(x) * pGrad(x));
    }
};

class DiffusionReactionEqnLinear : public DiffusionReactionEqn {
protected:
    Tensor K_;
    Node abc_;
    double d_, c_;
public:
    DiffusionReactionEqnLinear(        
        std::array<double, 4> abcd = { 0., 0., 0., 1. },
        double k = 1., double c = 0.
    ) 
    : c_(c)
    , abc_(Node(abcd[0], abcd[1], abcd[2]))
    , d_(abcd[3])
    , K_(3, 2) {
        K_.MakeDiagonal(k); 
    }
    double p(Node const & x) final {
        return abc_ * x + d_;
    }
    Node pGrad(Node const &) final {
        return abc_;
    }
    Tensor K(Node const &) final {
        return K_;
    }
    double f(Node const & x) final {
        return c_ * p(x);
    }
    double c() final {
        return c_;
    }
};

class DiffusionReactionEqnPwLinear : public DiffusionReactionEqn {
    // std::unordered_map<Node, size_t> c2r_;
    std::vector<DiffusionReactionEqnLinear> eqns_;
    size_t ctr2reg_(Node const & x) {
        if (eqns_.size() == 0) return 0;
        // ...
        // return c2r_.at(x)
        return 0;
    }
public:
    DiffusionReactionEqnPwLinear(
        std::array<double, 4> abcd = { 0., 0., 0., 1. },
        double k = 1., double c = 0.
    ) {
        DiffusionReactionEqnLinear eqn(abcd, k, c);
        eqns_.push_back(eqn);
    }
    double p(Node const & x) final {
        return eqns_[ctr2reg_(x)].p(x);
    }
    Node pGrad(Node const & x) final {
        return eqns_[ctr2reg_(x)].pGrad(x);
    }
    Tensor K(Node const & x) final {
        return eqns_[ctr2reg_(x)].K(x);
    }
    double f(Node const & x) final {
        return eqns_[ctr2reg_(x)].f(x);
    }
    double c() final {
        return eqns_.front().c();
    }
};

#endif