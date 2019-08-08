#ifndef DIFFUSION_REACTION_EQN_
#define DIFFUSION_REACTION_EQN_

#include <unordered_map>
#include "Tensor.hh"
#include "Point.hh"

using Node = Amanzi::AmanziGeometry::Point;
using Tensor = Amanzi::WhetStone::Tensor;

class DiffusionReactionEqn {
public:
    virtual double p(Node const &, size_t matIndex = 0) = 0; // pressure
    virtual Node   pGrad(Node const &, size_t matIndex = 0) = 0; // its gradient
    virtual Tensor K(Node const &, size_t matIndex = 0) = 0;
    virtual double f(Node const &, size_t matIndex = 0) = 0;
    virtual double c() = 0;
    Node u(Node const & x, size_t matIndex = 0) { // flux
        return -(K(x, matIndex) * pGrad(x, matIndex));
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
    std::array<double, 4> abcd() {
        return { abc_[0], abc_[1], abc_[2], d_ };
    }
    double p(Node const & x, size_t matIndex = 0) final {
        return abc_ * x + d_;
    }
    Node pGrad(Node const &, size_t matIndex = 0) final {
        return abc_;
    }
    Tensor K(Node const &, size_t matIndex = 0) final {
        return K_;
    }
    double f(Node const & x, size_t matIndex = 0) final {
        return c_ * p(x, matIndex);
    }
    double c() final {
        return c_;
    }
};

class DiffusionReactionEqnPwLinear : public DiffusionReactionEqn {
    std::vector<DiffusionReactionEqnLinear> eqns_;
public:
    DiffusionReactionEqnPwLinear(
        std::array<double, 4> abcd = { 0., 0., 0., 1. },
        double k = 1., double c = 0.
    ) {
        DiffusionReactionEqnLinear eqn(abcd, k, c);
        eqns_.push_back(eqn);
    }
    DiffusionReactionEqnPwLinear& addPiece(double k, Node const & p, Node const & n) {
        eqns_.push_back(eqns_.back());
        return *this;
    }
    std::array<double, 4> abcd(size_t matIndex) {
        return eqns_[matIndex].abcd();
    }
    double p(Node const & x, size_t matIndex = 0) final {
        return eqns_.at(matIndex).p(x);
    }
    Node pGrad(Node const & x, size_t matIndex = 0) final {
        return eqns_.at(matIndex).pGrad(x);
    }
    Tensor K(Node const & x, size_t matIndex = 0) final {
        return eqns_.at(matIndex).K(x);
    }
    double f(Node const & x, size_t matIndex = 0) final {
        return eqns_.at(matIndex).f(x);
    }
    double c() final {
        return eqns_.front().c();
    }
};

#endif