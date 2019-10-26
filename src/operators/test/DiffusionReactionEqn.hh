#ifndef DIFFUSION_REACTION_EQN_
#define DIFFUSION_REACTION_EQN_

#include <unordered_map>
#include "SingletonLogger.hpp"
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
    Node abc() {
        return abc_;
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
        size_t numbOfMaterials = 1, size_t matIndex = 0, 
        std::array<double, 4> abcd = { 0., 0., 0., 1. },
        double k = 1., double c = 0.
    ) : eqns_(numbOfMaterials) {
        eqns_[matIndex] = DiffusionReactionEqnLinear(abcd, k, c);
    }
    DiffusionReactionEqnPwLinear& addPiece(double k, Node const & p, Node const & n, size_t matIndex, size_t matIndexNew) {
        auto& logger = SingletonLogger::instance();
        auto& eqn = eqns_[matIndex];
        std::array<Node, 3> x;
        // pick 3 pts
        auto e = Node(1., 0., 0.);
        for (auto const & u : { Node(0., 1., 0.), Node(0., 0., 1.) })
            if (std::fabs(e * n) > std::fabs(u * n)) 
                e = u;
        auto t1 = e - (e * n / (n * n)) * n; // https://math.stackexchange.com/a/1681815/231246
        t1 /= sqrt(t1 * t1);
        auto t2 = t1 ^ n;
        t2 /= sqrt(t2 * t2);
        x[0] = p;
        x[1] = p + t1;
        x[2] = p + t2;
        auto fpEqual = [](double a, double b, double tol = 1e-8) { return std::fabs(a - b) < tol; };
        auto isInPlane = [&](Node const & v) { return fpEqual(n * (p - v), 0.); };
        if (!fpEqual((t1 ^ t2) * (t1 ^ t2), 1.))
            logger.wrn("points lie on a line");
        for (auto const & v : x)
            if (!isInPlane(v))
                logger.wrn("bad interface point, " + std::to_string(std::fabs(n * (p - v))));
        Amanzi::WhetStone::DenseMatrix mtx(4, 4);
        Amanzi::WhetStone::DenseVector rhs(4);
        // set up rhs
        rhs(0) = -(eqn.u(p) * n / k);
        for (size_t i : { 1, 2, 3 })
            rhs(i) = eqn.p(x[i - 1]);
        // set up mtx
        for (size_t i : { 0, 1, 2 })
            mtx(0, i) = n[i];
        mtx(0, 3) = 0.;
        for (size_t i : { 1, 2, 3 }) {
            for (size_t j : { 0, 1, 2 })
                mtx(i, j) = x[i - 1][j];
            mtx(i, 3) = 1.;
        }
        // recover sln
        mtx.Inverse();
        auto sln = mtx * rhs;
        DiffusionReactionEqnLinear newEqn({ sln(0), sln(1), sln(2), sln(3) }, k, c());
        if (!fpEqual(eqn.p(p) - newEqn.p(p), 0.))
            logger.wrn("solution is not cont");
        if (!fpEqual((eqn.u(p) - newEqn.u(p)) * n, 0.))
            logger.wrn("normal flux is not cont");
        eqns_[matIndexNew] = newEqn;
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

class DiffusionReactionEqnQuadratic : public DiffusionReactionEqn { // a(x^2 + y^2 + z^2 - r^2)
protected:
    Tensor K_;
    double a_, r_, c_;
public:
    DiffusionReactionEqnQuadratic(        
        double a = 1., double r = 1.,
        double k = 1., double c = 0.
    ) 
    : c_(c)
    , a_(a)
    , r_(r)
    , K_(3, 2) {
        K_.MakeDiagonal(k); 
    }
    double a() const {
        return a_;
    }
    double p(Node const & x, size_t matIndex = 0) final {
        return a_ * (x * x - r_ * r_);
    }
    Node pGrad(Node const & x, size_t matIndex = 0) final {
        return 2. * a_ * x;
    }
    Tensor K(Node const &, size_t matIndex = 0) final {
        return K_;
    }
    double f(Node const & x, size_t matIndex = 0) final {
        return c_ * p(x, matIndex) - 6. * a_ * K_(0, 0);
    }
    double c() final {
        return c_;
    }
};

class DiffusionReactionEqnPwQuadratic : public DiffusionReactionEqn {
    std::array<DiffusionReactionEqnQuadratic, 2> eqns_;
public:
    DiffusionReactionEqnPwQuadratic(double r, double k1 = 1., double k2 = 1., double c = 0.) {
        eqns_[0] = DiffusionReactionEqnQuadratic(1., r, k1, c);
        eqns_[1] = DiffusionReactionEqnQuadratic(k1 / k2, r, k2, c);
    }
    double a(size_t i) const {
        return eqns_.at(i).a();
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