#ifndef OPERATORS_PDE_DIFFUSION_MFD_ASC_H
#define OPERATORS_PDE_DIFFUSION_MFD_ASC_H

#include "Operator.hh"
#include "OperatorDefs.hh"
#include "PDE_Abstract.hh"
#include "WhetStoneDefs.hh"
#include "MFD3D_Diffusion.hh"
#include "MeshMini.hh"

inline bool fpEqual(double a, double b, double tol = 1e-8) {
    return std::fabs(a - b) < tol;
}

namespace Amanzi {
    namespace Operators {
        class PDE_DiffusionMFD_ASC : public PDE_Abstract {
        public:
            using Node = AmanziGeometry::Point;
            using ScalarFunc = std::function<double(Node const &, double)>;
            using TensorFuncInd = std::function<WhetStone::Tensor(size_t)>;
            using Predicate = std::function<bool(Node const &)>;
            enum class BCType { Dirichlet, Neumann };
            struct BC {
                BCType type;
                ScalarFunc f;
                Predicate p;
            };
            struct LocalSystem {
                WhetStone::DenseMatrix W; // inverse mass matrix
                WhetStone::DenseMatrix mB; // divergence matrix
                WhetStone::DenseMatrix Sigma; // pressure mass matrix
                WhetStone::DenseMatrix E; // concentration "interpolation" matrix
                WhetStone::DenseMatrix C; // - concentration mass matrix
                WhetStone::DenseMatrix R; // interpolation matrix
            };
            struct BackSubstLocalMatrices { // matrices needed to recover fluxes / pressure from pressure traces 
                WhetStone::DenseMatrix BWBt_plus_cSigma_inv;
                WhetStone::DenseMatrix BW;
                WhetStone::DenseMatrix EC;
                WhetStone::DenseMatrix W;
                WhetStone::DenseMatrix R;
            };
            PDE_DiffusionMFD_ASC(Teuchos::ParameterList& plist, Teuchos::RCP<const AmanziMesh::MeshMini> const & mesh)
                : PDE_Abstract(plist, mesh->macroMesh())
                , meshMini_(mesh)
                , plist_(plist)
                , MFD_(mesh->macroMesh()) 
                , backSubstLocalMatrices_(ncells_owned) 
                , K_(ncells_owned)
                , f_(ncells_owned) {
                operator_type_ = OPERATOR_DIFFUSION_MFD_XMOF;
                MFD_.ModifyStabilityScalingFactor(1.);
            }
            PDE_DiffusionMFD_ASC& assembleLocalConsentrationSystems();
            PDE_DiffusionMFD_ASC& computeExactConcentrations(Epetra_MultiVector&, ScalarFunc const &, double);
            PDE_DiffusionMFD_ASC& computeExactCellVals(Epetra_MultiVector&, ScalarFunc const &, double);
            PDE_DiffusionMFD_ASC& recoverSolution(CompositeVector&, CompositeVector&);
            PDE_DiffusionMFD_ASC& setDiffusion(TensorFuncInd const &);
            PDE_DiffusionMFD_ASC& setRHS(ScalarFunc const &, double);
            PDE_DiffusionMFD_ASC& setReaction(double);
            PDE_DiffusionMFD_ASC& setBC(BC const &, double);
        private:
            Teuchos::ParameterList plist_;
            Teuchos::RCP<const AmanziMesh::MeshMini> meshMini_;
            std::vector<BackSubstLocalMatrices> backSubstLocalMatrices_;
            std::vector<WhetStone::DenseMatrix>& S_ = local_op_->matrices; // local concentration matrices
            std::vector<std::vector<WhetStone::Tensor>> K_; // diffusion tensor
            std::vector<std::vector<double>> f_; // rhs
            double c_ = 0.; // reaction / accum coef
            WhetStone::MFD3D_Diffusion MFD_;
            WhetStone::DenseVector getLocalRHS_(size_t) const;
            WhetStone::DenseVector getLocalConcentrations_(size_t, Epetra_MultiVector const &) const;
            LocalSystem assembleLocalSystem_(size_t);
            BackSubstLocalMatrices computeBackSubstLocalMatrices_(LocalSystem const &);
            double getMoment_(size_t, size_t, ScalarFunc const &, double) const;
            bool faceIsFlat_(size_t) const;
            bool faceIsBndry_(size_t) const;
            bool massMatrixIsExact_(WhetStone::DenseMatrix const &, size_t c, double*) const;
        };
    };
};


#endif //OPERATORS_PDE_DiffusionMFD_ASC_H
