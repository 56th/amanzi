#include "PDE_DiffusionMFD_ASC.hh"
#include "Geometry.hh"
#include "SingletonLogger.hpp"

namespace Amanzi {
    namespace Operators {
        bool PDE_DiffusionMFD_ASC::faceIsFlat_(size_t faceIndex) const {
            auto n = mesh_->face_normal(faceIndex);
            n /= AmanziGeometry::norm(n);
            auto const c  = mesh_->face_centroid(faceIndex);
            auto isInPlane = [&](Node const & x) {
                return fpEqual(n * (c - x), 0.);
            };
            std::vector<Node> coords;
            mesh_->face_get_coordinates(faceIndex, &coords);
            for (auto const & coord : coords) 
                if (!isInPlane(coord)) return false;
            return true;
        }
        bool PDE_DiffusionMFD_ASC::faceIsBndry_(size_t faceIndex) const {
            AmanziMesh::Entity_ID_List cellIndicies;
            mesh_->face_get_cells(faceIndex, AmanziMesh::Parallel_type::ALL, &cellIndicies);
            return cellIndicies.size() == 1;
        }
        double PDE_DiffusionMFD_ASC::getMoment_(size_t m, size_t faceIndex, ScalarFunc const & f, double t = 0.) const {
            auto& logger = SingletonLogger::instance();
            std::string err = __func__;
            err += ": ";
            if (m != 0) 
                throw std::invalid_argument(err + "not implemented for m > 0");
            AmanziMesh::Entity_ID_List cellIndicies;
            mesh_->face_get_cells(faceIndex, AmanziMesh::Parallel_type::ALL, &cellIndicies);
            auto c = cellIndicies.front();
            auto faceIndicies = meshMini_->macroFacesIndicies(c);
            int faceLocalIndex = -1;
            for (size_t i = 0; i < faceIndicies.size(); ++i)
                if (faceIndicies[i] == faceIndex) {
                    faceLocalIndex = i;
                    break;
                }
            if (faceLocalIndex == -1)
                throw std::invalid_argument(err + "cannot find loc index of the face in its adj cell");
            auto res = 0.;
            for (auto i : meshMini_->childrenFacesGlobalIndicies(c, faceLocalIndex)) 
                res += f(meshMini_->faceCentroid(c, i), t) * meshMini_->area(c, i);
            res /= mesh_->face_area(faceIndex);
            auto diff = res - f(mesh_->face_centroid(faceIndex), 0.);
            // if (!fpEqual(diff, 0.)) {
            //     logger.buf << "macro face #" << faceIndex << " moments diff = " << diff;
            //     logger.log();
            // }
            return res;
        }
        WhetStone::DenseVector PDE_DiffusionMFD_ASC::getLocalRHS_(size_t c) const {
            return WhetStone::DenseVector(f_[c].size(), const_cast<double *>(f_[c].data()));
        }
        WhetStone::DenseVector PDE_DiffusionMFD_ASC::getLocalConcentrations_(size_t c, Epetra_MultiVector const & lambda) const {
            auto macroFacesIndicies = meshMini_->macroFacesIndicies(c);
            auto n = macroFacesIndicies.size();
            WhetStone::DenseVector lambdaCoarse(n);
            for (size_t i = 0; i < n; ++i) 
                lambdaCoarse(i) = lambda[0][macroFacesIndicies[i]];
            return backSubstLocalMatrices_[c].R * lambdaCoarse;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::setDiffusion(TensorFuncInd const & K) {
            auto& logger = SingletonLogger::instance();
            logger.beg("set diffusion");
                for (size_t c = 0; c < ncells_owned; ++c) {
                    logger.pro(c + 1, ncells_owned);
                    auto n = meshMini_->numbOfMaterials(c);
                    K_[c].resize(n);
                    for (size_t i = 0; i < n; ++i)
                        K_[c][i] = K(meshMini_->materialIndex(c, i));
                }
            logger.end();
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::setRHS(ScalarFunc const & f, double t = 0.) {
            auto& logger = SingletonLogger::instance();
            logger.beg("set rhs");
                for (size_t c = 0; c < ncells_owned; ++c) {
                    logger.pro(c + 1, ncells_owned);
                    auto n = meshMini_->numbOfMaterials(c);
                    f_[c].resize(n);
                    for (size_t i = 0; i < n; ++i) 
                        f_[c][i] = f(meshMini_->centroid(c, i), t) * meshMini_->volume(c, i);
                }
            logger.end();
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::setReaction(double c) {
            c_ = c;
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::setBC(PDE_DiffusionMFD_ASC::BC const & bc, double t = 0.) {
            if (bc.type == BCType::Neumann) throw std::invalid_argument("setBC: Neumann bc is not yet implemented");
            auto BC = Teuchos::rcp(new BCs(mesh_, AmanziMesh::FACE, WhetStone::DOF_Type::SCALAR));
            auto& logger = SingletonLogger::instance();
            size_t n = 0, m = 0;
            double area = 0., mean = 0.;
            logger.beg("set BC");
                for (size_t f = 0; f < nfaces_owned; ++f) {
                    logger.pro(f + 1, nfaces_owned);
                    m += !faceIsFlat_(f);
                    if (faceIsBndry_(f) && bc.p(mesh_->face_centroid(f))) {
                        BC->bc_value()[f] = getMoment_(0, f, bc.f, t);
                        BC->bc_model()[f] = OPERATOR_BC_DIRICHLET;
                        area += mesh_->face_area(f);
                        mean += mesh_->face_area(f) * BC->bc_value()[f];
                        ++n;
                    }
                }
                mean /= area;
                logger.buf 
                        << "numb of bndry faces:      " << n << '\n'
                        << "bndry area:               " << area << '\n'
                        << "concentration bndry mean: " << mean << '\n'
                        << "numb of curved faces:     " << m;
                logger.log();
                SetBCs(BC, BC);
            logger.end();
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::assembleLocalConsentrationSystems() {
            auto& logger = SingletonLogger::instance();
            auto& sGlobal = *global_op_->rhs()->ViewComponent("face", true);
            global_op_->rhs()->PutScalarGhosted(0.);
            logger.beg("assemble local concentration systems and global rhs");
                for (size_t c = 0; c < ncells_owned; ++c) {
                    logger.pro(c + 1, ncells_owned);
                    auto localSystem = assembleLocalSystem_(c);
                    backSubstLocalMatrices_[c] = computeBackSubstLocalMatrices_(localSystem);
                    // if (meshMini_->numbOfMaterials(c) > 1) {
                    //     logger.buf 
                    //         << "inverse mass mtx, W:\n"                       << localSystem.W << '\n'
                    //         << "divergence mtx, -B:\n"                        << localSystem.mB << '\n'
                    //         << "pressure mass mtx, \\Sigma:\n"                << localSystem.Sigma << '\n'
                    //         << "concentration \"interpolation\" matrix, E:\n" << localSystem.E << '\n'
                    //         << "- concentration mass matrix, C:\n"            << localSystem.C << '\n'
                    //         << "interpolation mtx, R:\n"                      << localSystem.R << '\n';
                    //     logger.log();
                    //     logger.buf 
                    //         << "EC:\n"                     << backSubstLocalMatrices_[c].EC << '\n'
                    //         << "BW:\n"                     << backSubstLocalMatrices_[c].BW << '\n'
                    //         << "(BWB^T + \\Sigma)^{-1}:\n" << backSubstLocalMatrices_[c].BWBt_plus_cSigma_inv << '\n';
                    //     logger.log();
                    // }
                    auto& W = backSubstLocalMatrices_[c].W;
                    auto& BW = backSubstLocalMatrices_[c].BW;
                    auto& BWBt_plus_cSigma_inv = backSubstLocalMatrices_[c].BWBt_plus_cSigma_inv;
                    auto& EC = backSubstLocalMatrices_[c].EC;
                    auto& R = backSubstLocalMatrices_[c].R;
                    // local condensation mtx
                    S_[c] = R.t() * EC.t() * (W - BW.t() * BWBt_plus_cSigma_inv * BW) * EC * R;
                    // local condensation vector
                    auto s = (R.t() * EC.t() * BW.t() * BWBt_plus_cSigma_inv) * getLocalRHS_(c);
                    // apply BC
                    auto macroFacesIndicies = meshMini_->macroFacesIndicies(c);
                    auto n = macroFacesIndicies.size();
                    // zero out rows
                    auto const & bcModelTest = bcs_test_[0]->bc_model();
                    for (size_t i = 0; i < n; ++i)
                        if (bcModelTest[macroFacesIndicies[i]] == OPERATOR_BC_DIRICHLET)
                            for (size_t j = 0; j < n; ++j)
                                S_[c](i, j) = 0.;
                    // zero out cols
                    auto const & bcModelTrial = bcs_trial_[0]->bc_model();
                    auto const & bcValueTrial = bcs_trial_[0]->bc_value();
                    for (size_t j = 0; j < n; ++j) {
                        auto jGlobal = macroFacesIndicies[j];
                        if (bcModelTrial[jGlobal] == OPERATOR_BC_DIRICHLET) {
                            auto val = bcValueTrial[jGlobal];
                            for (size_t i = 0; i < n; ++i) {
                                s(i) -= S_[c](i, j) * val;
                                S_[c](i, j) = 0.;

                            }
                            S_[c](j, j) = 1.;
                            s(j) = val;
                        }
                        // logger.buf << macroFacesIndicies[j] << ' ';
                    }
                    // logger.buf << "\nS:\n" << S_[c] << '\n' << "s:\n" << s;
                    // logger.log();
                    // assemble global rhs
                    for (size_t i = 0; i < n; ++i)
                        sGlobal[0][macroFacesIndicies[i]] += s(i);
                }
            logger.end();
            // logger.buf << "global s:\n" << sGlobal;
            // logger.log();
            return *this;
        }
        PDE_DiffusionMFD_ASC::LocalSystem PDE_DiffusionMFD_ASC::assembleLocalSystem_(size_t c) {
            auto& logger = SingletonLogger::instance();
            LocalSystem res;
            auto numbOfMacroFaces = meshMini_->macroFacesIndicies(c).size();
            auto numbOfMiniCells = meshMini_->numbOfMaterials(c);
            auto numbOfMiniFaces = meshMini_->numbOfFaces(c);
            auto numbOfExtMiniFaces = meshMini_->numbOfExtFaces(c);
            // interpolation matrix
            res.R.Reshape(numbOfExtMiniFaces, numbOfMacroFaces);
            res.R.PutScalar(0.);
            for (size_t i = 0; i < numbOfExtMiniFaces; ++i)
                res.R(i, meshMini_->parentFaceLocalIndex(c, i)) = 1.;
            // pressure mass matrix
            res.Sigma.Reshape(numbOfMiniCells, numbOfMiniCells);
            res.Sigma.PutScalar(0.);
            // - concentration mass matrix
            res.C.Reshape(numbOfExtMiniFaces, numbOfExtMiniFaces);
            res.C.PutScalar(0.);
            for (size_t i = 0; i < numbOfExtMiniFaces; ++i) 
                res.C(i, i) = -meshMini_->area(c, i);
            // concentration "interpolation" matrix
            res.E.Reshape(numbOfMiniFaces, numbOfExtMiniFaces);
            res.E.PutScalar(0.);
            for (size_t i = 0; i < numbOfExtMiniFaces; ++i)
                res.E(i, i) = 1.;
            // inverse mass matrix
            res.W.Reshape(numbOfMiniFaces, numbOfMiniFaces);
            res.W.PutScalar(0.);
            // div matrix
            res.mB.Reshape(numbOfMiniCells, numbOfMiniFaces);
            res.mB.PutScalar(0.);
            // assemble
            for (size_t m = 0; m < numbOfMiniCells; ++m) {
                auto ind = meshMini_->facesGlobalIndicies(c, m);
                auto n = ind.size();
                // pressure mass matrix
                res.Sigma(m, m) = meshMini_->volume(c, m);
                // div matrix
                for (auto j : ind)
                    res.mB(m, j) += meshMini_->normalSign(c, m, j) * meshMini_->area(c, j);
                // inverse mass matrix
                WhetStone::DenseMatrix locW;
                std::vector<Node> fc(n), fn(n);
                std::vector<double> fa(n);
                for (size_t i = 0; i < n; ++i) {
                    auto f = ind[i];
                    fc[i] = meshMini_->faceCentroid(c, f);
                    fn[i] = meshMini_->normal(c, f);
                    fn[i] /= AmanziGeometry::norm(fn[i]);
                    fa[i] = meshMini_->area(c, f);
                }
                if (MFD_.MassMatrixInverse(meshMini_->centroid(c, m), meshMini_->volume(c, m), fc, fn, fa, K_[c][m], locW, false /* no rescaling! */) == WhetStone::WHETSTONE_ELEMENTAL_MATRIX_FAILED)
                    throw std::logic_error("MFD: unexpected failure in WhetStone for mass matrix");
                for (size_t i = 0; i < n; ++i)
                    for (size_t j = 0; j < n; ++j)
                        res.W(ind[i], ind[j]) += locW(i, j);
            }
            double diff;
            if (numbOfMiniCells == 1 && !massMatrixIsExact_(res.W, c, &diff))
                logger.wrn("cell #" + std::to_string(c) + ": mass matrix is not exact for const fields, diff = " + std::to_string(diff));
            return res;
        }
        PDE_DiffusionMFD_ASC::BackSubstLocalMatrices PDE_DiffusionMFD_ASC::computeBackSubstLocalMatrices_(LocalSystem const & localSystem) {
            BackSubstLocalMatrices res;
            // EC
            res.EC = localSystem.E * localSystem.C;
            // BW = BM^{-1}
            res.BW = -localSystem.mB * localSystem.W;
            // (BWB^T + \Sigma)^{-1}
            res.BWBt_plus_cSigma_inv = -res.BW * localSystem.mB.t() + localSystem.Sigma * c_;
            res.BWBt_plus_cSigma_inv.Inverse();
            // W
            res.W = localSystem.W;
            // R
            res.R = localSystem.R;
            return res;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::recoverSolution(CompositeVector& P, CompositeVector& U, ScalarFunc const * gD = nullptr, double* fluxRes = nullptr) {
            auto& logger = SingletonLogger::instance();
            auto& p      = *P.ViewComponent("cell", true);
            auto& lambda = *P.ViewComponent("face", true); 
            auto& u      = *U.ViewComponent("face", true);
            std::vector<bool> vis(nfaces_owned, false);
            auto l2 = 0.;
            for (size_t c = 0; c < ncells_owned; ++c) {
                logger.pro(c + 1, ncells_owned);
                auto& W = backSubstLocalMatrices_[c].W;
                auto& BW = backSubstLocalMatrices_[c].BW;
                auto& BWBt_plus_cSigma_inv = backSubstLocalMatrices_[c].BWBt_plus_cSigma_inv;
                auto& EC = backSubstLocalMatrices_[c].EC;
                auto lambdaFine = getLocalConcentrations_(c, lambda);
                auto macroFacesIndicies = meshMini_->macroFacesIndicies(c);
                // use BCs to correct vals of lambdaFine
                if (gD != nullptr) {
                    auto const & bcModelTrial = bcs_trial_[0]->bc_model();
                    for (size_t i = 0; i < meshMini_->numbOfExtFaces(c); ++i) {
                        auto f = meshMini_->parentFaceLocalIndex(c, i);
                        auto F = macroFacesIndicies[f];
                        if (bcModelTrial[F] == OPERATOR_BC_DIRICHLET) {
                            // auto diff = lambdaFine(i) - (*gD)(meshMini_->faceCentroid(c, i), 0.);
                            // if (!fpEqual(diff, 0.)) {
                            //     logger.buf << "lambda fine exact (from BCs) / lambda fine recovered diff = " << diff;
                            //     logger.log();
                            // }
                            lambdaFine(i) = (*gD)(meshMini_->faceCentroid(c, i), 0.);
                        }
                    }
                }
                // recover pressure cell vals
                auto pLocal = BWBt_plus_cSigma_inv * (getLocalRHS_(c) + BW * EC * lambdaFine);
                // recover flux vals
                auto uLocal = W * EC * lambdaFine - BW.t() * pLocal;
                // global pressure cell vals
                for (size_t i = 0; i < pLocal.NumRows(); ++i)
                    p[i][c] = pLocal(i);   
                // global fluxes     
                for (size_t i = 0; i < macroFacesIndicies.size(); ++i) {
                    auto f = macroFacesIndicies[i];
                    auto uF = 0.;
                    for (auto j : meshMini_->childrenFacesGlobalIndicies(c, i)) 
                        uF += uLocal(j) * meshMini_->area(c, j);
                    uF /= mesh_->face_area(f);
                    if (vis[f]) l2 += pow(u[0][f] + uF, 2.) * mesh_->face_area(f);
                    else {
                        u[0][f] = uF;
                        vis[f] = true;
                    }
                }
            }
            if (fluxRes != nullptr) *fluxRes = sqrt(l2);
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::computeExactConcentrations(Epetra_MultiVector& res, ScalarFunc const & p, double t = 0.) {
            for (size_t f = 0; f < nfaces_owned; ++f)
                res[0][f] = getMoment_(0, f, p, t);
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::computeExactCellVals(Epetra_MultiVector& res, ScalarFunc const & p, double t = 0.) {
            for (size_t C = 0; C < ncells_owned; ++C) 
                for (size_t c = 0; c < meshMini_->numbOfMaterials(C); ++c)
                    res[c][C] = p(meshMini_->centroid(C, c), t);
            return *this;
        }
        bool PDE_DiffusionMFD_ASC::massMatrixIsExact_(WhetStone::DenseMatrix const & W, size_t c, double* diff) const {
            auto M = W;
            M.Inverse();
            auto macroFacesIndicies = meshMini_->macroFacesIndicies(c);
            auto macroFacesNormalsDirs = meshMini_->macroFacesNormalsDirs(c);
            auto numbOfMacroFaces = macroFacesIndicies.size();
            Node u(1., 2., 3.), v(4., 5., 6.);
            WhetStone::DenseVector uI(numbOfMacroFaces), vI(numbOfMacroFaces);
            for (size_t i = 0; i < numbOfMacroFaces; ++i) {
                auto f = macroFacesIndicies[i];
                auto n = macroFacesNormalsDirs[i] * mesh_->face_normal(f);
                n /= AmanziGeometry::norm(n);
                uI(i) = u * n;
                vI(i) = v * n;
            }
            auto KInv = 1. / K_[c][0](0, 0);
            *diff = (M * uI) * vI - (KInv * u) * v * mesh_->cell_volume(c);
            return fpEqual(*diff, 0., 1e-5);
        }
    };
};