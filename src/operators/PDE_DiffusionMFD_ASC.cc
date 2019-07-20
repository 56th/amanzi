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
            if (m != 0) throw std::invalid_argument("getMoment: not implemented for m > 0");
            return f(mesh_->face_centroid(faceIndex), t);
            // auto const n  = mesh_->face_normal(faceIndex);
            // auto const c  = mesh_->face_centroid(faceIndex);
            // if (fpEqual(n[2], 0.)) return f(c, t);
            // auto const nc = n * c;
            // auto z = [&](double x, double y) {
            //     return (nc - n[0] * x - n[1] * y) / n[2];
            // };
            // auto zX = -n[0] / n[2];
            // auto zY = -n[1] / n[2];
            // std::vector<Node> coords;
            // mesh_->face_get_coordinates(faceIndex, &coords);
            // for (auto& coord : coords) coord[2] = 0.;
            // double area;
            // Node normal(3), centroid(3);
            // AmanziGeometry::polygon_get_area_centroid_normal(coords, &area, &centroid, &normal);
            // return sqrt(1. + zX * zX + zY * zY) * area * f(Node(centroid[0], centroid[1], z(centroid[0], centroid[1])), t) / mesh_->face_area(faceIndex);
        }
        size_t PDE_DiffusionMFD_ASC::numbOfMaterials_(size_t c) const {
            return 1;
        }
        AmanziMesh::Entity_ID_List PDE_DiffusionMFD_ASC::getMacroFacesIndicies_(size_t c) const {
            AmanziMesh::Entity_ID_List macroFacesIndicies;
            mesh_->cell_get_faces(c, &macroFacesIndicies);
            return macroFacesIndicies;
        }
        std::vector<int> PDE_DiffusionMFD_ASC::getMacroFacesNormalDirs_(size_t c) const {
            AmanziMesh::Entity_ID_List macroFacesIndicies;
            std::vector<int> macroFacesNormalsDirs;
            mesh_->cell_get_faces_and_dirs(c, &macroFacesIndicies, &macroFacesNormalsDirs);
            return macroFacesNormalsDirs;
        }
        WhetStone::DenseVector PDE_DiffusionMFD_ASC::getLocalRHS_(size_t c) const {
            return WhetStone::DenseVector(1, const_cast<double *>(f_[c].data()));
        }
        WhetStone::DenseVector PDE_DiffusionMFD_ASC::getLocalConcentrations_(size_t c, Epetra_MultiVector const & lambda) const {
            auto macroFacesIndicies = getMacroFacesIndicies_(c);
            auto n = macroFacesIndicies.size();
            WhetStone::DenseVector lambdaCoarse(n);
            for (size_t i = 0; i < n; ++i) 
                lambdaCoarse(i) = lambda[0][macroFacesIndicies[i]];
            return backSubstLocalMatrices_[c].R * lambdaCoarse;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::setDiffusion(TensorFunc const & K, double t = 0.) {
            auto& logger = SingletonLogger::instance();
            logger.beg("set diffusion");
                for (size_t c = 0; c < ncells_owned; ++c) {
                    logger.pro(c + 1, ncells_owned);
                    K_[c].resize(1);
                    K_[c][0] = K(mesh_->cell_centroid(c), t);
                }
            logger.end();
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::setRHS(ScalarFunc const & f, double t = 0.) {
            auto& logger = SingletonLogger::instance();
            logger.beg("set rhs");
                for (size_t c = 0; c < ncells_owned; ++c) {
                    logger.pro(c + 1, ncells_owned);
                    f_[c].resize(1);
                    f_[c][0] = f(mesh_->cell_centroid(c), t) * mesh_->cell_volume(c);
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
                logger.buf << "numb of bndry faces:      " << n << '\n'
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
                    // logger.buf 
                    //     << "inverse mass mtx, W:\n"                       << localSystem.W << '\n'
                    //     << "divergence mtx, -B:\n"                        << localSystem.mB << '\n'
                    //     << "pressure mass mtx, \\Sigma:\n"                << localSystem.Sigma << '\n'
                    //     << "concentration \"interpolation\" matrix, E:\n" << localSystem.E << '\n'
                    //     << "- concentration mass matrix, C:\n"            << localSystem.C << '\n'
                    //     << "interpolation mtx, R:\n"                      << localSystem.R << '\n';
                    // logger.log();
                    // logger.buf 
                    //     << "EC:\n"                     << backSubstLocalMatrices_[c].EC << '\n'
                    //     << "BW:\n"                     << backSubstLocalMatrices_[c].BW << '\n'
                    //     << "(BWB^T + \\Sigma)^{-1}:\n" << backSubstLocalMatrices_[c].BWBt_plus_cSigma_inv << '\n';
                    // logger.log();
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
                    auto macroFacesIndicies = getMacroFacesIndicies_(c);
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
            auto macroFacesIndicies = getMacroFacesIndicies_(c);
            auto macroFacesNormalsDirs = getMacroFacesNormalDirs_(c);
            auto numbOfMacroFaces = macroFacesIndicies.size();
            // mass matrix
            // res.W.Reshape(numbOfMacroFaces, numbOfMacroFaces);
            // auto W = res.W;
            // if (MFD_.MassMatrixInverse(c, K_[c][0], W) == WhetStone::WHETSTONE_ELEMENTAL_MATRIX_FAILED) 
            //     throw std::logic_error("MFD: unexpected failure in WhetStone for mass matrix");
            std::vector<Node> fc(numbOfMacroFaces), fn(numbOfMacroFaces);
            std::vector<double> fa(numbOfMacroFaces);
            for (size_t i = 0; i < numbOfMacroFaces; ++i) {
                auto f = macroFacesIndicies[i];
                fc[i] = mesh_->face_centroid(f);
                fn[i] = macroFacesNormalsDirs[i] * mesh_->face_normal(f);
                fn[i] /= AmanziGeometry::norm(fn[i]);
                fa[i] = mesh_->face_area(f);
            }
            if (MFD_.MassMatrixInverse(mesh_->cell_centroid(c), mesh_->cell_volume(c), fc, fn, fa, K_[c][0], res.W, false /* no rescaling! */) == WhetStone::WHETSTONE_ELEMENTAL_MATRIX_FAILED)
                throw std::logic_error("MFD: unexpected failure in WhetStone for mass matrix");
            double diff;
            if (!massMatrixIsExact_(res.W, c, &diff))
                logger.wrn("cell #" + std::to_string(c) + ": mass matrix is not exact for const fields, diff = " + std::to_string(diff));
            // div matrix
            res.mB.Reshape(1, numbOfMacroFaces);
            for (size_t i = 0; i < numbOfMacroFaces; ++i) 
                res.mB(0, i) = mesh_->face_area(macroFacesIndicies[i]);
            // pressure mass matrix
            res.Sigma.Reshape(1, 1);
            res.Sigma(0, 0) = mesh_->cell_volume(c);
            // concentration "interpolation" matrix
            res.E.Reshape(numbOfMacroFaces, numbOfMacroFaces);
            res.E.PutScalar(0.);
            for (size_t i = 0; i < numbOfMacroFaces; ++i)
                res.E(i, i) = 1.;
            // - concentration mass matrix
            res.C.Reshape(numbOfMacroFaces, numbOfMacroFaces);
            res.C.PutScalar(0.);
            for (size_t i = 0; i < numbOfMacroFaces; ++i) 
                res.C(i, i) = -mesh_->face_area(macroFacesIndicies[i]);
            // interpolation matrix
            res.R.Reshape(numbOfMacroFaces, numbOfMacroFaces);
            res.R.PutScalar(0.);
            for (size_t i = 0; i < numbOfMacroFaces; ++i)
                res.R(i, i) = 1.;
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
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::recoverSolution(CompositeVector& P, CompositeVector& U) {
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
                // recover pressure cell vals
                auto pLocal = BWBt_plus_cSigma_inv * (getLocalRHS_(c) + BW * EC * lambdaFine);
                // recover flux vals
                auto uLocal = W * EC * lambdaFine - BW.t() * pLocal;

                // AmanziMesh::Entity_ID_List macroFacesIndicies;
                // std::vector<int> macroFacesNormalsDirs;
                // mesh_->cell_get_faces_and_dirs(c, &macroFacesIndicies, &macroFacesNormalsDirs);
                // auto div = 0.;
                // for (size_t i = 0; i < macroFacesIndicies.size(); ++i)
                //     div += macroFacesNormalsDirs[i] * uLocal(i) * mesh_->face_area(macroFacesIndicies[i]);
                // logger.buf << c << ": div = " << div;
                // logger.log();

                // global pressure cell vals
                for (size_t i = 0; i < numbOfMaterials_(c); ++i)
                    p[i][c] = pLocal(i);   
                // global fluxes     
                auto macroFacesIndicies = getMacroFacesIndicies_(c);
                for (size_t i = 0; i < macroFacesIndicies.size(); ++i) {
                    auto f = macroFacesIndicies[i];
                    if (vis[f]) l2 += pow(u[0][f] + uLocal(i), 2.) * mesh_->face_area(f);
                    else {
                        u[0][f] = uLocal(i);
                        vis[f] = true;
                    }
                }
            }
            l2 = sqrt(l2);
            logger.buf << "recovered flux residual: " << l2;
            logger.log();
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::computeExactConcentrations(Epetra_MultiVector& res, ScalarFunc const & p, double t = 0.) {
            for (size_t f = 0; f < nfaces_owned; ++f)
                res[0][f] = getMoment_(0, f, p, t);
            return *this;
        }
        PDE_DiffusionMFD_ASC& PDE_DiffusionMFD_ASC::computeExactCellVals(Epetra_MultiVector& res, ScalarFunc const & p, double t = 0.) {
            for (size_t c = 0; c < ncells_owned; ++c) 
                res[0][c] = p(mesh_->cell_centroid(c), t);
            return *this;
        }
        bool PDE_DiffusionMFD_ASC::massMatrixIsExact_(WhetStone::DenseMatrix const & W, size_t c, double* diff) const {
            auto M = W;
            M.Inverse();
            auto macroFacesIndicies = getMacroFacesIndicies_(c);
            auto macroFacesNormalsDirs = getMacroFacesNormalDirs_(c);
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
