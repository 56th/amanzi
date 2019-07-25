/*
  Author: Alexander Zhiliakov (alex@math.uh.edu)
*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>

// TPLs
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"
#include "UnitTest++.h"

// Amanzi
#include "MeshFactory.hh"
#include "GMVMesh.hh"
#include "LinearOperatorFactory.hh"
// #include "mfd3d_diffusion.hh"
// #include "tensor.hh"

// Operators
#include "OperatorDefs.hh"

// fancy colors for cout logging 
#include "SingletonLogger.hpp"

// mesh wrapper to work with tangram
#include "wonton/wonton/mesh/amanzi/amanzi_mesh_wrapper.h"

#include "PDE_DiffusionMFD_ASC.hh"
#include "DiffusionReactionEqn.hh"
#include "OutputXDMF.hh"

Tensor constTensor(double c) {
    Amanzi::WhetStone::Tensor K(3, 1);
    K.PutScalar(c);
    return K;
}

TEST(OPERATOR_DIFFUSION_ASC) {
    using namespace Teuchos;
    using namespace Amanzi;
    using namespace Amanzi::AmanziMesh;
    using namespace Amanzi::AmanziGeometry;
    using namespace Amanzi::Operators;
    auto& logger = SingletonLogger::instance();
    try {
        auto comm = Amanzi::getDefaultComm();
        auto MyPID = comm->MyPID();
        std::vector<std::string> meshNames;
        {
            using namespace boost::filesystem;
            path p("test/meshes");
            for (auto i = directory_iterator(p); i != directory_iterator(); ++i)
                if (!is_directory(i->path()))
                    meshNames.emplace_back(i->path().filename().string());
        }
        auto meshIndex = logger.opt("choose mesh", meshNames);
        auto meshName = meshNames[meshIndex];
        logger.beg("load mesh");
            std::string xmlFileName = "test/operator_diffusion_asc.xml";
            logger.log(xmlFileName);
            ParameterXMLFileReader xmlreader(xmlFileName);
            auto plist = xmlreader.getParameters();
            plist.get<Teuchos::ParameterList>("io").set<std::string>("file name base", plist.get<Teuchos::ParameterList>("io").get<std::string>("file name base") + '/' +  meshName);
            auto region_list = plist.get<Teuchos::ParameterList>("regions");
            Teuchos::RCP<GeometricModel> gm = Teuchos::rcp(new GeometricModel(3, region_list, *comm));
            MeshFactory meshfactory(comm, gm);
            meshfactory.set_preference(Preference({Framework::MSTK, Framework::STK}));
            Teuchos::RCP<const Mesh> mesh = meshfactory.create("test/meshes/" + meshName);
            auto numbOfCells = mesh->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::ALL);
            auto numbOfFaces = mesh->num_entities(AmanziMesh::FACE, AmanziMesh::Parallel_type::ALL);
            logger.buf << "numb of cells: " << numbOfCells << '\n'
                       << "numb of faces: " << numbOfFaces;
            logger.log();
            logger.beg("tangram");
                Wonton::Amanzi_Mesh_Wrapper meshWrapper(*mesh);
                logger.buf << "numb of cells: " << meshWrapper.num_owned_cells() << '\n'
                           << "numb of faces: " << meshWrapper.num_owned_faces();
                logger.log();
            logger.end();
        logger.end();
        logger.beg("set exact soln");
            DiffusionReactionEqn eqn;
            auto solnIndex = logger.opt("choose soln", { "p = 1", "p = x + 2y + 3z + 4", "x" });
            if (solnIndex == 0) {
                eqn.p = [](Node const &, double) { return 1.; };
                eqn.pGrad = [](Node const &, double) { return Node(0., 0., 0.); };
                auto const pHess = constTensor(0.);
                eqn.pHess = [=](Node const &, double) { return pHess; };
            } else if (solnIndex == 1) {
                eqn.p = [](Node const & x, double) { return x[0] + 2. * x[1] + 3. * x[2] + 4.; };
                eqn.pGrad = [](Node const &, double) { return Node(1., 2., 3.); };
                auto const pHess = constTensor(0.);
                eqn.pHess = [=](Node const &, double) { return pHess; };
            }
            else {
                eqn.p = [](Node const & x, double) { return x[0]; };
                eqn.pGrad = [](Node const &, double) { return Node(1., 0., 0.); };
                auto const pHess = constTensor(0.);
                eqn.pHess = [=](Node const &, double) { return pHess; };
            }
            double k;
            logger.inp("set diffusion coef", k);
            auto const K = constTensor(k);
            eqn.K = [=](Node const &, double) { return K; };
            logger.inp("set reaction coef", eqn.c);
            PDE_DiffusionMFD_ASC::BC bc;
            bc.type = PDE_DiffusionMFD_ASC::BCType::Dirichlet;
            bc.p = [=](Node const & x) { return true; };
            bc.f = [&](Node const & x, double t) { return eqn.p(x, t); };
        logger.end();
        logger.beg("set up operator");
            auto olist = plist.sublist("PK operator").sublist("diffusion operator asc");
            PDE_DiffusionMFD_ASC op(olist, mesh);
            op.setDiffusion(eqn.K, 0.).setReaction(eqn.c).setRHS(eqn.f(), 0.).setBC(bc, 0.).assembleLocalConsentrationSystems();
            CompositeVectorSpace cvsP, cvsU;
            cvsP.SetMesh(mesh)->SetGhosted(true)->AddComponent("cell", AmanziMesh::CELL, 1);
            cvsP.AddComponent("face", AmanziMesh::FACE, 1);
            cvsU.SetMesh(mesh)->SetGhosted(true)->AddComponent("face", AmanziMesh::FACE, 1);
            auto  p     = CompositeVector(cvsP);
            auto& pCell = *p.ViewComponent("cell", true);
            auto& pFace = *p.ViewComponent("face", true);
            auto  u     = CompositeVector(cvsU);
        logger.end();
        logger.beg("compute exact soln d.o.f.");
            auto pFaceExact = pFace;
            auto pCellExact = pCell;
            op.computeExactConcentrations(pFaceExact, eqn.p, 0.); 
            op.computeExactCellVals(pCellExact, eqn.p, 0.); 
        logger.end();
        auto solveIndex = logger.opt("get the solution", { "linear solve", "recover from exact concentrations" });
        if (solveIndex == 0) {
            logger.beg("assemble global system");
                auto opGlobal = op.global_operator();
                opGlobal->SymbolicAssembleMatrix();
                opGlobal->AssembleMatrix();
                // opGlobal->ExportMatlab("mtx.txt");
                auto& rhs = *opGlobal->rhs();
            logger.end();
            logger.beg("linear solve");
                // auto artSolution = p;
                // *artSolution.ViewComponent("face", true) = pFaceExact;
                // auto artRhs = rhs;
                // artRhs.PutScalar(0.);
                // opGlobal->ApplyAssembled(artSolution, artRhs, 0.);
                // logger.buf << "art rhs:\n";
                // artRhs.Print(logger.buf);
                // logger.buf << "comp rhs:\n";
                // rhs.Print(logger.buf);
                // auto diffRhs = rhs;
                // diffRhs.Update(1., artRhs, -1.);
                // // diffRhs.Print(logger.buf << "rhs diff:\n");
                // double rhsDiffNorm = 0.;
                // diffRhs.Norm2(&rhsDiffNorm);
                // logger.buf << "comp rhs and art rhs diff norm: " << rhsDiffNorm;
                // logger.log();
                auto slist = plist.get<Teuchos::ParameterList>("preconditioners");
                opGlobal->InitPreconditioner("Hypre AMG", slist);
                auto lop_list = plist.sublist("solvers").sublist("AztecOO CG").sublist("pcg parameters");
                AmanziSolvers::LinearOperatorPCG<Operator, CompositeVector, CompositeVectorSpace> solver(opGlobal, opGlobal);
                solver.Init(lop_list);
                auto ierr = solver.ApplyInverse(rhs, p);
                double l2 = 0.;
                for (size_t f = 0; f < numbOfFaces; ++f)
                    l2 += pow(pFaceExact[0][f] - pFace[0][f], 2.) * mesh->face_area(f);
                l2 = sqrt(l2);
                logger.buf 
                    << "residual norm: " << solver.residual() << '\n'
                    << "iters:         " << solver.num_itrs() << '\n'
                    << "code:          " << solver.returned_code() << '\n'
                    << "face l2 err:   " << l2;
                logger.log();
            logger.end();
        } else 
            pFace = pFaceExact;
        logger.beg("recover cell pressure vals and fluxes");
            op.recoverSolution(p, u);
            // logger.buf << "pressure concentration / cell computed solution:\n" << pFace << pCell;
            // logger.log();
            // logger.buf << "pressure cell exact solution:\n" << pCellExact;
            // logger.log();
            // auto pCellDiff = pCellExact;
            // pCellDiff.Update(1., pCell, -1.);
            // for (size_t c = 0; c < numbOfCells; ++c) {
            //     logger.buf << c << ": " << pCellDiff[0][c] << '\n';
            //     AmanziGeometry::Entity_ID_List macroFacesIndicies;
            //     mesh->cell_get_faces(c, &macroFacesIndicies);
            //     for (size_t i = 0; i < macroFacesIndicies.size(); ++i) 
            //         logger.buf << macroFacesIndicies[i] << ' ';
            //     logger.buf << '\n';
            //     for (size_t i = 0; i < macroFacesIndicies.size(); ++i) 
            //         logger.buf << pFace[0][macroFacesIndicies[i]] << ' ';
            //     logger.buf << '\n';
            //     mesh->cell_get_faces(c, &macroFacesIndicies, true);
            //     for (size_t i = 0; i < macroFacesIndicies.size(); ++i) 
            //         logger.buf << macroFacesIndicies[i] << ' ';
            //     logger.buf << '\n';
            //     for (size_t i = 0; i < macroFacesIndicies.size(); ++i) 
            //         logger.buf << pFace[0][macroFacesIndicies[i]] << ' ';
            //     logger.buf << '\n';
            //     if (fpEqual(pCellDiff[0][c], 0.)) logger.log();
            //     else logger.wrn();
            // }    
            // logger.buf << "p_* - p_h:\n" << pCellDiff;
            // logger.log();
            double l2 = 0., vol = 0., pCellMean = 0., pCellExactMean = 0.;
            for (size_t c = 0; c < numbOfCells; ++c) {
                l2 += pow(pCellExact[0][c] - pCell[0][c], 2.) * mesh->cell_volume(c);
                vol += mesh->cell_volume(c);
                pCellMean += pCell[0][c] * mesh->cell_volume(c);
                pCellExactMean += pCellExact[0][c] * mesh->cell_volume(c);
            }
            pCellMean /= vol;
            pCellExactMean /= vol;
            l2 = sqrt(l2);
            logger.buf << "cell l2 err: " << l2 << '\n' 
                       << "volume:      " << vol << '\n'
                       << "p_h mean:    " << pCellMean << '\n'
                       << "p_* mean:    " << pCellExactMean << '\n';
            logger.log();
        logger.end();
        logger.beg("export vtk");
            Amanzi::OutputXDMF io(plist.get<Teuchos::ParameterList>("io"), mesh, true, false);
            io.InitializeCycle(0., 0);
            io.WriteVector(*pCell(0), "p_h", AmanziMesh::CELL);
            // for (size_t c = 0; c < numbOfCells; ++c) pCellExact[1][c] = c;
            // io.WriteMultiVector(pCellExact, { "p_*", "cell_numeration" });
            io.WriteVector(*pCellExact(0), "p_*", AmanziMesh::CELL);
            io.FinalizeCycle();
        logger.end();
        logger.exp("stdin.txt");
    } catch (std::exception const & e) {
        logger.err(e.what());
    }
}


