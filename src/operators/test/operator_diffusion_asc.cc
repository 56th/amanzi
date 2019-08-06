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

// tangram (for MOF/XMOF interface reconstruction)
#include "wonton/wonton/mesh/amanzi/amanzi_mesh_wrapper.h"
#include "tangram/driver/driver.h"
#include "tangram/driver/write_to_gmv.h"
#include "tangram/reconstruct/MOF.h"
#include "tangram/utility/get_material_moments.h"

// mini-mesh
#include "MeshMiniEmpty.hh"
#include "MeshMiniTangram.hh"

#include "PDE_DiffusionMFD_ASC.hh"
#include "DiffusionReactionEqn.hh"
#include "OutputXDMF.hh"

Tensor constTensor(double c) {
    Tensor K(3, 2);
    K.MakeDiagonal(c);
    return K;
}

// tangram helpers
template<typename T>
std::istream& operator>>(std::istream& inp, T& vec) {
    for (size_t i = 0; i < 3; ++i) inp >> vec[i];
    return inp;
}
template std::istream& operator>>(std::istream&, Tangram::Point3&);
template std::istream& operator>>(std::istream&, Tangram::Vector3&);
bool operator==(Tangram::Vector3 const & u, Tangram::Vector3 const & v) {
    return u[0] == v[0] && u[1] == v[1] && u[2] == v[2]; 
}
bool operator!=(Tangram::Vector3 const & u, Tangram::Vector3 const & v) {
    return !(u == v); 
}
template<size_t N>
std::istream& operator>>(std::istream& inp, std::array<double, N>& vec) {
    for (size_t i = 0; i < N; ++i) inp >> vec[i];
    return inp;
}
template<size_t N>
std::ostream& operator<<(std::ostream& out, std::array<double, N> const & vec) {
    for (size_t i = 0; i < N; ++i) out << vec[i] << ' ';
    return out;
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
        logger.beg("set input parameters");
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
            logger.beg("set t-junction interface");
                Tangram::Point3 p1, p2;
                Tangram::Vector3 n1, n2;
                logger.inp("set p1", p1);
                logger.inp("set p2", p2);
                logger.inp("set n1", n1);
                logger.inp("set n2", n2);
                double noise;
                logger.inp("set noise in [0, 1]", noise);
                if (noise < 0.) noise = 0.;
                if (noise > 1.) noise = 1.;
            logger.end();
            std::array<double, 4> solnCoefs;
            logger.inp(
                "set coefs { a, b, c, d } for exact solution p(x, y, z) = a x + b y + c z + d in mat1 region",
                solnCoefs
            );
            logger.beg("set eqn coefs");
                double k, c;
                logger.inp("set diffusion coef", k);
                logger.inp("set reaction coef", c);
            logger.end();
            auto solveIndex = logger.opt("get the solution", { "linear solve", "recover from exact concentrations" });
            auto deleteEmptyFaces = logger.yes("delete empty mini-faces");
            auto meshMiniIndex = logger.opt("mini-mesh type", { "empty", "tangram" });
            if (n1 != n2 && meshMiniIndex == 1)
                logger.wrn("tangram does not divede T-junction face into two subface; this case currently is not handled by ASC");
            auto meshMiniCheck = logger.yes("check mini-mesh");
            auto exportWhat = logger.opt("export", { "mof mesh only", "soln", "none" });
            logger.exp("stdin.txt");
        logger.end();
        logger.beg("load mesh");
            std::string xmlFileName = "test/operator_diffusion_asc.xml";
            logger.log(xmlFileName);
            ParameterXMLFileReader xmlreader(xmlFileName);
            auto plist = xmlreader.getParameters();
            plist.get<Teuchos::ParameterList>("io").set<std::string>("file name base", plist.get<Teuchos::ParameterList>("io").get<std::string>("file name base") + '/' +  meshName);
            auto ioNameBase = plist.get<Teuchos::ParameterList>("io").get<std::string>("file name base", "amanzi_vis");
            auto region_list = plist.get<Teuchos::ParameterList>("regions");
            Teuchos::RCP<GeometricModel> gm = Teuchos::rcp(new GeometricModel(3, region_list, *comm));
            MeshFactory meshfactory(comm, gm);
            meshfactory.set_preference(Preference({Framework::MSTK, Framework::STK}));
            Teuchos::RCP<const Mesh> mesh = meshfactory.create("test/meshes/" + meshName);
            auto numbOfCells = mesh->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::OWNED);
            auto numbOfFaces = mesh->num_entities(AmanziMesh::FACE, AmanziMesh::Parallel_type::OWNED);
            logger.buf << "numb of cells: " << numbOfCells << '\n'
                       << "numb of faces: " << numbOfFaces;
            logger.log();
            logger.beg("tangram XMOF interface reconstruction");
                Wonton::Amanzi_Mesh_Wrapper meshWrapper(*mesh);
                logger.beg("generate input data");
                    // https://github.com/laristra/portage/blob/master/app/portageapp/portageapp_t-junction_jali.cc
                    std::vector<int> cell_num_mats;
                    std::vector<int> cell_mat_ids;
                    std::vector<double> cell_mat_volfracs;
                    std::vector<Wonton::Point<3>> cell_mat_centroids;
                    const std::vector<int> mesh_materials = {0, 1, 2};
                    const std::vector<Tangram::Vector3> material_interface_normals = { n1, n2 };
                    const std::vector<Tangram::Point3> material_interface_points = { p1, p2 };
                    auto decompose_cells = false; // assuming convex cells
                    int nmesh_materials = static_cast<int>(mesh_materials.size());
                    std::vector<Tangram::Plane_t<3>> material_interfaces(nmesh_materials - 1);
                    for (int iplane = 0; iplane < nmesh_materials - 1; iplane++) {
                        material_interfaces[iplane].normal = material_interface_normals[iplane];
                        material_interfaces[iplane].normal.normalize();
                        material_interfaces[iplane].dist2origin = -Wonton::dot(material_interface_points[iplane].asV(), material_interfaces[iplane].normal);
                    }
                    std::vector<std::vector<std::vector<r3d_poly>>> reference_mat_polys;
                    get_material_moments<Wonton::Amanzi_Mesh_Wrapper>(meshWrapper, material_interfaces, mesh_materials, cell_num_mats, cell_mat_ids, cell_mat_volfracs, cell_mat_centroids, reference_mat_polys, decompose_cells);
                    std::vector<int> offsets(numbOfCells, 0.);
                    for (int icell = 0; icell < numbOfCells - 1; icell++)
                        offsets[icell + 1] = offsets[icell] + cell_num_mats[icell];
                    // add noise
                    for (int icell = 0; icell < numbOfCells; icell++) 
                        if (cell_num_mats[icell] > 1) {
                            auto i1 = offsets[icell];
                            auto i2 = i1 + 1;
                            if (cell_mat_volfracs[i1] < cell_mat_volfracs[i2]) std::swap(i1, i2);
                            auto change = noise * cell_mat_volfracs[i1];
                            cell_mat_volfracs[i1] -= change;
                            cell_mat_volfracs[i2] += change;
                        }
                logger.end();
                logger.beg("interface reconstruction");
                    // https://github.com/laristra/tangram/blob/master/doc/example.md
                    std::vector<Tangram::IterativeMethodTolerances_t> ims_tols(2);
                    ims_tols[0] = {.max_num_iter = 1000, .arg_eps = 1.0e-15, .fun_eps = 1.0e-14};
                    ims_tols[1] = {.max_num_iter = 100, .arg_eps = 1.0e-12, .fun_eps = 1.0e-14};
                    auto all_cells_are_convex = true;
                    Tangram::Driver<Tangram::MOF, 3, Wonton::Amanzi_Mesh_Wrapper, Tangram::SplitR3D, Tangram::ClipR3D> mof_driver(meshWrapper, ims_tols, all_cells_are_convex);
                    mof_driver.set_volume_fractions(cell_num_mats, cell_mat_ids, cell_mat_volfracs, cell_mat_centroids);
                    mof_driver.reconstruct();
                    auto cellmatpoly_list = mof_driver.cell_matpoly_ptrs();
                    // https://github.com/laristra/tangram/blob/master/app/test_mof/test_mof_3d.cc
                    // create MatPoly's for single-material cells
                    for (size_t icell = 0; icell < numbOfCells; icell++) 
                        if (cell_num_mats[icell] == 1 || cellmatpoly_list[icell] == nullptr) {
                            if (cell_num_mats[icell] != 1) {
                                logger.buf << "MMC #" << icell << " marked to have " << cell_num_mats[icell] << " materials, but tangram was not able to create a polycell for it\ncreating a SMC instead";
                                logger.wrn();
                            }
                            std::shared_ptr<Tangram::CellMatPoly<3>> cmp_ptr(new Tangram::CellMatPoly<3>(icell));
                            Tangram::MatPoly<3> cell_matpoly;
                            cell_get_matpoly(meshWrapper, icell, &cell_matpoly);
                            cell_matpoly.set_mat_id(cell_mat_ids[offsets[icell]]);
                            cmp_ptr->add_matpoly(cell_matpoly);
                            cellmatpoly_list[icell] = cmp_ptr;
                        }
                    if (cellmatpoly_list.size() != numbOfCells)
                        throw std::logic_error("cellmatpoly_list.size() != numbOfCells");
                logger.end();
            logger.end();
        logger.end();
        logger.beg("set up mini-mesh");
            Teuchos::RCP<const MeshMini> meshMini;
            if (meshMiniIndex == 0) meshMini = Teuchos::rcp(new MeshMiniEmpty(mesh));
            else meshMini = Teuchos::rcp(new MeshMiniTangram(mesh, cellmatpoly_list, deleteEmptyFaces, meshMiniCheck));
        logger.end();
        logger.beg("set exact soln");
            DiffusionReactionEqn eqn;
            eqn.c = c;
            Node ABC(solnCoefs[0], solnCoefs[1], solnCoefs[2]);
            eqn.p = [=](Node const & x, double) { return ABC * x + solnCoefs[3]; };
            eqn.pGrad = [=](Node const &, double) { return ABC; };
            auto const pHess = constTensor(0.);
            eqn.pHess = [=](Node const &, double) { return pHess; };
            eqn.K = constTensor(k);
            PDE_DiffusionMFD_ASC::BC bc;
            bc.type = PDE_DiffusionMFD_ASC::BCType::Dirichlet;
            bc.p = [=](Node const & x) { return true; };
            bc.f = [&](Node const & x, double t) { return eqn.p(x, t); };
        logger.end();
        logger.beg("set up operator");
            auto olist = plist.sublist("PK operator").sublist("diffusion operator asc");
            PDE_DiffusionMFD_ASC op(olist, meshMini);
            op.setDiffusion([=](size_t) {
                return constTensor(k);
            }).setReaction(eqn.c).setRHS(eqn.f(), 0.).setBC(bc, 0.).assembleLocalConsentrationSystems();
            CompositeVectorSpace cvsP, cvsU;
            cvsP.SetMesh(mesh)->SetGhosted(true)->AddComponent("cell", AmanziMesh::CELL, meshMini->maxNumbOfMaterials());
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
            for (size_t C = 0; C < numbOfCells; ++C) 
                for (size_t c = 0; c < meshMini->numbOfMaterials(C); ++c) {
                    l2 += pow(pCellExact[c][C] - pCell[c][C], 2.) * meshMini->volume(C, c);
                    vol += meshMini->volume(C, c);
                    pCellMean += pCell[c][C] * meshMini->volume(C, c);
                    pCellExactMean += pCellExact[c][C] * meshMini->volume(C, c);
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
        if (exportWhat == 1) {
            logger.beg("export soln (test/io/*)");
                logger.beg("export tangram poly-cells to .gmv and convert to .exo w/o cell renumbering");
                    for (auto& cmp : cellmatpoly_list) // we need it so meshconvert does not renumber cells
                        for (auto& id : const_cast<std::vector<int>&>(cmp->matpoly_matids()))
                            id = 0;
                    for (auto& id : cell_mat_ids) id = 0;
                    write_to_gmv(meshWrapper, meshMini->maxNumbOfMaterials(), cell_num_mats, cell_mat_ids, cellmatpoly_list, ioNameBase + "_flat.gmv");
                    // write_to_gmv(cellmatpoly_list, ioNameBase + "_flat.gmv");
                    int code;
                    #ifdef MESHCONVERT
                        std::string meshconvert = MESHCONVERT;
                        code = system((
                            meshconvert + ' ' + ioNameBase + "_flat.gmv " + ioNameBase + "_flat.exo ; rm " + ioNameBase + "_flat.gmv"
                            // meshconvert + ' ' + ioNameBase + "_flat.gmv " + ioNameBase + "_flat.exo"
                        ).c_str()); 
                    #endif
                logger.end();
                logger.beg("import .exo mesh to amanzi and flatten soln vectors");
                    auto meshFlattened = meshfactory.create(ioNameBase + "_flat.exo", true, false);
                    code = system(("rm " + ioNameBase + "_flat.exo").c_str()); 
                    auto numbOfCellsFlattened = meshFlattened->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::OWNED);
                    auto numbOfFacesFlattened = meshFlattened->num_entities(AmanziMesh::FACE, AmanziMesh::Parallel_type::OWNED);
                    logger.buf 
                        << "numb of cells: " << numbOfCellsFlattened << '\n'
                        << "numb of faces: " << numbOfFacesFlattened;
                    logger.log();
                    CompositeVectorSpace cvsFlattenedP;
                    cvsFlattenedP.SetMesh(meshFlattened)->SetGhosted(true)->AddComponent("cell", AmanziMesh::CELL, 1);
                    auto  pFlattened = CompositeVector(cvsFlattenedP);
                    auto& pCellFlattened = *pFlattened.ViewComponent("cell", true);
                    auto pCellExactFlattened = pCellFlattened;
                    size_t i = 0;
                    for (size_t C = 0; C < numbOfCells; ++C) 
                        for (size_t c = 0; c < meshMini->numbOfMaterials(C); ++c, ++i) {
                            pCellFlattened[0][i] = pCell[c][C];
                            pCellExactFlattened[0][i] = pCellExact[c][C];
                        }
                    logger.buf << "flattened # of pressure d.o.f. = " << i;
                    logger.log();
                logger.end();
                logger.beg("export to .hdf5");
                    Amanzi::OutputXDMF io(plist.get<Teuchos::ParameterList>("io"), meshFlattened, true, false);
                    io.InitializeCycle(0., 0);
                    io.WriteVector(*pCellFlattened(0), "p_h", AmanziMesh::CELL);
                    io.WriteVector(*pCellExactFlattened(0), "p_*", AmanziMesh::CELL);
                    io.FinalizeCycle();
                logger.end();
                // Amanzi::OutputXDMF io(plist.get<Teuchos::ParameterList>("io"), mesh, true, false);
                // io.InitializeCycle(0., 0);
                // io.WriteVector(*pCell(0), "p_h", AmanziMesh::CELL);
                // // for (size_t c = 0; c < numbOfCells; ++c) pCellExact[1][c] = c;
                // // io.WriteMultiVector(pCellExact, { "p_*", "cell_numeration" });
                // io.WriteVector(*pCellExact(0), "p_*", AmanziMesh::CELL);
                // io.FinalizeCycle();
            logger.end();
        }
    } catch (std::exception const & e) {
        logger.err(e.what());
    }
}