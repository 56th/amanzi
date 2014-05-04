/*
  This is the operators component of the Amanzi code. 

  Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "UnitTest++.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"

#include "MeshFactory.hh"
#include "Mesh_MSTK.hh"
#include "GMVMesh.hh"

#include "tensor.hh"
#include "mfd3d_diffusion.hh"

#include "LinearOperatorFactory.hh"
#include "OperatorDefs.hh"
#include "Operator.hh"
#include "OperatorDiffusionSurface.hh"
#include "OperatorAccumulation.hh"
#include "OperatorSource.hh"

#include "NonlinearCoefficient.hh"

namespace Amanzi{

class HeatConduction : public Operators::NonlinearCoefficient {
 public:
  HeatConduction(Teuchos::RCP<const AmanziMesh::Mesh> mesh) : mesh_(mesh) { 
    cvalues_ = Teuchos::rcp(new Epetra_Vector(mesh_->cell_map(true)));
    fvalues_ = Teuchos::rcp(new Epetra_Vector(mesh_->face_map(true)));
    fderivatives_ = Teuchos::rcp(new Epetra_Vector(mesh_->face_map(true)));
  }
  ~HeatConduction() {};

  // main members
  void UpdateValues(const CompositeVector& u) { 
    const Epetra_MultiVector& uc = *u.ViewComponent("cell", true); 
    const Epetra_MultiVector& uf = *u.ViewComponent("face", true); 

    int ncells = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::USED);
    for (int c = 0; c < ncells; c++) {
      (*cvalues_)[c] = 0.3 + uc[0][c];
    }

    int nfaces = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::USED);
    for (int f = 0; f < nfaces; f++) {
      (*fvalues_)[f] = 0.3 + uf[0][f];
    }
  }
  void UpdateDerivatives(const CompositeVector& u) {
    fderivatives_->PutScalar(1.0);
  }

 private:
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
};

}  // namespace Amanzi


/* *****************************************************************
* This test replaves tensor and boundary conditions by continuous
* functions. This is a prototype for future solvers.
* **************************************************************** */
TEST(NONLINEAR_OPERATOR) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::Operators;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  int MyPID = comm.MyPID();

  if (MyPID == 0) std::cout << "\nTest: Singular-perturbed nonlinear Laplace Beltrami solver" << std::endl;

  // read parameter list
  std::string xmlFileName = "test/operator_laplace_beltrami.xml";
  ParameterXMLFileReader xmlreader(xmlFileName);
  ParameterList plist = xmlreader.getParameters();

  // create an SIMPLE mesh framework
  ParameterList region_list = plist.get<Teuchos::ParameterList>("Regions Closed");
  GeometricModelPtr gm = new GeometricModel(2, region_list, &comm);

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  RCP<const Mesh> mesh = meshfactory("test/sphere.exo", gm);
  RCP<const Mesh_MSTK> mesh_mstk = rcp_static_cast<const Mesh_MSTK>(mesh);

  // extract surface mesh
  std::vector<std::string> setnames;
  setnames.push_back(std::string("Top surface"));

  RCP<Mesh> surfmesh = Teuchos::rcp(new Mesh_MSTK(*mesh_mstk, setnames, AmanziMesh::FACE));

  /* modify diffusion coefficient */
  std::vector<WhetStone::Tensor> K;
  int ncells_owned = surfmesh->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED);
  int nfaces_wghost = surfmesh->num_entities(AmanziMesh::FACE, AmanziMesh::USED);

  for (int c = 0; c < ncells_owned; c++) {
    WhetStone::Tensor Kc(2, 1);
    Kc(0, 0) = 1.0;
    K.push_back(Kc);
  }

  // create boundary data
  std::vector<int> bc_model(nfaces_wghost, OPERATOR_BC_NONE);
  std::vector<double> bc_values(nfaces_wghost);

  // create solution map.
  Teuchos::RCP<CompositeVectorSpace> cvs = Teuchos::rcp(new CompositeVectorSpace());
  cvs->SetMesh(surfmesh);
  cvs->SetGhosted(true);
  cvs->SetComponent("cell", AmanziMesh::CELL, 1);
  cvs->SetOwned(false);
  cvs->AddComponent("face", AmanziMesh::FACE, 1);

  // create and initialize state variables.
  Teuchos::RCP<CompositeVector> flux = Teuchos::rcp(new CompositeVector(*cvs));

  CompositeVector solution(*cvs);
  solution.PutScalar(0.0);  // solution at time T=0

  CompositeVector phi(*cvs);
  phi.PutScalar(0.2);

  // create source and add it to the operator
  CompositeVector source(*cvs);
  source.PutScalarMasterAndGhosted(0.0);
  
  Epetra_MultiVector& src = *source.ViewComponent("cell");
  for (int c = 0; c < 20; c++) {
    if (MyPID == 0) src[0][c] = 1.0;
  }

  // Create nonlinear coefficient.
  Teuchos::RCP<HeatConduction> knc = Teuchos::rcp(new HeatConduction(surfmesh));

  // MAIN LOOP
  double dT = 1.0;
  for (int loop = 0; loop < 3; loop++) {
    Teuchos::RCP<OperatorSource> op1 = Teuchos::rcp(new OperatorSource(cvs, 0));
    op1->Init();
    op1->UpdateMatrices(source);

    // create accumulation operator
    Teuchos::RCP<OperatorAccumulation> op2 = Teuchos::rcp(new OperatorAccumulation(*op1));
    op2->UpdateMatrices(solution, phi, dT);

    // add diffusion operator
    knc->UpdateValues(solution);
    knc->UpdateDerivatives(solution);
    Teuchos::RCP<OperatorDiffusionSurface> op3 = Teuchos::rcp(new OperatorDiffusionSurface(*op2));

    Teuchos::ParameterList olist;
    int schema_base = Operators::OPERATOR_SCHEMA_BASE_CELL;
    int schema_dofs = Operators::OPERATOR_SCHEMA_DOFS_FACE + Operators::OPERATOR_SCHEMA_DOFS_CELL;
    op3->InitOperator(K, knc, schema_base, schema_dofs, olist);
    op3->UpdateMatrices(flux);
    op3->ApplyBCs(bc_model, bc_values);
    op3->SymbolicAssembleMatrix(Operators::OPERATOR_SCHEMA_DOFS_FACE);
    op3->AssembleMatrixSpecial();

    // create preconditoner
    ParameterList slist = plist.get<Teuchos::ParameterList>("Preconditioners");
    op3->InitPreconditionerSpecial("Hypre AMG", slist, bc_model, bc_values);

    // solve the problem
    ParameterList lop_list = plist.get<Teuchos::ParameterList>("Solvers");
    AmanziSolvers::LinearOperatorFactory<OperatorDiffusionSurface, CompositeVector, CompositeVectorSpace> factory;
    Teuchos::RCP<AmanziSolvers::LinearOperator<OperatorDiffusionSurface, CompositeVector, CompositeVectorSpace> >
       solver = factory.Create("AztecOO CG", lop_list, op3);

    CompositeVector rhs = *op3->rhs();
    int ierr = solver->ApplyInverse(rhs, solution);

    if (MyPID == 0) {
      std::cout << "pressure solver (" << solver->name() 
                << "): ||r||=" << solver->residual() << " itr=" << solver->num_itrs()
                << " code=" << solver->returned_code() << std::endl;
    }

    // derive diffusion flux.
    op3->UpdateFlux(solution, *flux, 0.0);
    // const Epetra_MultiVector& flux_data = *flux->ViewComponent("face");
    // std::cout << flux_data << std::endl;

    // turn off the source
    source.PutScalar(0.0);
  }

  if (MyPID == 0) {
    // visualization
    const Epetra_MultiVector& p = *solution.ViewComponent("cell");
    GMV::open_data_file(*surfmesh, (std::string)"surface_closed.gmv");
    GMV::start_data();
    GMV::write_cell_data(p, 0, "solution");
    GMV::close_data_file();
  }
}


