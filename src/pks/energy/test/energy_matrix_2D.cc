/*
  This is the energy component of the Amanzi code. 

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

#include "EnergyTwoPhase_PK.hh"
#include "MeshFactory.hh"
#include "GMVMesh.hh"
#include "Operator.hh"
#include "OperatorDiffusion.hh"
#include "OperatorAdvection.hh"
#include "OperatorAccumulation.hh"
#include "OperatorDiffusionFactory.hh"
#include "State.hh"

/* **************************************************************** 
* Generates a preconditioner for the implicit discretization of
* the thermal operator.
* ************************************************************** */
TEST(ENERGY_2D_MATRIX) {
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::Operators;
  using namespace Amanzi::Energy;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  int MyPID = comm.MyPID();

  if (MyPID == 0) std::cout << "Test: 2D homogeneous medium, preconditioner" << std::endl;

  // read parameter list 
  std::string xmlFileName = "test/energy_matrix_2D.xml";
  Teuchos::ParameterXMLFileReader xmlreader(xmlFileName);
  const Teuchos::RCP<Teuchos::ParameterList> plist = 
      Teuchos::rcp(new Teuchos::ParameterList(xmlreader.getParameters()));

  // create a mesh framework
  Teuchos::ParameterList region_list = plist->get<Teuchos::ParameterList>("Regions");
  GeometricModelPtr gm = new GeometricModel(2, region_list, &comm);

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);
  pref.push_back(STKMESH);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  Teuchos::RCP<const Mesh> mesh = meshfactory(0.0, 0.0, 1.0, 1.0, 17, 17, gm);

  // create a simple state and populate it
  Amanzi::VerboseObject::hide_line_prefix = true;

  Teuchos::ParameterList state_list = plist->sublist("State");
  Teuchos::RCP<State> S = Teuchos::rcp(new State(state_list));
  S->RegisterDomainMesh(Teuchos::rcp_const_cast<Mesh>(mesh));

  // initialize the Energy process kernel 
  Teuchos::ParameterList pk_tree;
  Teuchos::RCP<TreeVector> soln;
  EnergyTwoPhase_PK* EPK = new EnergyTwoPhase_PK(pk_tree, plist, S, soln);
  EPK->Setup();
std::cout << "Passes EPK.Setup()" << std::endl;
  S->Setup();
std::cout << "Passed S.Setup()" << std::endl;
  S->InitializeFields();
std::cout << "Passed S.InitilizeFields()" << std::endl;
  EPK->InitializeFields();
std::cout << "Passed EPK.InitilizeField()" << std::endl;
  S->InitializeEvaluators();
std::cout << "Passed S.InitilizeEvaluators()" << std::endl;
  S->WriteDependencyGraph();
  S->CheckAllFieldsInitialized();

  // modify the default state for the problem at hand 
  // create the initial temperature function 
  std::string passwd("thermal");
  Epetra_MultiVector& temperature = *S->GetFieldData("temperature", passwd)->ViewComponent("cell");
  temperature.PutScalar(273.0);
  EPK->get_temperature_eval()->SetFieldAsChanged(S.ptr());

  // compute conductivity
  EPK->UpdateConductivityData(S.ptr());

  // create boundary data
  int nfaces_wghost = mesh->num_entities(AmanziMesh::FACE, AmanziMesh::USED);
  std::vector<int> bc_model(nfaces_wghost, Operators::OPERATOR_BC_NONE);
  std::vector<double> bc_value(nfaces_wghost);
  std::vector<double> bc_mixed(nfaces_wghost);
  
  for (int f = 0; f < nfaces_wghost; f++) {
    const AmanziGeometry::Point& xf = mesh->face_centroid(f);
    if (fabs(xf[0]) < 1e-6 || fabs(xf[0] - 1.0) < 1e-6 ||
        fabs(xf[1]) < 1e-6 || fabs(xf[1] - 1.0) < 1e-6) {
      bc_model[f] = Operators::OPERATOR_BC_DIRICHLET;
      bc_value[f] = 200.0;
    }
  }
  Teuchos::RCP<BCs> bc = Teuchos::rcp(new BCs(OPERATOR_BC_TYPE_FACE, bc_model, bc_value, bc_mixed));
  
  // create diffusion operator 
  const Teuchos::ParameterList& elist = plist->sublist("PKs").sublist("Energy");
  Teuchos::ParameterList oplist = elist.sublist("operators")
                                       .sublist("diffusion operator")
                                       .sublist("preconditioner");
  OperatorDiffusionFactory opfactory;
  AmanziGeometry::Point g(2);
  Teuchos::RCP<OperatorDiffusion> op1 = opfactory.Create(mesh, bc, oplist, g, 0);
  op1->SetBCs(bc);

  // populate the diffusion operator
  double rho(1.0), mu(1.0);
  Teuchos::RCP<std::vector<WhetStone::Tensor> > Kptr = Teuchos::rcpFromRef(EPK->get_K());
  op1->Setup(Kptr, Teuchos::null, Teuchos::null, rho, mu);
  op1->UpdateMatrices(Teuchos::null, Teuchos::null);
  Teuchos::RCP<Operator> op = op1->global_operator();

  // add accumulation term
  Teuchos::RCP<OperatorAccumulation> op2 = Teuchos::rcp(new OperatorAccumulation(AmanziMesh::CELL, op));
  double dT = 1.0;
  CompositeVector solution(op->DomainMap());

  S->GetFieldEvaluator("energy")->HasFieldDerivativeChanged(S.ptr(), passwd, "temperature");
  const CompositeVector& dEdT = *S->GetFieldData("denergy_dtemperature");

  op2->AddAccumulationTerm(solution, dEdT, dT, "cell");

  // add advection term: u = q_l n_l c_v
  // we do not upwind n_l c_v  in this test.
  S->GetFieldEvaluator("internal_energy_liquid")->HasFieldDerivativeChanged(S.ptr(), passwd, "temperature");
  const Epetra_MultiVector& c_v = *S->GetFieldData("dinternal_energy_liquid_dtemperature")
      ->ViewComponent("cell", true);
  const Epetra_MultiVector& n_l = *S->GetFieldData("molar_density_liquid")->ViewComponent("cell", true);

  CompositeVector flux(op->DomainMap());
  Epetra_MultiVector& q_l = *flux.ViewComponent("face");

  AmanziMesh::Entity_ID_List cells;

  AmanziGeometry::Point velocity(1e-4, 1e-4);
  for (int f = 0; f < nfaces_wghost; f++) {
    const AmanziGeometry::Point& normal = mesh->face_normal(f);
    q_l[0][f] = velocity * normal;
    
    mesh->face_get_cells(f, AmanziMesh::USED, &cells);
    int ncells = cells.size();
    double tmp(0.0);
    for (int i = 0; i < ncells; i++) {
      int c = cells[i];
      tmp += n_l[0][c] * c_v[0][c];
    }
    q_l[0][f] *= tmp / ncells;
  }

  Teuchos::ParameterList alist;
  Teuchos::RCP<OperatorAdvection> op3 = Teuchos::rcp(new OperatorAdvection(alist, op));
  op3->Setup(flux);
  op3->UpdateMatrices(flux);

  // build the matrix
  op1->ApplyBCs(true);
  op3->ApplyBCs(bc);
  op->SymbolicAssembleMatrix();
  op->AssembleMatrix();

  // make preconditioner
  // Teuchos::RCP<Operator> op3 = Teuchos::rcp(new Operator(*op2));

  Teuchos::ParameterList slist = plist->sublist("Preconditioners");
  op->InitPreconditioner("Hypre AMG", slist);

  if (MyPID == 0) {
    GMV::open_data_file(*mesh, (std::string)"energy.gmv");
    GMV::start_data();
    GMV::write_cell_data(temperature, 0, "temperature");
    GMV::close_data_file();
  }

  delete EPK;
}
