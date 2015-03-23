/*
  This is the flow component of the Amanzi code. 

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

// TPLs
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "UnitTest++.h"

// Amanzi
#include "GMVMesh.hh"
#include "MeshFactory.hh"
#include "State.hh"

// Flow
#include "Darcy_PK.hh"

/* **************************************************************** */
TEST(FLOW_2D_TRANSIENT_DARCY) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::Flow;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  int MyPID = comm.MyPID();

  if (MyPID == 0) std::cout << "Test: 2D transient Darcy, 2-layer model" << std::endl;

  /* read parameter list */
  std::string xmlFileName = "test/flow_darcy_transient_2D.xml";
  Teuchos::RCP<ParameterList> plist = Teuchos::getParametersFromXmlFile(xmlFileName);

  /* create an SIMPLE mesh framework */
  ParameterList region_list = plist->get<Teuchos::ParameterList>("Regions");
  GeometricModelPtr gm = new GeometricModel(2, region_list, &comm);

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);
  pref.push_back(STKMESH);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  RCP<const Mesh> mesh = meshfactory(0.0, -2.0, 1.0, 0.0, 18, 18, gm);

  /* create a simple state and populate it */
  Amanzi::VerboseObject::hide_line_prefix = true;
  Amanzi::VerboseObject::global_default_level = Teuchos::VERB_EXTREME;

  Teuchos::ParameterList state_list = plist->sublist("State");
  RCP<State> S = rcp(new State(state_list));
  S->RegisterDomainMesh(rcp_const_cast<Mesh>(mesh));

  Teuchos::RCP<Darcy_PK> DPK = Teuchos::rcp(new Darcy_PK(plist, "Flow", S));
  DPK->Setup();
  std::cout << "Owner of " << S->GetField("permeability")->fieldname() 
            << " is " << S->GetField("permeability")->owner() << "\n";

  S->Setup();
  S->InitializeFields();
  S->InitializeEvaluators();

  /* modify the default state for the problem at hand */
  std::string passwd("flow"); 
  Epetra_MultiVector& K = *S->GetFieldData("permeability", passwd)->ViewComponent("cell", false);

  AmanziMesh::Entity_ID_List block;
  mesh->get_set_entities("Material 1", AmanziMesh::CELL, AmanziMesh::OWNED, &block);
  for (int i = 0; i != block.size(); ++i) {
    int c = block[i];
    K[0][c] = 0.1;
    K[1][c] = 2.0;
  }

  mesh->get_set_entities("Material 2", AmanziMesh::CELL, AmanziMesh::OWNED, &block);
  for (int i = 0; i != block.size(); ++i) {
    int c = block[i];
    K[0][c] = 0.5;
    K[1][c] = 0.5;
  }

  *S->GetScalarData("fluid_density", passwd) = 1.0;
  *S->GetScalarData("fluid_viscosity", passwd) = 1.0;
  Epetra_Vector& gravity = *S->GetConstantVectorData("gravity", "state");
  gravity[1] = -1.0;

  S->GetFieldData("specific_storage", passwd)->PutScalar(2.0);

  /* create the initial pressure function */
  Epetra_MultiVector& p = *S->GetFieldData("pressure", passwd)->ViewComponent("cell", false);

  for (int c = 0; c < p.MyLength(); c++) {
    const Point& xc = mesh->cell_centroid(c);
    p[0][c] = xc[1] * (xc[1] + 2.0);
  }

  /* initialize the Darcy process kernel */
  DPK->Initialize();
  S->CheckAllFieldsInitialized();

  /* transient solution */
  double t_old(0.0), t_new, dt(0.1);
  for (int n = 0; n < 10; n++) {
    t_new = t_old + dt;

    DPK->AdvanceStep(t_old, t_new);
    DPK->CommitStep(t_old, t_new);

    t_old = t_new;

    if (MyPID == 0 && n > 5) {
      GMV::open_data_file(*mesh, (std::string)"flow.gmv");
      GMV::start_data();
      GMV::write_cell_data(p, 0, "pressure");
      GMV::close_data_file();
    }
  }

  // Testing secondary fields
  DPK->UpdateLocalFields_();
  const Epetra_MultiVector& darcy_velocity = *S->GetFieldData("darcy_velocity")->ViewComponent("cell");
  Point p5(darcy_velocity[0][5], darcy_velocity[1][5]);

  // Testing recovery
  std::vector<AmanziGeometry::Point> xyz;
  std::vector<AmanziGeometry::Point> velocity;
  DPK->CalculateDarcyVelocity(xyz, velocity);

  CHECK(L22(p5 - velocity[5]) < 1e-10);
  
  for (int n = 0; n < 10; n++) { 
    // std::cout << n << " xyz=" << xyz[n] << " vel=" << velocity[n] << std::endl;
  } 
}


/* **************************************************************** */
TEST(FLOW_3D_TRANSIENT_DARCY) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::Flow;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  int MyPID = comm.MyPID();
  if (MyPID == 0) std::cout << "\nTest: 3D transient Darcy, 3-layer model" << std::endl;

  // read parameter list
  std::string xmlFileName = "test/flow_darcy_transient_3D.xml";
  Teuchos::RCP<ParameterList> plist = Teuchos::getParametersFromXmlFile(xmlFileName);

  /* create an SIMPLE mesh framework */
  ParameterList region_list = plist->get<Teuchos::ParameterList>("Regions");
  GeometricModelPtr gm = new GeometricModel(3, region_list, &comm);

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  RCP<const Mesh> mesh = meshfactory(0.0, 0.0, -2.0, 1.0, 2.0, 0.0, 18, 18, 18, gm);

  /* create and populate flow state */
  Amanzi::VerboseObject::hide_line_prefix = true;

  Teuchos::ParameterList state_list = plist->sublist("State");
  RCP<State> S = rcp(new State(state_list));
  S->RegisterDomainMesh(rcp_const_cast<Mesh>(mesh));

  Teuchos::RCP<Darcy_PK> DPK = Teuchos::rcp(new Darcy_PK(plist, "Flow", S));
  DPK->Setup();
  S->Setup();
  S->InitializeFields();
  S->InitializeEvaluators();

  /* modify the default state for the problem at hand */
  std::string passwd("flow"); 
  Epetra_MultiVector& K = *S->GetFieldData("permeability", passwd)->ViewComponent("cell", false);
  
  AmanziMesh::Entity_ID_List block;
  mesh->get_set_entities("Material 1", AmanziMesh::CELL, AmanziMesh::OWNED, &block);
  for (int i = 0; i != block.size(); ++i) {
    int c = block[i];
    K[0][c] = 0.1;
    K[1][c] = 0.1;
    K[2][c] = 2.0;
  }

  mesh->get_set_entities("Material 2", AmanziMesh::CELL, AmanziMesh::OWNED, &block);
  for (int i = 0; i != block.size(); ++i) {
    int c = block[i];
    K[0][c] = 0.5;
    K[1][c] = 0.5;
    K[2][c] = 0.5;
  }

  *S->GetScalarData("fluid_density", passwd) = 1.0;
  *S->GetScalarData("fluid_viscosity", passwd) = 1.0;
  Epetra_Vector& gravity = *S->GetConstantVectorData("gravity", "state");
  gravity[2] = -1.0;

  S->GetFieldData("specific_storage", passwd)->PutScalar(1.0);
  S->GetFieldData("specific_yield", passwd)->PutScalar(0.0);

  /* create the initial pressure function */
  Epetra_MultiVector& p = *S->GetFieldData("pressure", passwd)->ViewComponent("cell", false);

  for (int c = 0; c < p.MyLength(); c++) {
    const Point& xc = mesh->cell_centroid(c);
    p[0][c] = xc[2] * (xc[2] + 2.0) * (xc[1] + 1.0) * (xc[0] - 1.0);
  }

  /* initialize the Darcy process kernel */
  DPK->Initialize();
  S->CheckAllFieldsInitialized();

  /* transient solution */
  double t_old(0.0), t_new, dt(0.1);
  for (int n = 0; n < 5; n++) {
    t_new = t_old + dt;

    DPK->AdvanceStep(t_old, t_new);
    DPK->CommitStep(t_old, t_new);

    if (MyPID == 0) {
      GMV::open_data_file(*mesh, (std::string)"flow.gmv");
      GMV::start_data();
      GMV::write_cell_data(p, 0, "pressure");
      GMV::close_data_file();
    }
  }

  // Testing recovery
  std::vector<AmanziGeometry::Point> xyz;
  std::vector<AmanziGeometry::Point> velocity;
  DPK->CalculateDarcyVelocity(xyz, velocity);

  int nvel = velocity.size();
  for (int n = 0; n < nvel; n++) { 
    // std::cout << xyz[n] << " " << velocity[n] << std::endl;
  } 
}
