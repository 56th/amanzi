#include <iostream>
#include "stdlib.h"
#include "math.h"

#include <Epetra_Comm.h>
#include <Epetra_MpiComm.h>
#include "Epetra_SerialComm.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"
#include "UnitTest++.h"

#include "CycleDriver.hh"
#include "MeshFactory.hh"
#include "Mesh.hh"
#include "PK_Factory.hh"
#include "PK.hh"
#include "pks_flow_registration.hh"
#include "State.hh"
#include "wrm_flow_registration.hh"


TEST(MPC_DRIVER_FLOW) {

using namespace Amanzi;
using namespace Amanzi::AmanziMesh;
using namespace Amanzi::AmanziGeometry;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  
  // read the main parameter list
  std::string xmlInFileName = "test/mpc_driver_flow.xml";
  Teuchos::ParameterXMLFileReader xmlreader(xmlInFileName);
  Teuchos::ParameterList plist = xmlreader.getParameters();
  
  // For now create one geometric model from all the regions in the spec
  Teuchos::ParameterList region_list = plist.get<Teuchos::ParameterList>("Regions");
  GeometricModelPtr gm = new GeometricModel(2, region_list, &comm);

  // create mesh
  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  Teuchos::RCP<Amanzi::AmanziMesh::Mesh> mesh = meshfactory(0.0, 0.0, 216.0, 120.0, 54, 60, gm);
  ASSERT(!mesh.is_null());

  // create dummy observation data object
  Amanzi::ObservationData obs_data;    
  Teuchos::RCP<Teuchos::ParameterList> glist_rcp = Teuchos::rcp(new Teuchos::ParameterList(plist));

  if (plist.isSublist("State")) {
    Teuchos::ParameterList state_plist = plist.sublist("State");
    Teuchos::RCP<Amanzi::State> S = Teuchos::rcp(new Amanzi::State(state_plist));
    S->RegisterMesh("domain", mesh);      

    Amanzi::CycleDriver cycle_driver(glist_rcp, S, &comm, obs_data);
    cycle_driver.Go();

    CHECK( S->cycle()==60);
  }
}


