#include <iostream>
#include "stdlib.h"
#include "math.h"

#include <Epetra_Comm.h>
#include <Epetra_MpiComm.h>
#include "Epetra_SerialComm.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "UnitTest++.h"

#include "CycleDriver.hh"
#include "energy_tcm_registration.hh"
#include "energy_iem_registration.hh"
#include "eos_registration.hh"
#include "Mesh.hh"
#include "MeshFactory.hh"
#include "mpc_pks_registration.hh"
#include "PK_Factory.hh"
#include "PK.hh"
#include "energy_registration.hh"
#include "pks_flow_registration.hh"
#include "pks_transport_registration.hh"
#include "State.hh"
#include "wrm_flow_registration.hh"


TEST(MPC_DRIVER_THERMAL_RICHARDS) {

using namespace Amanzi;
using namespace Amanzi::AmanziMesh;
using namespace Amanzi::AmanziGeometry;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  
  // read the main parameter list
  std::string xmlInFileName = "test/mpc_driver_thermal_richards.xml";
  Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::getParametersFromXmlFile(xmlInFileName);
  
  // For now create one geometric model from all the regions in the spec
  Teuchos::ParameterList region_list = plist->get<Teuchos::ParameterList>("Regions");
  GeometricModelPtr gm = new GeometricModel(2, region_list, &comm);

  // create mesh
  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);
  pref.push_back(STKMESH);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  Teuchos::RCP<Mesh> mesh = meshfactory(0.0, 0.0, 216.0, 120.0, 54, 60, gm);
  ASSERT(!mesh.is_null());

  // create dummy observation data object
  Amanzi::ObservationData obs_data;    

  // Create the state.    
  Teuchos::RCP<Teuchos::ParameterList> state_plist = Teuchos::sublist(plist, "State", true);
  Teuchos::RCP<Amanzi::State> S = Teuchos::rcp(new Amanzi::State(*state_plist));
  S->RegisterMesh("domain", mesh);

  Amanzi::CycleDriver cycle_driver(plist, S, &comm, obs_data);
  cycle_driver.Go();
}


