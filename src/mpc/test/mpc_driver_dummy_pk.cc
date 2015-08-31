#include <iostream>
#include "stdlib.h"
#include "math.h"
#include "UnitTest++.h"

#include <Epetra_Comm.h>
#include <Epetra_MpiComm.h>
#include "Epetra_SerialComm.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"

#include "State.hh"
#include "CycleDriver.hh"
#include "PK_Factory.hh"
#include "PK.hh"
#include "pks_dummy_registration.hh"

#include "MeshFactory.hh"
#include "Mesh.hh"
#include "Domain.hh"
#include "GeometricModel.hh"


TEST(NEW_DRIVER_DUMMY_PK) {

using namespace std;

#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else  
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif
  
  std::string xmlInFileName = "test/mpc_driver_dummy.xml";

  // read the main parameter list
  Teuchos::ParameterList plist;
  Teuchos::ParameterXMLFileReader xmlreader(xmlInFileName);
  plist = xmlreader.getParameters();
  
  // For now create one geometric model from all the regions in the spec
  Teuchos::ParameterList reg_params = plist.sublist("Regions");

  int spdim = 2;
  Amanzi::AmanziGeometry::GeometricModelPtr 
      geom_model_ptr( new Amanzi::AmanziGeometry::GeometricModel(spdim, reg_params, comm) );

  Amanzi::AmanziGeometry::Domain *simdomain_ptr = new Amanzi::AmanziGeometry::Domain(spdim);

  simdomain_ptr->Add_Geometric_Model(geom_model_ptr);

  // ---------------- MESH -----------------------------------------------
  int rank, ierr, aerr, size;

  // get the Mesh sublist
  Teuchos::ParameterList mesh_parameter_list = plist.sublist("Mesh");

  Amanzi::VerboseObject *meshverbobj = 
    new Amanzi::VerboseObject("Mesh", plist);

  // Create a mesh factory for this geometric model
  Amanzi::AmanziMesh::MeshFactory factory(comm,meshverbobj) ;

  // get the Mesh sublist
  ierr = 0;
  Teuchos::ParameterList mesh_params = plist.sublist("Mesh");
  
  Teuchos::ParameterList unstr_mesh_params = mesh_params.sublist("Unstructured");

  // Decide on which mesh framework to use
  bool expert_params_specified = unstr_mesh_params.isSublist("Expert");

  Teuchos::RCP<Amanzi::AmanziMesh::Mesh> mesh;

  if (unstr_mesh_params.isSublist("Generate Mesh")) {  // If Read parameters are specified
    Teuchos::ParameterList gen_params = unstr_mesh_params.sublist("Generate Mesh");
    ierr = 0;
    
    try {
      // create the mesh by internal generation
      mesh = factory.create(gen_params, geom_model_ptr);

    } catch (const std::exception& e) {
      std::cerr << rank << ": error: " << e.what() << std::endl;
      ierr++;
    }
  
    comm->SumAll(&ierr, &aerr, 1);
    if (aerr > 0) {
      exit(-aerr);
    }

  } else {  // If Generate parameters are specified
    std::cerr << rank << ": error: " << "Neither Read nor Generate options specified for mesh" << std::endl;
    throw std::exception();
  }

  ASSERT(!mesh.is_null());

  bool mpc_new = true;
  
  // if (input_parameter_list.isParameter("New multi-process coordinator")){
  //   mpc_new = input_parameter_list.get<bool>("New multi-process coordinator",false);
  //   //mpc_new = true;
  // }

  // create dummy observation data object
  Amanzi::ObservationData obs_data;
  Teuchos::RCP<Teuchos::ParameterList> glist_rcp = Teuchos::rcp(new Teuchos::ParameterList(plist));

  Teuchos::RCP<Amanzi::State> S;
  double dt_last;

  if (mpc_new){
    if (plist.isSublist("State")){
      // Create the state.    
      Teuchos::ParameterList state_plist = plist.sublist("State");
      S = Teuchos::rcp(new Amanzi::State(state_plist));
      S->RegisterMesh("domain",mesh);      

      // -------------- MULTI-PROCESS COORDINATOR------- --------------------
      Amanzi::CycleDriver cycle_driver(glist_rcp, S, comm, obs_data);
      //--------------- DO THE SIMULATION -----------------------------------
      cycle_driver.Go();
      //-----------------------------------------------------

      dt_last = cycle_driver.get_dt();
    }
  }

  delete comm;


  CHECK(S->cycle()==28 && fabs(dt_last- 9062.43)<1);
  


}


