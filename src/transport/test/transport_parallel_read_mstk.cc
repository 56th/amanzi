#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "UnitTest++.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"

#include "MeshFactory.hh"
#include "State.hh"
#include "Transport_PK.hh"


double f_step(const Amanzi::AmanziGeometry::Point& x, double t ) { 
  if ( x[0] <= t ) return 1;
  return 0;
}

 
TEST(ADVANCE_WITH_MSTK_PARALLEL_READ) {
  using namespace std;
  using namespace Teuchos;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziTransport;
  using namespace Amanzi::AmanziGeometry;

  std::cout << "Test: advance using parallel MSTK mesh with parallel file read" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  // read parameter list
  ParameterList parameter_list;
  string xmlFileName = "test/transport_parallel_read_mstk.xml";

  ParameterXMLFileReader xmlreader(xmlFileName);
  parameter_list = xmlreader.getParameters();

  // create an MSTK mesh framework 
  ParameterList region_list = parameter_list.get<Teuchos::ParameterList>("Regions");
  GeometricModelPtr gm = new GeometricModel(3, region_list, (Epetra_MpiComm *)comm);
  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory meshfactory(comm);
  meshfactory.preference(pref);
  RCP<Mesh> mesh = meshfactory("test/cube_4x4x4.par", gm);

  //Amanzi::MeshAudit audit(mesh);
  //audit.Verify();
   
  RCP<Transport_State> TS = rcp(new Transport_State(mesh, 2));

  Point u(1.0, 0.0, 0.0);
  TS->Initialize();
  TS->set_darcy_flux(u);
  TS->set_porosity(0.2);
  TS->set_water_saturation(1.0);
  TS->set_prev_water_saturation(1.0);
  TS->set_water_density(1000.0); 
  TS->set_total_component_concentration(f_step,0.0,0);
  TS->set_total_component_concentration(f_step,0.0,1);

  Transport_PK TPK(parameter_list, TS);
  TPK.InitPK();

  // advance the state
  double dT = TPK.CalculateTransportDt();  
  TPK.Advance(dT);

  // printing cell concentration
  int  iter, k;
  double T = 0.0;
  RCP<Transport_State> TS_next = TPK.transport_state_next();
  RCP<Epetra_MultiVector> tcc = TS->total_component_concentration();
  RCP<Epetra_MultiVector> tcc_next = TS_next->total_component_concentration();

  iter = 0;
  while(T < 1.0) {
    dT = TPK.CalculateTransportDt();
    TPK.Advance(dT);
    T += dT;
    iter++;

    if (iter < 10 && TPK.MyPID == 2) {
      printf("T=%7.2f  C_0(x):", T);
      for (int k=0; k<2; k++) printf("%7.4f", (*tcc_next)[0][k]); cout << endl;
    }
    *tcc = *tcc_next;
  }

  for (int k=0; k<12; k++) 
    CHECK_CLOSE((*tcc_next)[0][k], 1.0, 1e-6);
}
 
 


