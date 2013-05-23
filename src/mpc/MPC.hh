#ifndef _MPC_HPP_
#define _MPC_HPP_


#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Epetra_MpiComm.h"
#include "State.hh"
//#include "chemistry_state.hh"
//#include "chemistry_pk.hh"
#include "Transport_State.hh"
#include "Transport_PK.hh"
#include "Flow_State.hh"
#include "Flow_PK.hh"
#include "ObservationData.hh"
#include "Unstructured_observations.hh"
#include "visualization.hh"
#include "checkpoint.hh"
#include "chemistry_data.hh"

namespace Amanzi {

enum time_integration_mode { STEADY, TRANSIENT, INIT_TO_STEADY };

class MPC : public Teuchos::VerboseObject<MPC> {
 public:
  MPC(Teuchos::ParameterList parameter_list_,
      Teuchos::RCP<AmanziMesh::Mesh> mesh_maps_,
      Epetra_MpiComm* comm_,
      Amanzi::ObservationData& output_observations_); 
  ~MPC () {};
    
  void cycle_driver ();
    
 private:
  void mpc_init();
  void read_parameter_list();
  double time_step_limiter (double T, double dT, double T_end);

  // states
  Teuchos::RCP<State> S;
  // Teuchos::RCP<amanzi::chemistry::Chemistry_State> CS;
  Teuchos::RCP<AmanziTransport::Transport_State> TS; 
  Teuchos::RCP<AmanziFlow::Flow_State> FS;
    
  // misc setup information
  Teuchos::ParameterList parameter_list;
  Teuchos::RCP<AmanziMesh::Mesh> mesh_maps;
    
  // storage for the component concentration intermediate values
  Teuchos::RCP<Epetra_MultiVector> total_component_concentration_star;
    
  // process kernels
  // Teuchos::RCP<amanzi::chemistry::Chemistry_PK> CPK;
  Teuchos::RCP<AmanziTransport::Transport_PK> TPK;
  Teuchos::RCP<AmanziFlow::Flow_PK> FPK; 
    
  Teuchos::ParameterList mpc_parameter_list;
    
  double T0, T1, dTsteady, dTtransient, Tswitch;
  int end_cycle;
  time_integration_mode ti_mode;
  
  bool steady, init_to_steady;
  bool flow_enabled, transport_enabled, chemistry_enabled;

  int transport_subcycling; 
  double chem_trans_dt_ratio;

  std::string flow_model;
    
  // restart from checkpoint file
  bool restart_requested;
  std::string restart_from_filename;

  // Epetra communicator
  Epetra_MpiComm* comm;
    
  // observations
  Amanzi::ObservationData&  output_observations;
  Amanzi::Unstructured_observations* observations;
    
  // visualization
  Amanzi::Visualization *visualization;
  std::vector<std::string> auxnames;
    
  // checkpoint/restart 
  Amanzi::Checkpoint *restart;
 
  // time period control
  Teuchos::Array<double> reset_times_;
  Teuchos::Array<double> reset_times_dt_;
  
  // picard flag
  bool do_picard_;

  // stor for chemistry data to allow repeat of chemistry step
  Teuchos::RCP<chemistry_data> chem_data_;
};
    
} // namespace Amanzi

#endif
