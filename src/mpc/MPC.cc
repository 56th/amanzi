#include <utility>

#include "errors.hh"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObjectParameterListHelpers.hpp"
#include "Epetra_Comm.h"
#include "Epetra_MpiComm.h"
#include "MPC.hh"
#include "State.hh"
#include "Flow_State.hh"
#include "Darcy_PK.hh"
#include "Richards_PK.hh"
#include "Transport_State.hh"
#include "Transport_PK.hh"

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "TimeStepManager.hh"

// make sure that we use default parameters
// that are consistent with the input translator
#include "InputParserIS-defaults.hh"

#include "TimerManager.hh"

#include "DataDebug.hh"

// Alquimia/Amanzi chemistry packages have different exception types.
#ifdef ALQUIMIA_ENABLED
typedef Exceptions::Amanzi_exception ChemistryExceptionType;
#else
typedef Amanzi::AmanziChemistry::ChemistryException ChemistryExceptionType;
#endif

namespace Amanzi {


/* *******************************************************************/
MPC::MPC(Teuchos::ParameterList parameter_list_,
         Teuchos::RCP<Amanzi::AmanziMesh::Mesh> mesh_maps_,
         Epetra_MpiComm* comm_,
         Amanzi::ObservationData& output_observations_):
    parameter_list(parameter_list_),
    mesh_maps(mesh_maps_),
    chemistry_enabled(false),
    comm(comm_),
    output_observations(output_observations_),
    transport_subcycling(0)
{
  mpc_init();
}


/* *******************************************************************/
void MPC::mpc_init() {
  // set the line prefix for output
  this->setLinePrefix("Amanzi::MPC         ");
  // make sure that the line prefix is printed
  this->getOStream()->setShowLinePrefix(true);

  // Read the sublist for verbosity settings.
  Teuchos::readVerboseObjectSublist(&parameter_list,this);

  using Teuchos::OSTab;
  Teuchos::EVerbosityLevel verbLevel = this->getVerbLevel();
  Teuchos::RCP<Teuchos::FancyOStream> out = this->getOStream();
  OSTab tab = this->getOSTab(); // This sets the line prefix and adds one tab

  mpc_parameter_list =  parameter_list.sublist("MPC");

  read_parameter_list();

  // let users selectively disable individual process kernels
  // to allow for testing of the process kernels separately
  transport_enabled =
      (mpc_parameter_list.get<string>("disable Transport_PK","no") == "no");

  if (mpc_parameter_list.get<string>("Chemistry Model","Off") != "Off") {
    chemistry_enabled = true;
  }

  flow_enabled =
      (mpc_parameter_list.get<string>("disable Flow_PK","no") == "no");

  if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true)) {
    *out << "The following process kernels are enabled: ";
    if (flow_enabled) *out << "Flow ";
    if (transport_enabled) *out << "Transport ";
    if (chemistry_enabled) *out << "Chemistry ";
    *out << std::endl;
  }

  if (transport_enabled || flow_enabled || chemistry_enabled) {
    Teuchos::ParameterList state_parameter_list = parameter_list.sublist("State");
    S = Teuchos::rcp(new State(state_parameter_list));
    S->RegisterMesh("domain",mesh_maps);
  }

  
  // create auxilary state objects for the process models
  

  // chemistry...
  if (chemistry_enabled) {
    Teuchos::ParameterList chemistry_parameter_list = parameter_list.sublist("Chemistry");    
    CS = Teuchos::rcp( new AmanziChemistry::Chemistry_State( chemistry_parameter_list, S ) );
  }
      
  // transport...
  if (transport_enabled) {
    if (mpc_parameter_list.isParameter("component names")) {
      Teuchos::Array<std::string> comp_names;
      comp_names = mpc_parameter_list.get<Teuchos::Array<std::string> >("component names");
      TS = Teuchos::rcp(new AmanziTransport::Transport_State(S, comp_names.toVector() ));
    } else {
      
    }
  }

  // transport and chemistry...
  chem_trans_dt_ratio = CHEM_TRANS_DT_RATIO;
  if (transport_enabled && chemistry_enabled) {
    chem_trans_dt_ratio = parameter_list.sublist("MPC").get<double>("max chemistry to transport timestep ratio",CHEM_TRANS_DT_RATIO);
  }

  // flow...
  if (flow_enabled) {
    FS = Teuchos::rcp(new AmanziFlow::Flow_State(S));
  }

  if (flow_model == "Steady State Richards") {
    *out << "Flow will be off during the transient phase" << std::endl;
  }

  S->Setup();
  
  if (chemistry_enabled) {
    CS->Initialize();
  }
  if (transport_enabled) {
    TS->Initialize();
  }
  if (flow_enabled) {
    FS->Initialize();
  }

  S->Initialize();
 

  if (transport_enabled) {
    bool subcycling = parameter_list.sublist("MPC").get<bool>("transport subcycling", false);
    transport_subcycling = (subcycling) ? 1 : 0;
    TPK = Teuchos::rcp(new AmanziTransport::Transport_PK(parameter_list, TS));
    TPK->InitPK();
  }
    
  if (flow_enabled) { 
    flow_model = mpc_parameter_list.get<string>("Flow model", "Darcy");
    if (flow_model == "Darcy") {
      FPK = Teuchos::rcp(new AmanziFlow::Darcy_PK(parameter_list, FS));
    } else if (flow_model == "Steady State Saturated") {
      FPK = Teuchos::rcp(new AmanziFlow::Darcy_PK(parameter_list, FS));
    } else if (flow_model == "Richards") {
      FPK = Teuchos::rcp(new AmanziFlow::Richards_PK(parameter_list, FS));
    } else if (flow_model == "Steady State Richards") {
      FPK = Teuchos::rcp(new AmanziFlow::Richards_PK(parameter_list, FS));
    } else {
      cout << "MPC: unknown flow model: " << flow_model << endl;
      throw std::exception();
    }   

    FPK->InitPK();
  }

  if (chemistry_enabled) {
    try {
      Teuchos::ParameterList chemistry_parameter_list =
	parameter_list.sublist("Chemistry");
#ifdef ALQUIMIA_ENABLED
      CPK = Teuchos::rcp( new AmanziChemistry::Alquimia_Chemistry_PK(chemistry_parameter_list, CS) );
#else
      CPK = Teuchos::rcp( new AmanziChemistry::Chemistry_PK(chemistry_parameter_list, CS) );
#endif
      CPK->InitializeChemistry();
    } catch (const ChemistryExceptionType& chem_error) {
      std::ostringstream error_message;
      error_message << "MPC:mpc_init(): error... Alquimia_Chemistry_PK.InitializeChemistry returned an error status: ";
      error_message << chem_error.what();
      Errors::Message message(error_message.str());
      Exceptions::amanzi_throw(message);
    }   
  } 
  // done creating auxilary state objects and  process models



  // create the observations
  if (parameter_list.isSublist("Observation Data")) {
    Teuchos::ParameterList observation_plist = parameter_list.sublist("Observation Data");
    observations = new Amanzi::Unstructured_observations(observation_plist, output_observations);

    if (mpc_parameter_list.isParameter("component names")) {
      Teuchos::Array<std::string> comp_names;
      comp_names = mpc_parameter_list.get<Teuchos::Array<std::string> >("component names");
      observations->register_component_names(comp_names.toVector());
    }
  } else {
    observations = NULL;
  }

  // create the visualization object
  if (parameter_list.isSublist("Visualization Data"))  {
    Teuchos::ParameterList vis_parameter_list = parameter_list.sublist("Visualization Data");
    visualization = Teuchos::ptr(new Amanzi::Visualization(vis_parameter_list, comm));
    visualization->CreateFiles();
  } else {  // create a dummy vis object
    visualization = Teuchos::ptr(new Amanzi::Visualization());
  }


  // create the restart object
  if (parameter_list.isSublist("Checkpoint Data")) {
    Teuchos::ParameterList checkpoint_parameter_list = parameter_list.sublist("Checkpoint Data");
    restart = Teuchos::ptr(new Amanzi::Checkpoint(checkpoint_parameter_list, comm));
  } else {
    restart = Teuchos::ptr(new Amanzi::Checkpoint());
  }

  // are we restarting from a file?
  // first assume we're not
  restart_requested = false;

  // then check if indeed we are
  if (mpc_parameter_list.isSublist("Restart from Checkpoint Data File")) {
    restart_requested = true;

    Teuchos::ParameterList& restart_parameter_list =
        mpc_parameter_list.sublist("Restart from Checkpoint Data File");

    restart_from_filename = restart_parameter_list.get<string>("Checkpoint Data File Name");

    // make sure that the restart file actually exists, if not throw an error
    boost::filesystem::path restart_from_filename_path(restart_from_filename);
    if (!boost::filesystem::exists(restart_from_filename_path)) {
      Errors::Message message("MPC: the specified restart file does not exist or is not a regular file.");
      Exceptions::amanzi_throw(message);
    }
  }
}


/* *******************************************************************/
void MPC::read_parameter_list()  {
  end_cycle = mpc_parameter_list.get<int>("End Cycle",-1);

  Teuchos::ParameterList& ti_list =  mpc_parameter_list.sublist("Time Integration Mode");
  if (ti_list.isSublist("Initialize To Steady")) {
    ti_mode = INIT_TO_STEADY;

    Teuchos::ParameterList& init_to_steady_list = ti_list.sublist("Initialize To Steady");

    T0 = init_to_steady_list.get<double>("Start");
    Tswitch = init_to_steady_list.get<double>("Switch");
    T1 = init_to_steady_list.get<double>("End");

    dTsteady = init_to_steady_list.get<double>("Steady Initial Time Step");
    dTtransient = init_to_steady_list.get<double>("Transient Initial Time Step");

    do_picard_ = init_to_steady_list.get<bool>("Use Picard",false);
  } else if ( ti_list.isSublist("Steady")) {
    ti_mode = STEADY;

    Teuchos::ParameterList& steady_list = ti_list.sublist("Steady");

    T0 = steady_list.get<double>("Start");
    T1 = steady_list.get<double>("End");
    dTsteady = steady_list.get<double>("Initial Time Step");

    do_picard_ = steady_list.get<bool>("Use Picard",false);
  } else if ( ti_list.isSublist("Transient") ) {
    ti_mode = TRANSIENT;

    Teuchos::ParameterList& transient_list = ti_list.sublist("Transient");

    T0 = transient_list.get<double>("Start");
    T1 = transient_list.get<double>("End");
    dTtransient =  transient_list.get<double>("Initial Time Step");

    do_picard_ = false;
  } else {
    Errors::Message message("MPC: no valid Time Integration Mode was specified, you must specify exactly one of Initialize To Steady, Steady, or Transient.");
    Exceptions::amanzi_throw(message);
  }

  if (mpc_parameter_list.isSublist("Time Period Control")) {
    Teuchos::ParameterList& tpc_list =  mpc_parameter_list.sublist("Time Period Control");

    reset_times_    = tpc_list.get<Teuchos::Array<double> >("Start Times");
    reset_times_dt_ = tpc_list.get<Teuchos::Array<double> >("Initial Time Step");

    if (reset_times_.size() != reset_times_dt_.size()) {
      Errors::Message message("You must specify the same number of Reset Times and Initial Time Steps under Time Period Control");
      Exceptions::amanzi_throw(message);
    }
  }
}


/* *******************************************************************/
void MPC::cycle_driver() {

  // Amanzi::timer_manager.add("AnalyticJacobian", Amanzi::Timer::ACCUMULATE);
  // Amanzi::timer_manager.add("Function", Amanzi::Timer::ACCUMULATE);
  // Amanzi::timer_manager.add("Update precon", Amanzi::Timer::ACCUMULATE);
  // Amanzi::timer_manager.add("Apply precon", Amanzi::Timer::ACCUMULATE);


  // start timers
  Amanzi::timer_manager.add("Chemistry PK", Amanzi::Timer::ACCUMULATE);
  Amanzi::timer_manager.add("Flow PK", Amanzi::Timer::ACCUMULATE);
  Amanzi::timer_manager.add("Transport PK", Amanzi::Timer::ACCUMULATE);
  Amanzi::timer_manager.add("I/O", Amanzi::Timer::ACCUMULATE);

  // create the time step manager
  Teuchos::Ptr<Amanzi::TimeStepManager> TSM = Teuchos::ptr(new TimeStepManager());
  // register visualization times with the time step manager
  visualization->RegisterWithTimeStepManager(TSM);
  // register observation times with the time step manager
  if (observations) observations->RegisterWithTimeStepManager(TSM);
  // register reset_times
  TSM->RegisterTimeEvent(reset_times_.toVector());
  // if this is an init to steady run, register the switchover time
  if (ti_mode == INIT_TO_STEADY) TSM->RegisterTimeEvent(Tswitch);
  // register the final time
  TSM->RegisterTimeEvent(T1);

  enum time_step_limiter_type {FLOW_LIMITS, TRANSPORT_LIMITS, CHEMISTRY_LIMITS, MPC_LIMITS};
  time_step_limiter_type tslimiter;

  using Teuchos::OSTab;
  Teuchos::EVerbosityLevel verbLevel = this->getVerbLevel();
  Teuchos::RCP<Teuchos::FancyOStream> out = this->getOStream();
  OSTab tab = this->getOSTab(); // This sets the line prefix and adds one tab

  if (transport_enabled || flow_enabled || chemistry_enabled) {
    S->set_time(T0);  // start at time T=T0;
    S->set_initial_time(T0);
    S->set_intermediate_time(Tswitch);
  }

  // if (chemistry_enabled) {
  //   Amanzi::timer_manager.start("Chemistry PK");
  //   try {
  //     // these are the vectors that chemistry will populate with
  //     // the names for the auxillary output vectors and the
  //     // names of components
  //     std::vector<string> compnames;

  //     // // total view needs this to be outside the constructor
  //     // CPK->InitializeChemistry();
  //     // CPK->set_chemistry_output_names(&auxnames);
  //     // CPK->set_component_names(&compnames);

  //     // set the names in the visualization object
  //     // S->set_compnames(compnames);

  //   } catch (const Amanzi::AmanziChemistry::ChemistryException& chem_error) {
  //     std::ostringstream error_message;
  //     error_message << "MPC:mpc_init(): error... Chemistry_PK.InitializeChemistry returned an error status: ";
  //     error_message << chem_error.what();
  //     Errors::Message message(error_message.str());
  //     Exceptions::amanzi_throw(message);
  //   }
  //   Amanzi::timer_manager.stop("Chemistry PK");
  // }


  if (chemistry_enabled) {
    // create stor for chemistry data
    int number_of_secondaries(0);
    if (CS->secondary_activity_coeff() != Teuchos::null) {
      number_of_secondaries = CS->secondary_activity_coeff()->NumVectors();
    }
    chem_data_ = Teuchos::rcp( new chemistry_data (mesh_maps->cell_map(false),
                                                   S->GetFieldData("total_component_concentration")->ViewComponent("cell", true)->NumVectors(),
                                                   CS->number_of_minerals(),
                                                   number_of_secondaries,
                                                   CS->number_of_ion_exchange_sites(),
                                                   CS->number_of_sorption_sites(),
                                                   CS->using_sorption(),
                                                   CS->using_sorption_isotherms()) );
  }

  int iter = 0;  // set the iteration counter to zero
  S->set_cycle(iter);

  // read the checkpoint file as requested
  if (restart_requested == true) {
    // // re-initialize the state object
    // restart->read_state(*S, restart_from_filename);
    // iter = S->get_cycle();
    
    // if (!reset_times_.empty()) {
    //   while (reset_times_.front()<S->get_time()) {
    //     reset_times_.erase(reset_times_.begin());
    //     reset_times_dt_.erase(reset_times_dt_.begin());
    //   }
    // }
  } else { // no restart, we will call the PKs to allow them to init their auxilary data and massage initial conditions
    Amanzi::timer_manager.start("Flow PK");
    if (flow_enabled) FPK->InitializeAuxiliaryData();
    if (do_picard_) {
      FPK->InitPicard(S->time());
      FPK->CommitState(FS);
    }
    Amanzi::timer_manager.stop("Flow PK");
  }

  Amanzi::timer_manager.start("Flow PK");
  if (flow_enabled) {
    if (ti_mode == STEADY  && flow_model != std::string("Steady State Saturated")) {
      // this is the case Richards time stepped to steady state
      // we simply initialize the Flow PK accordingly and the 
      // time stepping loop below takes care of the rest
      FPK->InitSteadyState(S->time(), dTsteady);
    } else if ( ti_mode == TRANSIENT && flow_model !=std::string("Steady State Richards")) {
      FPK->InitTransient(S->time(), dTtransient);
    } else if ( (ti_mode == INIT_TO_STEADY || ti_mode == STEADY) && flow_model == std::string("Steady State Saturated")) {
      // this is the case where we need to solve the Darcy problem first
      // and then either stop (in the STEADY case), or move to the switch time
      // to get the transient part going (INIT_TO_STEADY).
      // note that if a restart was requested, we get the flow field from
      // the checkpoint file we skip the linear Darcy solve and proceed with 
      // the initialization of the transient problem
      if (!restart_requested) {
        FPK->InitSteadyState(S->time(), dTsteady);
        FPK->InitializeSteadySaturated();
        FPK->CommitState(FS);
	if (ti_mode == INIT_TO_STEADY) S->advance_time(Tswitch-T0);
	if (ti_mode == STEADY)         S->advance_time(T1-T0);
      } else {
	FPK->InitTransient(S->time(), dTtransient);
      }
    } else if (ti_mode == INIT_TO_STEADY) {
      if (S->time() < Tswitch) {
        FPK->InitSteadyState(S->time(), dTsteady);
      } else {
        if (flow_model !=std::string("Steady State Richards")) {
          FPK->InitTransient(S->time(), dTtransient);
        }
      }
    }
  }
  Amanzi::timer_manager.stop("Flow PK");


  // write visualization output as requested
  Amanzi::timer_manager.start("I/O");
  visualization->set_mesh(mesh_maps);
  visualization->CreateFiles();
  if (chemistry_enabled) {
    // get the auxillary data from chemistry
    Teuchos::RCP<Epetra_MultiVector> aux = CPK->get_extra_chemistry_output_data();
    // write visualization data for timestep
    WriteVis(visualization,S.ptr()); // TODO: make sure that aux names are used for vis
  } else {
    //always write the initial visualization dump
    WriteVis(visualization,S.ptr());
  }

  // write a restart dump if requested (determined in dump_state)
  WriteCheckpoint(restart,S.ptr(),S->time());
  Amanzi::timer_manager.stop("I/O");


  if (flow_enabled || transport_enabled || chemistry_enabled) {
    if (observations) {
      if (observations->DumpRequested(S->time(), iter)) {
	observations->make_observations(*S);
      }
    }
    // we need to create an EpetraMulitVector that will store the
    // intermediate value for the total component concentration
    if (chemistry_enabled || transport_enabled) {
      total_component_concentration_star =
        Teuchos::rcp(new Epetra_MultiVector(*S->GetFieldData("total_component_concentration")->ViewComponent("cell", true)));
    }
    // then start time stepping
    while ((S->time() < T1) && ((end_cycle == -1) || (iter <= end_cycle))) {

      // log that we are starting a time step
      if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true)) {
        *out << setprecision(5);
        *out << "Cycle " << iter;
        *out << ": starting time step at time(y) = "<< scientific << S->time() / (365.25*60*60*24);
        *out << std::endl;
      }

      // determine the time step we are now going to take
      double chemistry_dT = 1e+99, transport_dT = 1e+99, flow_dT = 1e+99;
      double mpc_dT = 1e+99, limiter_dT = 1e+99, observation_dT = 1e+99;

      // Update our reset times (delete the next one if we just did it)
      if (!reset_times_.empty()) {
        if (S->last_time()>=reset_times_.front()) {
          reset_times_.erase(reset_times_.begin());
          reset_times_dt_.erase(reset_times_dt_.begin());
        }
      }

      // catch the switchover time to transient
      Amanzi::timer_manager.start("Flow PK");
      if (flow_enabled) {
        if (ti_mode == INIT_TO_STEADY && S->last_time() < Tswitch && S->time() >= Tswitch) {
          if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true)) {
            *out << "Steady state computation complete... now running in transient mode." << std::endl;
	    *out << "Tswitch = " << Tswitch << " S->time() = " << S->time() << " S->last_time() = " << S->last_time() << std::endl;
          }
          // only init the transient problem if we need to
          if (flow_model != "Steady State Richards") { //  && flow_model != "Steady State Saturated" )  {
            FPK->InitTransient(S->time(), dTtransient);
	  }
        }
      }

      // find the flow time step
      if (flow_enabled) {
        // only if we are actually running with flow

        if ((ti_mode == STEADY) ||
            (ti_mode == TRANSIENT && flow_model != std::string("Steady State Richards")) ||
            (ti_mode == INIT_TO_STEADY &&
             ( (flow_model == std::string("Steady State Richards") && S->time() >= Tswitch) ||
               (flow_model == std::string("Steady State Saturated") && S->time() >= Tswitch) ||
	       (flow_model == std::string("Richards") ) ) ) ) {
          flow_dT = FPK->CalculateFlowDt();
	}
      }
      Amanzi::timer_manager.stop("Flow PK");

      if (ti_mode == TRANSIENT || (ti_mode == INIT_TO_STEADY && S->time() >= Tswitch)) {
        if (transport_enabled) {
          Amanzi::timer_manager.start("Transport PK");
          double transport_dT_tmp = TPK->EstimateTransportDt();
          if (transport_subcycling == 0) transport_dT = transport_dT_tmp;
          Amanzi::timer_manager.stop("Transport PK");
        }
        if (chemistry_enabled) {
          Amanzi::timer_manager.start("Chemistry PK");
          chemistry_dT = CPK->max_time_step();
          Amanzi::timer_manager.stop("Chemistry PK");
        }
      }

      // take the mpc time step as the min of all suggested time steps
      mpc_dT = std::min(flow_dT, transport_dT);

      // take the mpc time step as the min of the last limiter and itself
      mpc_dT = TSM->TimeStep(S->time(), mpc_dT);

      // figure out who limits the time step
      if (mpc_dT == flow_dT) {
        tslimiter = FLOW_LIMITS;
      } else if (mpc_dT == transport_dT) {
        tslimiter = TRANSPORT_LIMITS;
      } else if (mpc_dT == chemistry_dT) {
        tslimiter = CHEMISTRY_LIMITS;
      } else {
        tslimiter = MPC_LIMITS;
      }

      // make sure we reset the timestep at switchover time
      if (ti_mode == INIT_TO_STEADY && S->time() >= Tswitch && S->last_time() < Tswitch) {
        mpc_dT = std::min( mpc_dT, dTtransient );
        tslimiter = MPC_LIMITS;
      }

      // make sure that if we are currently on a reset time, to reset the time step
      if (! ti_mode == STEADY) {
        if (!reset_times_.empty()) {
          // this is probably iffy...
          if (S->time() == reset_times_.front()) {
            *out << setprecision(5) << "Resetting the time integrator at time(y) = "
                 << std::fixed << S->time()/(365.25*24*60*60) << std::endl;
            mpc_dT = reset_times_dt_.front();
            mpc_dT = TSM->TimeStep(S->time(), mpc_dT);
            tslimiter = MPC_LIMITS;
            // now reset the flow time integrator..
            Amanzi::timer_manager.start("Flow PK");
            if (flow_enabled) FPK->InitTransient(S->time(), mpc_dT);
            Amanzi::timer_manager.stop("Flow PK");
	  }
        }
      }

      if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_MEDIUM,true)) {
        *out << setprecision(5);
        *out << "Cycle " << iter;
        *out << ": proposed time step before flow step dT(y) = " << scientific << mpc_dT / (365.25*60*60*24);
        *out << std::endl;
      }
      

      // steady flow is special, it might redo a time step, so we print
      // time step info after we've advanced steady flow
      // first advance flow
      Amanzi::timer_manager.start("Flow PK");
      if (flow_enabled) {
        if ((ti_mode == STEADY) ||
            (ti_mode == TRANSIENT && flow_model != std::string("Steady State Richards")) ||
            (ti_mode == INIT_TO_STEADY &&
	     ( (flow_model == std::string("Steady State Richards") && S->time() >= Tswitch) ||
               (flow_model == std::string("Steady State Saturated") && S->time() >= Tswitch) || 
	       (flow_model == std::string("Richards")) ) ) ) {
          bool redo(false);
          do {
            redo = false;
            try {
              FPK->Advance(mpc_dT);
            }
            catch (int itr) {
              mpc_dT = 0.5*mpc_dT;
              redo = true;
              tslimiter = FLOW_LIMITS;
              *out << "will repeat time step with smaller dT = " << mpc_dT << std::endl;
            }
          } while (redo);
          FPK->CommitState(FS);
        }
        S->set_final_time(S->initial_time() + mpc_dT);
      }
      Amanzi::timer_manager.stop("Flow PK");
      // write some info about the time step we are about to take
      // first determine what we will write about the time step limiter
      std::string limitstring("");
      switch (tslimiter) {
        case(MPC_LIMITS):
          limitstring = std::string("(mpc limits timestep)");
          break;
        case (TRANSPORT_LIMITS):
          limitstring = std::string("(transport limits timestep)");
          break;
        case (CHEMISTRY_LIMITS):
          limitstring = std::string("(chemistry limits timestep)");
          break;
        case (FLOW_LIMITS):
          break;
      }

      if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true)) {
        *out << setprecision(5);
        *out << "Cycle " << iter;
        *out << ": time step after flow step dT(y) = " << scientific << mpc_dT / (365.25*60*60*24);
        *out << " " << limitstring;
        *out << std::endl;
      }
      // ==============================================================

      // then advance transport and chemistry
      if (ti_mode == TRANSIENT || (ti_mode == INIT_TO_STEADY && S->time() >= Tswitch) ) {
        double tc_dT(mpc_dT);
        double c_dT(chemistry_dT);
        int ntc(1);

	S->set_intermediate_time(S->initial_time());

        if (chemistry_enabled) {
          Amanzi::timer_manager.start("Chemistry PK");
          // reduce chemistry time step according to the
          // ratio with transport time step that is specified
          // in the input file
          double t_dT(transport_dT);
          if (transport_enabled) {
            t_dT = TPK->EstimateTransportDt();
            double ratio(c_dT/t_dT);
            if (ratio > chem_trans_dt_ratio) {
              c_dT = chem_trans_dt_ratio * t_dT;
            }
          }
          if (mpc_dT > c_dT) {
            ntc = floor(mpc_dT/c_dT)+1;
            tc_dT = mpc_dT/static_cast<double>(ntc);
          }

          if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true)) {
            *out << "Subcycling info: MPC is taking " << ntc << " chemistry subcycling timesteps" << std::endl;
            if (transport_enabled) {
              *out << "  (chemistry sub cycling time step) / (transport time step) = " << tc_dT/t_dT << std::endl;
            }
            *out << "  chemistry subcycling timestep = " << tc_dT << std::endl;
          }
          Amanzi::timer_manager.stop("Chemistry PK");
        }

        // at this time we know the time step that we are going to use during subcycling: tc_dT

        if (chemistry_enabled) {
          // first store the chemistry state

	  chem_data_->store(CS->free_ion_species(),
                            CS->primary_activity_coeff(),
                            CS->secondary_activity_coeff(),
                            CS->mineral_volume_fractions(),
                            CS->mineral_specific_surface_area(),
                            CS->total_sorbed(),
                            CS->sorption_sites(),
                            CS->surface_complex_free_site_conc(),
                            CS->ion_exchange_sites(),
                            CS->ion_exchange_ref_cation_conc(),
                            CS->isotherm_kd(),
                            CS->isotherm_freundlich_n(),
                            CS->isotherm_langmuir_b());
        }
        // store the total component concentration, so that we
        // can restore it in the case of a chemistry failure
	Teuchos::RCP<Epetra_MultiVector> tcc_stor;
	if (chemistry_enabled || transport_enabled) {
	  tcc_stor = Teuchos::rcp(new Epetra_MultiVector(*total_component_concentration_star));
	}
        bool success(true);
        int tries(0);

        do {
          // try to subcycle with tc_dT, if that fails, we will cut that time step and try again
	  try {
	    
            // subcycling loop
            for (int iss = 0; iss<ntc; ++iss) {
	      
              // first we do a transport step, or if transport is off, we simply prepare
              // total_component_concentration_star for the chemistry step
              if (transport_enabled) {
                Amanzi::timer_manager.start("Transport PK");
                TPK->Advance(tc_dT);
                if (TPK->get_transport_status() == AmanziTransport::TRANSPORT_STATE_COMPLETE) {
                  // get the transport state and commit it to the state
                  Teuchos::RCP<AmanziTransport::Transport_State> TS_next = TPK->transport_state_next();
                  *total_component_concentration_star = *TS_next->total_component_concentration();
                } else {
                  Errors::Message message("MPC: error... Transport_PK.advance returned an error status");
                  Exceptions::amanzi_throw(message);
                }
                Amanzi::timer_manager.stop("Transport PK");
              } else if (chemistry_enabled ) { // if we're not advancing transport we still need to prepare for chemistry
		*total_component_concentration_star = *S->GetFieldData("total_component_concentration")->ViewComponent("cell", true);
	      }

              // second we do a chemistry step, or if chemistry is off, we simply update 
              // total_component_concentration in state
              if (chemistry_enabled) {

                if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true)) {
                  *out << "Chemistry PK: advancing, current subcycling time step = " << tc_dT << std::endl;
                }
                Amanzi::timer_manager.start("Chemistry PK");
                CPK->advance(tc_dT, total_component_concentration_star);

                Amanzi::timer_manager.stop("Chemistry PK");
		
		*S->GetFieldData("total_component_concentration","state")->ViewComponent("cell", true)
		  = *total_component_concentration_star;		
	      } else {
		if (chemistry_enabled || transport_enabled) {
		  *S->GetFieldData("total_component_concentration","state")->ViewComponent("cell", true)
		    = *total_component_concentration_star;
		}
	      }

              // all went well, so we can advance intermediate time, and call commit state
              // for each pk
              S->set_intermediate_time(S->intermediate_time() + tc_dT);
              Amanzi::timer_manager.start("Transport PK");
              if (transport_enabled) TPK->CommitState(TS);
              Amanzi::timer_manager.stop("Transport PK");
              Amanzi::timer_manager.start("Chemistry PK");
	      if (chemistry_enabled) CPK->commit_state(CS, tc_dT);
              Amanzi::timer_manager.stop("Chemistry PK");
            }
            success = true;
    } catch (const ChemistryExceptionType& chem_error) {
            
            // if the chemistry step failed, we back up to the beginning of
            // the chemistry subcycling loop, but to do that we must restore
            // a few things, such as the chemistry state, total component
            // concentration, and back up the intermediate time

            // decrease the chemistry subcycling timestep and adjust the
            // number of subcycles we need to take accordingly
            ntc = 2*ntc;
            tc_dT = 0.5 * tc_dT;

            // increase the retry count
            ++tries;

            // bail if we've cut the subcycling timestep too many times
            if (tries>=3) {
              Errors::Message message("MPC: cut chemistry subcycling time step too many times, bailing...");
              Exceptions::amanzi_throw(message);
            }

            // the the user know that we're backing up due to a chemistry failure
            if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true)) {
              *out << "Chemistry step failed, reducing chemistry subcycling time step." << std::endl;
              *out << "  new chemistry subcycling time step = " << tc_dT << std::endl;
            }
	    
            // restore chemistry data to the beginning of the subcycling

            chem_data_->retrieve(CS->free_ion_species(),
                                 CS->primary_activity_coeff(),
                                 CS->secondary_activity_coeff(),
                                 CS->mineral_volume_fractions(),
                                 CS->mineral_specific_surface_area(),
                                 CS->total_sorbed(),
                                 CS->sorption_sites(),
                                 CS->surface_complex_free_site_conc(),
                                 CS->ion_exchange_sites(),
                                 CS->ion_exchange_ref_cation_conc(),
                                 CS->isotherm_kd(),
                                 CS->isotherm_freundlich_n(),
                                 CS->isotherm_langmuir_b());

            // restore the total component concentration to the beginning of chemistry subcycling
	    *S->GetFieldData("total_component_concentration","transport")->ViewComponent("cell", true)
	      = *tcc_stor;		
            
            // reset the intermediate time to the beginning
            S->set_intermediate_time(S->initial_time());
	    
            success = false;
          }
	  
	  
	  
        } while (!success);
	
      }
      
      // update the times in the state object
      S->advance_time(mpc_dT);
      S->set_initial_time(S->time());

      // ===========================================================
      // we're done with this time step, commit the state
      // in the process kernels

      if (ti_mode == TRANSIENT || (ti_mode == INIT_TO_STEADY && S->time() >= Tswitch) ) {
        Amanzi::timer_manager.start("Transport PK");
        if (transport_enabled) TPK->CommitState(TS);
        Amanzi::timer_manager.stop("Transport PK");

        Amanzi::timer_manager.start("Chemistry PK");
        if (chemistry_enabled) CPK->commit_state(CS, mpc_dT);
        Amanzi::timer_manager.stop("Chemistry PK");
      }

      if(out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true)) {
        *out << setprecision(5);
        *out << "Cycle " << iter;
        *out << ": complete, new time = " << S->time() / (365.25*60*60*24);
        *out << std::endl;
      }

      // advance the iteration count
      iter++;
      S->set_cycle(iter);

      // make observations
      if (observations) {
	if (observations->DumpRequested(S->time(), iter)) {
	  observations->make_observations(*S);
	}
      } 

      // write visualization if requested
      // force a vis dump and checkpoint in certain cases,
      // such as at the end of the simulation, and
      // at switch-over time
      bool force(false);
      if (abs(S->time() - T1) < 1e-7) {
        force = true;
      }

      if (ti_mode == INIT_TO_STEADY)
        if (abs(S->time() - Tswitch) < 1e-7) {
          force = true;
        }
      
      Amanzi::timer_manager.start("I/O");
      if (chemistry_enabled) {
        // get the auxillary data
        Teuchos::RCP<Epetra_MultiVector> aux = CPK->get_extra_chemistry_output_data();
	if (force || visualization->DumpRequested(S->time(),iter)) {
	  WriteVis(visualization,S.ptr());
	}
      } else {
	if (force || visualization->DumpRequested(S->time(), iter)) {
	  WriteVis(visualization,S.ptr());
	}
      }

      
      // figure out if in the next iteration, we
      // will reset the time integrator, if so we
      // force a checkpoint
      bool force_checkpoint(false);
      if (! ti_mode == STEADY) 
        if (!reset_times_.empty()) 
	  if (S->time() == reset_times_.front())
	    force_checkpoint = true;

      // write restart dump if requested
      if (force || force_checkpoint || restart->DumpRequested(S->time(),iter)) {
	WriteCheckpoint(restart,S.ptr(),S->time());
      }
      Amanzi::timer_manager.stop("I/O");
    }
  }

  // write final visualization dump and checkpoint 
  // if no time stepping was done
  if (iter == 0) {
    ++iter;
    S->set_cycle(iter);
    Amanzi::timer_manager.start("I/O");
    WriteVis(visualization,S.ptr());

    WriteCheckpoint(restart,S.ptr(),S->time());
    Amanzi::timer_manager.stop("I/O");
  }


  // some final output
  if (out.get() && includesVerbLevel(verbLevel,Teuchos::VERB_LOW,true))
  {
    *out << "Simulation complete at cycle " << iter;
    *out << " and Time(y) = "<< S->time()/ (365.25*60*60*24);
    *out << std::endl;
  }
}


/* *******************************************************************/
double MPC::time_step_limiter (double T, double dT, double T_end) {
  double time_remaining = T_end - T;

  if (time_remaining < 0.0) {
    Errors::Message message("MPC: time step limiter logic error, T_end must be greater than T.");
    Exceptions::amanzi_throw(message);
  }

  if (dT >= time_remaining) {
    return time_remaining;
  } else if ( dT > 0.75*time_remaining ) {
    return 0.5*time_remaining;
  } else {
    return dT;
  }
}


/*
void MPC::populate_walkabout_data() {
  // update centroid and velocity in state
  std::vector<Amanzi::AmanziGeometry::Point> walkabout_xyz;
  std::vector<Amanzi::AmanziGeometry::Point> walkabout_velocity;
  std::vector<double> walkabout_phi;
  std::vector<double> walkabout_ws;
  
  // FPK->CalculateDarcyVelocity(xyz, velocity);
  FPK->CalculatePoreVelocity(walkabout_xyz, walkabout_velocity, walkabout_phi, walkabout_ws);
  
  int dim = walkabout_xyz[0].dim();
  
  // if this is the first time we're updating state, create the epetra multivectors
  // first, find the global number of elements
  if (S->get_walkabout_velocity() == Teuchos::null) {
    int numel = walkabout_xyz.size();
    int gnumel;
    
    comm->SumAll(&numel, &gnumel, 1);
    
    Epetra_BlockMap map(gnumel, numel, 1, 0, *comm);
    
    S->get_walkabout_xyz() = Teuchos::rcp(new Epetra_MultiVector(map, dim));
    S->get_walkabout_velocity() = Teuchos::rcp(new Epetra_MultiVector(map, dim));
    
    S->get_walkabout_porosity() = Teuchos::rcp(new Epetra_Vector(map));
    S->get_walkabout_water_saturation() = Teuchos::rcp(new Epetra_Vector(map));
  }
  
  Teuchos::RCP<Epetra_MultiVector> wa_xyz = S->get_walkabout_xyz();
  Teuchos::RCP<Epetra_MultiVector> wa_vel = S->get_walkabout_velocity();
  Teuchos::RCP<Epetra_Vector> wa_phi = S->get_walkabout_porosity();
  Teuchos::RCP<Epetra_Vector> wa_ws = S->get_walkabout_water_saturation();
  
  // then populate the state vectors
  for (int i = 0; i < walkabout_xyz.size(); ++i) {
    for (int j = 0; j < dim; ++j) {
      (*(*wa_xyz)(j))[i] = walkabout_xyz[i][j];
      (*(*wa_vel)(j))[i] = walkabout_velocity[i][j];
    }
    (*wa_phi)[i] = walkabout_phi[i];
    (*wa_ws)[i] = walkabout_ws[i];
  }
}
*/

}  // namespace Amanzi

