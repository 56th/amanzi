/*
  This is the MPC component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Ethan Coon
          Daniil Svyatskiy

  Implementation for the CycleDriver.  CycleDriver is basically just a class to hold
  the cycle driver, which runs the overall, top level timestep loop.  It
  instantiates states, ensures they are initialized, and runs the timestep loop
  including Vis and restart/checkpoint dumps.  It contains one and only one PK
  -- most likely this PK is an MPC of some type -- to do the actual work.
*/

#include <iostream>
#include <unistd.h>
#include <sys/resource.h>
#include "errors.hh"

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

#include "Teuchos_VerboseObjectParameterListHelpers.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include "checkpoint.hh"
#include "CycleDriver.hh"
#include "ObservationData.hh"
#include "PK.hh"
#include "PK_Factory.hh"
#include "TimeStepManager.hh"
#include "TimerManager.hh"
#include "TreeVector.hh"
#include "Unstructured_observations.hh"
#include "State.hh"
#include "visualization.hh"

#define DEBUG_MODE 1

namespace Amanzi {

bool reset_info_compfunc(std::pair<double,double> x, std::pair<double,double> y) {
  return (x.first < y.first);
}


double rss_usage() { // return ru_maxrss in MBytes
#if (defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__) || defined(__MACH__))
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
#if (defined(__APPLE__) || defined(__MACH__))
  return static_cast<double>(usage.ru_maxrss)/1024.0/1024.0;
#else
  return static_cast<double>(usage.ru_maxrss)/1024.0;
#endif
#else
  return 0.0;
#endif
}


/* ******************************************************************
* Constructor.
****************************************************************** */
CycleDriver::CycleDriver(Teuchos::RCP<Teuchos::ParameterList> glist_,
                         Teuchos::RCP<State>& S,
                         Epetra_MpiComm* comm,
                         Amanzi::ObservationData& output_observations) :
    parameter_list_(glist_),
    S_(S),
    comm_(comm),
    output_observations_(output_observations),
    restart_requested_(false) {

  // create and start the global timer
  CoordinatorInit_();

  vo_ = Teuchos::rcp(new VerboseObject("CycleDriver", parameter_list_->sublist("Cycle Driver")));
};


/* ******************************************************************
* High-level initialization.
****************************************************************** */
void CycleDriver::CoordinatorInit_() {
  coordinator_list_ = Teuchos::sublist(parameter_list_, "Cycle Driver");
  ReadParameterList_();

  // create the global solution vector
  soln_ = Teuchos::rcp(new TreeVector());
}


/* ******************************************************************
* Create the pk tree root node which then creates the rest of the tree.
****************************************************************** */
void CycleDriver::Init_PK(int time_pr_id) {
  PKFactory pk_factory;

  Teuchos::RCP<Teuchos::ParameterList> time_periods_list = Teuchos::sublist(coordinator_list_, "time periods", true);

  std::ostringstream ss; ss << time_pr_id;
  std::string tp_list_name = "TP "+ ss.str();
  Teuchos::RCP<Teuchos::ParameterList> tp_list =Teuchos::sublist( time_periods_list, tp_list_name.data(), true);
  Teuchos::ParameterList pk_tree_list = tp_list->sublist("PK Tree");
  if (pk_tree_list.numParams() != 1) {
    Errors::Message message("CycleDriver: PK Tree list should contain exactly one root node list");
    Exceptions::amanzi_throw(message);
  }
  Teuchos::ParameterList::ConstIterator pk_item = pk_tree_list.begin();
  const std::string &pk_name = pk_tree_list.name(pk_item);

  if (!pk_tree_list.isSublist(pk_name)) {
    Errors::Message message("CycleDriver: PK Tree list does not have node \"" + pk_name + "\".");
    Exceptions::amanzi_throw(message);
  }

  pk_ = pk_factory.CreatePK(pk_tree_list.sublist(pk_name), parameter_list_, S_, soln_);
}


/* ******************************************************************
* Setup PK first follwed by State's setup.
****************************************************************** */
void CycleDriver::Setup() {
  // Set up the states, creating all data structures.

  // create the observations
  if (parameter_list_->isSublist("Observation Data")) {
    Teuchos::ParameterList observation_plist = parameter_list_->sublist("Observation Data");
    observations_ = Teuchos::rcp(new Amanzi::Unstructured_observations(observation_plist, output_observations_, comm_));
    if (coordinator_list_->isParameter("component names")) {
      Teuchos::Array<std::string> comp_names =
          coordinator_list_->get<Teuchos::Array<std::string> >("component names");
      observations_->RegisterComponentNames(comp_names.toVector());
    }
  }

  // create the checkpointing
  if (parameter_list_->isSublist("Checkpoint Data")) {
    Teuchos::ParameterList& chkp_plist = parameter_list_->sublist("Checkpoint Data");
    checkpoint_ = Teuchos::rcp(new Amanzi::Checkpoint(chkp_plist, comm_));
  }
  else{
    checkpoint_ = Teuchos::rcp(new Amanzi::Checkpoint());
  }

  // create the walkabout
  if (parameter_list_->isSublist("Walkabout Data")){
    Teuchos::ParameterList& walk_plist = parameter_list_->sublist("Walkabout Data");
    walkabout_ = Teuchos::rcp(new Amanzi::Walkabout_observations(walk_plist, comm_));
  }
  else {
    walkabout_ = Teuchos::rcp(new Amanzi::Walkabout_observations());
  }

  // vis successful steps
  bool surface_done = false;
  for (State::mesh_iterator mesh=S_->mesh_begin(); mesh!=S_->mesh_end(); ++mesh) {
    if (mesh->first == "surface_3d") {
      // pass
    } else if ((mesh->first == "surface") && surface_done) {
      // pass
    } else {
      // vis successful steps
      std::string plist_name = "Visualization Data "+mesh->first;
      // in the case of just a domain mesh, we want to allow no name.
      if ((mesh->first == "domain") && !parameter_list_->isSublist(plist_name)) {
        plist_name = "Visualization Data";
      }

      if (parameter_list_->isSublist(plist_name)) {
        Teuchos::ParameterList& vis_plist = parameter_list_->sublist(plist_name);
        Teuchos::RCP<Visualization> vis = Teuchos::rcp(new Visualization(vis_plist, comm_));
        vis->set_mesh(mesh->second.first);
        vis->CreateFiles();
        visualization_.push_back(vis);
      }

      // vis unsuccessful steps
      std::string fail_plist_name = "Visualization Data "+mesh->first+" Failed Steps";
      // in the case of just a domain mesh, we want to allow no name.
      if ((mesh->first == "domain") && !parameter_list_->isSublist(fail_plist_name)) {
        fail_plist_name = "Visualization Data Failed Steps";
      }

      if (parameter_list_->isSublist(fail_plist_name)) {
        Teuchos::ParameterList& fail_vis_plist = parameter_list_->sublist(fail_plist_name);
        Teuchos::RCP<Visualization> fail_vis =
          Teuchos::rcp(new Visualization(fail_vis_plist, comm_));
        fail_vis->set_mesh(mesh->second.first);
        fail_vis->CreateFiles();
        failed_visualization_.push_back(fail_vis);
      }
    }
  }


  pk_->Setup();
  S_->RequireScalar("dt", "coordinator");
  S_->Setup();

  // create the time step manager
  tsm_ = Teuchos::ptr(new TimeStepManager(parameter_list_->sublist("Cycle Driver")));
  //tsm_ = Teuchos::ptr(new TimeStepManager(vo_));

  // set up the TSM
  // -- register visualization times
  for (std::vector<Teuchos::RCP<Visualization> >::iterator vis=visualization_.begin();
       vis!=visualization_.end(); ++vis) {
    (*vis)->RegisterWithTimeStepManager(tsm_.ptr());
  }
  // -- register checkpoint times
  if (checkpoint_ != Teuchos::null) 
  checkpoint_->RegisterWithTimeStepManager(tsm_.ptr());
  // -- register observation times
  if (observations_ != Teuchos::null) 
    observations_->RegisterWithTimeStepManager(tsm_.ptr());
  // -- register the final time
  // register reset_times
  for(std::vector<std::pair<double,double> >::const_iterator it = reset_info_.begin();
      it != reset_info_.end(); ++it) tsm_->RegisterTimeEvent(it->first);


  for (int i=0;i<num_time_periods_; i++) {
    tsm_->RegisterTimeEvent(tp_end_[i]);
    tsm_->RegisterTimeEvent(tp_start_[i] + tp_dt_[i]);
  } 


  if (vo_->os_OK(Teuchos::VERB_MEDIUM)) {
    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "Setup is complete." << std::endl;
  }
}


/* ******************************************************************
* Initialize State followed by initialization of PK.
****************************************************************** */
void CycleDriver::Initialize() {
 
  *S_->GetScalarData("dt", "coordinator") = tp_dt_[0];
  S_->GetField("dt", "coordinator")->set_initialized();

  // Initialize the state (initializes all dependent variables).
  S_->InitializeFields();
  S_->InitializeEvaluators();

  // Initialize the process kernels
  pk_->Initialize();

  // Final checks.
  S_->CheckNotEvaluatedFieldsInitialized();
  S_->CheckAllFieldsInitialized();

  // S_->WriteDependencyGraph();

  S_->GetMeshPartition("materials");

  // commit the initial conditions.
  // pk_->CommitStep(t0_-get_dt(), get_dt());
  if (!restart_requested_) {
    pk_->CommitStep(S_->time(), S_->time());
    // visualize();
    // checkpoint(*S_->GetScalarData("dt", "coordinator"));
  }
}


/* ******************************************************************
* Force checkpoint at the end of simulation.
* Only do if the checkpoint was not already written, or we would be writing
* the same file twice.
* This really should be removed, but for now is left to help stupid developers.
****************************************************************** */
void CycleDriver::Finalize() {
  if (!checkpoint_->DumpRequested(S_->cycle(), S_->time())) {
    pk_->CalculateDiagnostics();
    Amanzi::WriteCheckpoint(checkpoint_.ptr(), S_.ptr(), 0.0);
  }
}


// double rss_usage() { // return ru_maxrss in MBytes
// #if (defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__) || defined(__MACH__))
//   struct rusage usage;
//   getrusage(RUSAGE_SELF, &usage);
// #if (defined(__APPLE__) || defined(__MACH__))
//   return static_cast<double>(usage.ru_maxrss)/1024.0/1024.0;
// #else
//   return static_cast<double>(usage.ru_maxrss)/1024.0;
// #endif
// #else
//   return 0.0;
// #endif
// }


/* ******************************************************************
* Report the memory high water mark (using ru_maxrss)
* this should be called at the very end of a simulation
****************************************************************** */
void CycleDriver::ReportMemory() {
  if (vo_->os_OK(Teuchos::VERB_MEDIUM)) {
    double global_ncells(0.0);
    double local_ncells(0.0);
    for (State::mesh_iterator mesh = S_->mesh_begin(); mesh != S_->mesh_end(); ++mesh) {
      Epetra_Map cell_map = (mesh->second.first)->cell_map(false);
      global_ncells += cell_map.NumGlobalElements();
      local_ncells += cell_map.NumMyElements();
    }    

    double mem = Amanzi::rss_usage();
    
    double percell(mem);
    if (local_ncells > 0) {
      percell = mem/local_ncells;
    }

    double max_percell(0.0);
    double min_percell(0.0);
    comm_->MinAll(&percell,&min_percell,1);
    comm_->MaxAll(&percell,&max_percell,1);

    double total_mem(0.0);
    double max_mem(0.0);
    double min_mem(0.0);
    comm_->SumAll(&mem,&total_mem,1);
    comm_->MinAll(&mem,&min_mem,1);
    comm_->MaxAll(&mem,&max_mem,1);

    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "======================================================================" << std::endl;
    *vo_->os() << "All meshes combined have " << global_ncells << " cells." << std::endl;
    *vo_->os() << "Memory usage (high water mark):" << std::endl;
    *vo_->os() << std::fixed << std::setprecision(1);
    *vo_->os() << "  Maximum per core:   " << std::setw(7) << max_mem 
          << " MBytes,  maximum per cell: " << std::setw(7) << max_percell*1024*1024 
          << " Bytes" << std::endl;
    *vo_->os() << "  Minumum per core:   " << std::setw(7) << min_mem 
          << " MBytes,  minimum per cell: " << std::setw(7) << min_percell*1024*1024 
         << " Bytes" << std::endl;
    *vo_->os() << "  Total:              " << std::setw(7) << total_mem 
          << " MBytes,  total per cell:   " << std::setw(7) << total_mem/global_ncells*1024*1024 
          << " Bytes" << std::endl;
  }

  
  double doubles_count(0.0);
  for (State::field_iterator field=S_->field_begin(); field!=S_->field_end(); ++field) {
    doubles_count += static_cast<double>(field->second->GetLocalElementCount());
  }
  double global_doubles_count(0.0);
  double min_doubles_count(0.0);
  double max_doubles_count(0.0);
  comm_->SumAll(&doubles_count,&global_doubles_count,1);
  comm_->MinAll(&doubles_count,&min_doubles_count,1);
  comm_->MaxAll(&doubles_count,&max_doubles_count,1);

  Teuchos::OSTab tab = vo_->getOSTab();
  *vo_->os() << "Doubles allocated in state fields " << std::endl;
  *vo_->os() << "  Maximum per core:   " << std::setw(7)
             << max_doubles_count*8/1024/1024 << " MBytes" << std::endl;
  *vo_->os() << "  Minimum per core:   " << std::setw(7)
             << min_doubles_count*8/1024/1024 << " MBytes" << std::endl; 
  *vo_->os() << "  Total:              " << std::setw(7)
             << global_doubles_count*8/1024/1024 << " MBytes" << std::endl;
}


/* ******************************************************************
* TBW.
****************************************************************** */
void CycleDriver::ReadParameterList_() {
  // std::cout<<*coordinator_list_<<"\n";
  // t0_ = coordinator_list_->get<double>("start time");
  // t1_ = coordinator_list_->get<double>("end time");
  // std::string t0_units = coordinator_list_->get<std::string>("start time units", "s");
  // std::string t1_units = coordinator_list_->get<std::string>("end time units", "s");

  // if (t0_units == "s") {
  //   // internal units in s
  // } else if (t0_units == "d") { // days
  //   t0_ = t0_ * 24.0*3600.0;
  // } else if (t0_units == "yr") { // years
  //   t0_ = t0_ * 365.25*24.0*3600.0;
  // } else {
  //   Errors::Message message("CycleDriver: error, invalid start time units");
  //   Exceptions::amanzi_throw(message);
  // }

  // if (t1_units == "s") {
  //   // internal units in s
  // } else if (t1_units == "d") { // days
  //   t1_ = t1_ * 24.0*3600.0;
  // } else if (t1_units == "yr") { // years
  //   t1_ = t1_ * 365.25*24.0*3600.0;
  // } else {
  //   Errors::Message message("CycleDriver: error, invalid end time units");
  //   Exceptions::amanzi_throw(message);
  // }

  max_dt_ = coordinator_list_->get<double>("max time step size", 1.0e99);
  min_dt_ = coordinator_list_->get<double>("min time step size", 1.0e-12);
  cycle0_ = coordinator_list_->get<int>("start cycle",0);
  cycle1_ = coordinator_list_->get<int>("end cycle",-1);

  Teuchos::ParameterList time_periods_list = coordinator_list_->sublist("time periods");

  num_time_periods_ = time_periods_list.numParams();
  Teuchos::ParameterList::ConstIterator item;
  tp_start_.resize(num_time_periods_);
  tp_end_.resize(num_time_periods_);
  tp_dt_.resize(num_time_periods_);
  tp_max_cycle_.resize(num_time_periods_);

  int i = 0;
  for (item = time_periods_list.begin(); item !=time_periods_list.end(); ++item) {
    const std::string & tp_name = time_periods_list.name(item);
    tp_start_[i] = time_periods_list.sublist(tp_name).get<double>("start period time");
    tp_end_[i] = time_periods_list.sublist(tp_name).get<double>("end period time");
    tp_dt_[i] = time_periods_list.sublist(tp_name).get<double>("initial time step", 1.0);
    tp_max_cycle_[i] = time_periods_list.sublist(tp_name).get<int>("maximum cycle number", -1);
   
    std::string t_units = time_periods_list.sublist(tp_name).get<std::string>("start time units", "s");
    if (t_units == "s") {
      // internal units in s
    } else if (t_units == "d") {  // days
      tp_start_[i] = tp_start_[i] * 24.0*3600.0;
    } else if (t_units == "yr") {  // years
      tp_start_[i] = tp_start_[i] * 365.25*24.0*3600.0;
    } else {
      Errors::Message message("CycleDriver: error, invalid start time units");
      Exceptions::amanzi_throw(message);
    }
    t_units = time_periods_list.sublist(tp_name).get<std::string>("end time units", "s");
    if (t_units == "s") {
      // internal units in s
    } else if (t_units == "d") {  // days
      tp_end_[i] = tp_end_[i] * 24.0*3600.0;
    } else if (t_units == "yr") {  // years
      tp_end_[i] = tp_end_[i] * 365.25*24.0*3600.0;
    } else {
      Errors::Message message("CycleDriver: error, invalid end time units");
      Exceptions::amanzi_throw(message);
    }
    t_units = time_periods_list.sublist(tp_name).get<std::string>("initial time step units", "s");
    if (t_units == "s") {
      // internal units in s
    } else if (t_units == "d") {  // days
      tp_dt_[i] = tp_dt_[i] * 24.0*3600.0;
    } else if (t_units == "yr") {  // years
      tp_dt_[i] = tp_dt_[i] * 365.25*24.0*3600.0;
    } else {
      Errors::Message message("CycleDriver: error, invalid initial time step time units");
      Exceptions::amanzi_throw(message);
    }
    i++;
  }

  // restart control
  // are we restarting from a file?
  // first assume we're not
  restart_requested_ = false;

  if (coordinator_list_->isSublist("restart")) {
    restart_requested_ = true;

    Teuchos::ParameterList restart_list = coordinator_list_->sublist("restart");
    restart_filename_ = restart_list.get<std::string>("file name");

    // make sure that the restart file actually exists, if not throw an error
    boost::filesystem::path restart_from_filename_path(restart_filename_);
    if (!boost::filesystem::exists(restart_from_filename_path)) {
      Errors::Message message("CycleDriver::the specified restart file does not exist or is not a regular file.");
      Exceptions::amanzi_throw(message);
    }

    // if (restart_requested_) {
    //   if (vo_->os_OK(Teuchos::VERB_LOW)) {
    //     *vo_->os() << "Restarting from checkpoint file: " << restart_filename_ << std::endl;
    //   }
    // } else {
    //   if (vo_->os_OK(Teuchos::VERB_LOW)) {
    //     *vo_->os() << "Initializing data from checkpoint file: " << restart_filename_ << std::endl
    //                << "    (Ignoring all initial conditions.)" << std::endl;
    //   }
    // }
  }

  if (coordinator_list_->isSublist("time period control")) {
    Teuchos::ParameterList& tpc_list = coordinator_list_->sublist("time period control");
    Teuchos::Array<double> reset_times = tpc_list.get<Teuchos::Array<double> >("start times");
    Teuchos::Array<double> reset_times_dt = tpc_list.get<Teuchos::Array<double> >("initial time step");   
    ASSERT(reset_times.size() == reset_times_dt.size());

    Teuchos::Array<double>::const_iterator it_tim;
    Teuchos::Array<double>::const_iterator it_dt;
    for (it_tim = reset_times.begin(), it_dt = reset_times_dt.begin();
         it_tim != reset_times.end();
         ++it_tim, ++it_dt) {
      reset_info_.push_back(std::make_pair(*it_tim, *it_dt));
    }  

    if (tpc_list.isParameter("Maximal Time Step")) {
      Teuchos::Array<double> reset_max_dt = tpc_list.get<Teuchos::Array<double> >("Maximal Time Step");
      ASSERT(reset_times.size() == reset_max_dt.size());

      Teuchos::Array<double>::const_iterator it_tim;
      Teuchos::Array<double>::const_iterator it_max;
      for (it_tim = reset_times.begin(), it_max = reset_max_dt.begin();
           it_tim != reset_times.end();
           ++it_tim, ++it_max) {
        reset_max_.push_back(std::make_pair(*it_tim, *it_max));
      }  
    }

    // now we sort in ascending order by time
    std::sort(reset_info_.begin(), reset_info_.end(), reset_info_compfunc);
    std::sort(reset_max_.begin(),  reset_max_.end(),  reset_info_compfunc);
  }
}


/* ******************************************************************
* Acquire the chosen timestep size
*******************************************************************/
double CycleDriver::get_dt(bool after_failure) {
  // get the physical step size
  double dt;

  dt = pk_->get_dt();

  std::vector<std::pair<double,double> >::const_iterator it;
  std::vector<std::pair<double,double> >::const_iterator it_max;

  for (it = reset_info_.begin(), it_max = reset_max_.begin(); it != reset_info_.end(); ++it, ++it_max) {
    if (S_->time() == it->first) {
      if (reset_max_.size() > 0) max_dt_ = it_max->second;
      dt = it->second;
      pk_->set_dt(dt);
      after_failure = true;
      break;
    }
  }

  // check if the step size has gotten too small
  if (dt < min_dt_) {
    Errors::Message message("CycleDriver: error, timestep too small");
    Exceptions::amanzi_throw(message);
  }

  if (S_->time() > 0) {
    if (dt/S_->time() < 1e-14) {
      Errors::Message message("CycleDriver: error, timestep too small with respect to current time");
      Exceptions::amanzi_throw(message);
    }
  }

  // ask the step manager if this step is ok
  dt = tsm_->TimeStep(S_->time(), dt, after_failure);

  // cap the max step size
  if (dt > max_dt_) {
    dt = max_dt_;   
    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "Time step is larger than maximum allowed "<<dt<<"\n";
  }

  return dt;
}


/* ******************************************************************
* Time step management.
****************************************************************** */
void CycleDriver::set_dt(double dt) {
  double dt_;

  // check if the step size has gotten too small
  if (dt < min_dt_) {
    Errors::Message message("CycleDriver: error, timestep too small");
    Exceptions::amanzi_throw(message);
  }

  // cap the max step size
  if (dt > max_dt_) {
    dt_ = max_dt_;
  }

  // ask the step manager if this step is ok
  dt_ = tsm_->TimeStep(S_->time() + dt, dt);

  // set the physical step size
  pk_->set_dt(dt_);
}


/* ******************************************************************
* This is used by CLM
****************************************************************** */
double CycleDriver::Advance(double dt) {

  bool advance = true;
  bool fail = false;
  bool reinit = false;
  double dt_new;

  if (tp_end_[time_period_id_] == tp_start_[time_period_id_]) 
    advance = false;

  Teuchos::OSTab tab = vo_->getOSTab();
  
  if (advance) {
    std::vector<std::pair<double,double> >::const_iterator it;
    for (it = reset_info_.begin(); it != reset_info_.end(); ++it) {
      if (it->first == S_->time()) break;
    }

    if (it != reset_info_.end()) {
      if (vo_->os_OK(Teuchos::VERB_MEDIUM)) {
	*vo_->os() << vo_->color("blue") << " Reinitializing PKs due to BCs or sources/sinks" 
                   << vo_->reset() << std::endl;
      }
      reinit = true;
    }      

    fail = pk_->AdvanceStep(S_->time(), S_->time()+dt, reinit);
  }

  if (!fail) {
    pk_->CommitStep(S_->last_time(), S_->time());
    // advance the iteration count and timestep size
    if (advance) {
      S_->advance_cycle();
      S_->advance_time(dt);
    }

    bool force_vis(false);
    bool force_check(false);
    bool force_obser(false);

    if (abs(S_->time() - tp_end_[time_period_id_]) < 1e-10) {
      force_vis = true;
      force_check = true;                       
      force_obser = true;
      S_->set_position(TIME_PERIOD_END);
    }

    dt_new = get_dt(fail);

    if (!reset_info_.empty())
        if (S_->time() == reset_info_.front().first)
            force_check = true;


    // make observations, vis, and checkpoints

    //Amanzi::timer_manager.start("I/O");
    if (advance) {
      pk_->CalculateDiagnostics();
      Visualize(force_vis);
      WriteCheckpoint(dt_new, force_check);   // write Checkpoint with new dt
      Observations(force_obser);
      WriteWalkabout(force_check);
    }
    //Amanzi::timer_manager.start("I/O");

    if (vo_->os_OK(Teuchos::VERB_MEDIUM)) {
      *vo_->os() << "New time(y) = "<< S_->time() / (60*60*24*365.25);
      *vo_->os() << std::endl;
    }
  } else {
    // Failed the timestep.  
    // Potentially write out failed timestep for debugging
    for (std::vector<Teuchos::RCP<Visualization> >::iterator vis=failed_visualization_.begin();
         vis!=failed_visualization_.end(); ++vis) {
      WriteVis((*vis).ptr(), S_.ptr());
    }
    // The timestep sizes have been updated, so copy back old soln and try again.
    // NOT YET IMPLEMENTED, requires PKs to deal with failure.  Fortunately
    // transport and chemistry never fail, so we shouldn't break things.
    // Otherwise this would be very broken, as flow could succeed, but
    // transport fail, and we wouldn't have a way of backing up. --ETC
  }
  return dt_new;
}


/* ******************************************************************
* Make observations.
****************************************************************** */
void CycleDriver::Observations(bool force) {
  if (observations_ != Teuchos::null) {
    if (observations_->DumpRequested(S_->cycle(), S_->time()) || force) {
      // pk_->CalculateDiagnostics();
      int n = observations_->MakeObservations(*S_);
      Teuchos::OSTab tab = vo_->getOSTab();
      *vo_->os() << "Cycle " << S_->cycle() << ": writing observations... " << n << std::endl;
    }
  }
}


/* ******************************************************************
* Write visualization if requested.
****************************************************************** */
void CycleDriver::Visualize(bool force) {
  bool dump = force;
  if (!dump) {
    for (std::vector<Teuchos::RCP<Visualization> >::iterator vis=visualization_.begin();
         vis!=visualization_.end(); ++vis) {
      if ((*vis)->DumpRequested(S_->cycle(), S_->time())) {
        dump = true;
      }
    }
  }

  if (dump || force) //pk_->CalculateDiagnostics();
  
  for (std::vector<Teuchos::RCP<Visualization> >::iterator vis=visualization_.begin();
       vis!=visualization_.end(); ++vis) {
    if (force || (*vis)->DumpRequested(S_->cycle(), S_->time())) {
      WriteVis((*vis).ptr(), S_.ptr());
      Teuchos::OSTab tab = vo_->getOSTab();
      *vo_->os() << "writing visualization file" << std::endl;
    }
  }
}


/* ******************************************************************
* Write a checkpoint file if requested.
****************************************************************** */
void CycleDriver::WriteCheckpoint(double dt, bool force) {
  if (force || checkpoint_->DumpRequested(S_->cycle(), S_->time())) {
    Amanzi::WriteCheckpoint(checkpoint_.ptr(), S_.ptr(), dt);
    
    // if (force) pk_->CalculateDiagnostics();
    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "writing checkpoint file" << std::endl;
  }
}


void CycleDriver::WriteWalkabout(bool force){
  if (walkabout_ != Teuchos::null) {
    if (walkabout_->DumpRequested(S_->cycle(), S_->time()) || force) {
      if (!walkabout_->is_disabled())
         *vo_->os() << "Cycle " << S_->cycle() << ": writing walkabout file" << std::endl;
      walkabout_->WriteWalkabout(S_);
    }
  }

}

/* ******************************************************************
* timestep loop.
****************************************************************** */
void CycleDriver::Go() {

  time_period_id_ = 0;
  int position = 0;
  double restart_time = 0.;

  double dt;
  double restart_dT(1.0e99);

  if (!restart_requested_) {  // No restart
    Init_PK(time_period_id_);
    // start at time t = t0 and initialize the state.
    S_->set_time(tp_start_[time_period_id_]);
    S_->set_cycle(cycle0_);
    S_->set_position(TIME_PERIOD_START);

    Setup();
    Initialize();

    dt = tp_dt_[time_period_id_];
    dt = tsm_->TimeStep(S_->time(), dt);
    pk_->set_dt(dt);
  } else {
    // Read restart file
    restart_time = ReadCheckpointInitialTime(comm_, restart_filename_);
    position = ReadCheckpointPosition(comm_, restart_filename_);
    for (int i = 0; i < num_time_periods_; i++) {
      if (restart_time - tp_end_[i] > -1e-10) 
	time_period_id_++;
    }    
    if (position == TIME_PERIOD_END) 
      if (time_period_id_>0) 
	time_period_id_--;   

    Init_PK(time_period_id_); 
    Setup();
    // Only field which are in State are initialize from the input file
    // to initialize field which are not in the restart file
    S_->InitializeFields();
    S_->InitializeEvaluators();
    
    // re-initialize the state object
    restart_dT = ReadCheckpoint(comm_, Teuchos::ptr(&*S_), restart_filename_);
    cycle0_ = S_->cycle();
    for (std::vector<std::pair<double,double> >::iterator it = reset_info_.begin();
          it != reset_info_.end(); ++it) {
      if (it->first < S_->time()) it = reset_info_.erase(it);
      if (it == reset_info_.end() ) break;
    }

    if (vo_->os_OK(Teuchos::VERB_LOW)) {
      Teuchos::OSTab tab = vo_->getOSTab();
      *vo_->os() << "Restarting from checkpoint file: " << restart_filename_ << std::endl;
    }

    if (position == TIME_PERIOD_END) {
      if (time_period_id_ < num_time_periods_ - 1) time_period_id_++;
      ResetDriver(time_period_id_); 
      restart_dT =  tp_dt_[time_period_id_];
    }
    else {
      Initialize();
    }

    S_->set_initial_time(S_->time());
    dt = tsm_->TimeStep(S_->time(), restart_dT);
    pk_->set_dt(dt);
  }

  *S_->GetScalarData("dt", "coordinator") = dt;
  S_->GetField("dt","coordinator")->set_initialized();

  // visualization at IC
  //Amanzi::timer_manager.start("I/O");
  pk_->CalculateDiagnostics();
  Visualize();
  WriteCheckpoint(dt);
  Observations();
  S_->WriteStatistics(vo_);
  //Amanzi::timer_manager.stop("I/O");
 
  // iterate process kernels
  {
#if !DEBUG_MODE
  try {
#endif
    //bool fail = false;

    while (time_period_id_ < num_time_periods_) {
      int start_cycle_num = S_->cycle();
      do {
        if (vo_->os_OK(Teuchos::VERB_MEDIUM)) {
          Teuchos::OSTab tab = vo_->getOSTab();
          *vo_->os() << "\nCycle " << S_->cycle()
                     << ": time(y) = " << S_->time() / (60*60*24*365.25)
                     << ", dt(y) = " << dt / (60*60*24*365.25) << std::endl;
        }
        *S_->GetScalarData("dt", "coordinator") = dt;
        S_->set_initial_time(S_->time());
        S_->set_final_time(S_->time() + dt);
        S_->set_position(TIME_PERIOD_INSIDE);

        dt = Advance(dt);
        //dt = get_dt(fail);

      }  // while not finished
      while ((S_->time() < tp_end_[time_period_id_]) && ((tp_max_cycle_[time_period_id_] == -1) 
                                     || (S_->cycle() - start_cycle_num <= tp_max_cycle_[time_period_id_])));

      time_period_id_++;
      if (time_period_id_ < num_time_periods_) {
        ResetDriver(time_period_id_); 
        dt = get_dt(false);
      }      
    }
#if !DEBUG_MODE
  }

  catch (Exceptions::Amanzi_exception &e) {
    // write one more vis for help debugging
    S_->advance_cycle();
    visualize(true); // force vis

    // flush observations to make sure they are saved
    observations_->Flush();

    // catch errors to dump two checkpoints -- one as a "last good" checkpoint
    // and one as a "debugging data" checkpoint.
    checkpoint_->set_filebasename("error_checkpoint");
    WriteCheckpoint(checkpoint_.ptr(), S_.ptr(), dt);
    throw e;
  }
#endif
  }
  
  // finalizing simulation
  S_->WriteStatistics(vo_);
  ReportMemory();
  // Finalize();
} 


/* ******************************************************************
* TBW.
****************************************************************** */
void CycleDriver::ResetDriver(int time_pr_id) {

  if (vo_->os_OK(Teuchos::VERB_LOW)) {
    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "Reseting CD: TP " << time_pr_id - 1 << " -> TP " << time_pr_id << "." << std::endl;
  }

  Teuchos::RCP<AmanziMesh::Mesh> mesh = Teuchos::rcp_const_cast<AmanziMesh::Mesh>(S_->GetMesh("domain"));
  S_old_ = S_;

  Teuchos::ParameterList state_plist = parameter_list_->sublist("State");
  S_ = Teuchos::rcp(new Amanzi::State(state_plist));
  S_->RegisterMesh("domain", mesh);
  S_->set_cycle(S_old_->cycle());
  S_->set_time(tp_start_[time_pr_id]); 
  S_->set_position(TIME_PERIOD_START);

  //delete the old global solution vector
  // soln_ = Teuchos::null;
  // pk_ = Teuchos::null;
  
  //if (pk_.get()) delete pk_.get(); 
  pk_ = Teuchos::null;

  //if (soln_.get()) delete soln_.get(); 
  soln_ = Teuchos::null;

  // create the global solution vector
  soln_ = Teuchos::rcp(new TreeVector());
  
  // create new pk
  Init_PK(time_pr_id);

  // register observation times with the time step manager
  //if (observations_ != Teuchos::null) observations_->RegisterWithTimeStepManager(tsm_);

  // Setup
  pk_->Setup();

  S_->RequireScalar("dt", "coordinator");
  S_->Setup();
  *S_->GetScalarData("dt", "coordinator") = tp_dt_[time_pr_id];
  S_->GetField("dt", "coordinator")->set_initialized();

  // Initialize
  S_->InitializeFields();
  S_->InitializeEvaluators();

  // Initialize the state from the old state.
  S_->Initialize(S_old_);

  // Initialize the process kernels variables 
  pk_->Initialize();

  // Final checks
  S_->CheckNotEvaluatedFieldsInitialized();
  S_->CheckAllFieldsInitialized();

  S_->GetMeshPartition("materials");

  pk_->CalculateDiagnostics();
  // Visualize();
  // WriteCheckpoint(dt);
  Observations();
  S_->WriteStatistics(vo_);

  pk_->set_dt(tp_dt_[time_pr_id]);

  S_old_ = Teuchos::null;
}

}  // namespace Amanzi

