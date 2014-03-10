#include <winstd.H>
#include <ParmParse.H>
#include <Interpolater.H>
#include <MultiGrid.H>
#include <ArrayLim.H>
#include <Profiler.H>
#include <TagBox.H>
#include <DataServices.H>
#include <AmrData.H>
#include <Utility.H>
#include <time.h> 

#include <PorousMedia.H>
#include <PMAMR_Labels.H>
#include <RegType.H> 
#include <PROB_PM_F.H>
#include <PMAMR_Labels.H>
#include <PMAmr.H> 

#ifdef _OPENMP
#include "omp.h"
#endif

#define SHOWVALARR(val)                        \
{                                              \
    std::cout << #val << " = ";                \
    for (int i=0;i<val.size();++i)             \
    {                                          \
        std::cout << val[i] << " " ;           \
    }                                          \
    std::cout << std::endl;                    \
}                                             
#define SHOWVALARRA(val) { SHOWVALARR(val); BoxLib::Abort();}
#define SHOWVAL(val) { std::cout << #val << " = " << val << std::endl;}
#define SHOWVALA(val) { SHOWVAL(val); BoxLib::Abort();}


#ifdef AMANZI

#ifdef ALQUIMIA_ENABLED
#else 
#include "simple_thermo_database.hh"
#include "activity_model_factory.hh"
#endif 

#endif

#include <TabularFunction.H>

std::ostream& operator<< (std::ostream& os, const Array<std::string>& rhs)
{
    for (int i=0; i<rhs.size(); ++i) {
        os << rhs[i] << " ";
    }
    return os;
}

std::ostream& operator<< (std::ostream& os, const Array<int>& rhs)
{
    for (int i=0; i<rhs.size(); ++i) {
        os << rhs[i] << " ";
    }
    return os;
}

std::ostream& operator<< (std::ostream& os, const Array<Real>& rhs)
{
    for (int i=0; i<rhs.size(); ++i) {
        os << rhs[i] << " ";
    }
    return os;
}

namespace
{
  const std::string solid("Solid");
  const std::string absorbed("Absorbed");
}

//
//**********************************************************************
//
// Set all default values for static variables in InitializeStaticVariables()!!!
//
//**********************************************************************
//

int PorousMedia::echo_inputs;
//
// The num_state_type actually varies with model.
//
// Add 2 if do_tracer_chemistry>0 later.
//
int PorousMedia::num_state_type;
//
// Region.
//
std::string      PorousMedia::surf_file;
PArray<Material> PorousMedia::materials;
//
// Rock
//
MultiFab*   PorousMedia::kappadata;
MultiFab*   PorousMedia::phidata;
Real        PorousMedia::saturation_threshold_for_vg_Kr;
int         PorousMedia::use_shifted_Kr_eval;
//
// Source.
//
bool          PorousMedia::do_source_term;
//
// Phases and components.
//
Array<std::string>  PorousMedia::pNames;
Array<std::string>  PorousMedia::cNames;
Array<int >         PorousMedia::pType;
Array<Real>         PorousMedia::density;
PArray<RegionData>  PorousMedia::ic_array;
PArray<RegionData>  PorousMedia::bc_array;
PArray<RegionData>  PorousMedia::source_array;
Array<Real>         PorousMedia::muval;
int                 PorousMedia::nphases;
int                 PorousMedia::ncomps;
int                 PorousMedia::ndiff;
int                 PorousMedia::idx_dominant;
//
// Tracers.
//
Array<std::string>  PorousMedia::qNames;
Array<std::string>  PorousMedia::tNames;
int                 PorousMedia::ntracers;
Array<int>          PorousMedia::tType; 
Array<Real>         PorousMedia::tDen;
Array<PArray<RegionData> > PorousMedia::tic_array;
Array<PArray<RegionData> > PorousMedia::tbc_array;
Array<PArray<RegionData> > PorousMedia::tsource_array;
std::map<std::string,Array<int> > PorousMedia::group_map;

//
// Minerals and Sorption sites
//
double             PorousMedia::uninitialized_data;
int                PorousMedia::nminerals;
Array<std::string> PorousMedia::minerals;
int                PorousMedia::nsorption_sites;
Array<std::string> PorousMedia::sorption_sites;
int                PorousMedia::ncation_exchange;
int                PorousMedia::nsorption_isotherms;
std::map<std::string, int> PorousMedia::aux_chem_variables;
bool               PorousMedia::using_sorption;
PorousMedia::ChemICMap PorousMedia::sorption_isotherm_ics;
PorousMedia::ChemICMap PorousMedia::mineralogy_ics;
PorousMedia::ChemICMap PorousMedia::surface_complexation_ics;
PorousMedia::ICParmPair PorousMedia::cation_exchange_ics;
PorousMedia::ChemICMap PorousMedia::solute_chem_ics;
PorousMedia::ICLabelParmPair PorousMedia::sorption_chem_ics;
PorousMedia::LabelIdx PorousMedia::mineralogy_label_map;
PorousMedia::LabelIdx PorousMedia::sorption_isotherm_label_map;
PorousMedia::LabelIdx PorousMedia::surface_complexation_label_map;
std::map<std::string,int> PorousMedia::cation_exchange_label_map;
PorousMedia::LabelIdx PorousMedia::solute_chem_label_map;
PorousMedia::LabelIdx PorousMedia::sorption_chem_label_map;

// Pressure.
//
#ifdef MG_USE_FBOXLIB
int         PorousMedia::richard_iter;
#endif
Real        PorousMedia::wt_lo;
Real        PorousMedia::wt_hi;
Array<Real> PorousMedia::press_lo;
Array<Real> PorousMedia::press_hi;
Array<int>  PorousMedia::inflow_bc_lo;
Array<int>  PorousMedia::inflow_bc_hi;
Array<int>  PorousMedia::rinflow_bc_lo;
Array<int>  PorousMedia::rinflow_bc_hi;
//
// Temperature.
//
Real  PorousMedia::temperature;
//
// Flow.
//
int  PorousMedia::verbose;
Real PorousMedia::cfl;
Real PorousMedia::init_shrink;
Real PorousMedia::dt_grow_max;
Real PorousMedia::dt_shrink_max;
Real PorousMedia::fixed_dt;
Real PorousMedia::steady_richard_max_dt;
Real PorousMedia::transient_richard_max_dt;
Real PorousMedia::dt_cutoff;
Real PorousMedia::gravity;
int  PorousMedia::gravity_dir;
Real PorousMedia::z_location;
int  PorousMedia::initial_step;
int  PorousMedia::initial_iter;
int  PorousMedia::sum_interval;
int  PorousMedia::NUM_SCALARS;
int  PorousMedia::NUM_STATE;
int  PorousMedia::full_cycle;
Real PorousMedia::dt_init;
int  PorousMedia::max_n_subcycle_transport;
int  PorousMedia::max_dt_iters_flow;
int  PorousMedia::verbose_chemistry;
bool PorousMedia::abort_on_chem_fail;
int  PorousMedia::show_selected_runtimes;

Array<AdvectionForm> PorousMedia::advectionType;
Array<DiffusionForm> PorousMedia::diffusionType;
//
// Viscosity parameters.
//
Real PorousMedia::be_cn_theta;
Real PorousMedia::visc_tol;
Real PorousMedia::visc_abs_tol;
bool PorousMedia::def_harm_avg_cen2edge;
//
// Capillary pressure flag.
//
int  PorousMedia::have_capillary;
Real PorousMedia::atmospheric_pressure_atm;
std::map<std::string,bool> PorousMedia::use_gauge_pressure;

//
// Molecular diffusion flag.
//
int  PorousMedia::variable_scal_diff;

Array<int>  PorousMedia::is_diffusive;
Array<Real> PorousMedia::visc_coef;
Array<Real> PorousMedia::diff_coef;
//
// Transport flags
//
bool PorousMedia::do_tracer_advection;
bool PorousMedia::do_tracer_diffusion;
bool PorousMedia::setup_tracer_transport;
bool PorousMedia::advect_tracers;
bool PorousMedia::diffuse_tracers;
bool PorousMedia::tensor_tracer_diffusion;
bool PorousMedia::solute_transport_limits_dt;

//
// Chemistry flag.
//
bool  PorousMedia::do_tracer_chemistry;
bool  PorousMedia::react_tracers;
int  PorousMedia::do_full_strang;
int  PorousMedia::n_chem_interval;
int  PorousMedia::it_chem;
Real PorousMedia::dt_chem;
int  PorousMedia::max_grid_size_chem;
bool PorousMedia::no_initial_values;
bool PorousMedia::use_funccount;
bool PorousMedia::do_richard_sat_solve;
//
// Lists.
//
std::map<std::string, int> PorousMedia::phase_list;
std::map<std::string, int> PorousMedia::comp_list;
std::map<std::string, int> PorousMedia::tracer_list;
Array<std::string> PorousMedia::user_derive_list;
PorousMedia::MODEL_ID PorousMedia::model;
//
// AMANZI flags.
//
#ifdef AMANZI

#ifdef ALQUIMIA_ENABLED
Amanzi::AmanziChemistry::Chemistry_Engine* PorousMedia::chemistry_engine;
#else
std::string PorousMedia::amanzi_database_file;
std::string PorousMedia::amanzi_activity_model;
PArray<Amanzi::AmanziChemistry::SimpleThermoDatabase>    PorousMedia::chemSolve(PArrayManage);
Array<Amanzi::AmanziChemistry::Beaker::BeakerComponents> PorousMedia::components;
Array<Amanzi::AmanziChemistry::Beaker::BeakerParameters> PorousMedia::parameters;
#endif

#endif
//
// Internal switches.
//
int  PorousMedia::do_simple;
int  PorousMedia::do_multilevel_full;
bool PorousMedia::use_PETSc_snes_for_evolution;
int  PorousMedia::do_reflux;
int  PorousMedia::do_correct;
int  PorousMedia::no_corrector;
int  PorousMedia::do_kappa_refine;
int  PorousMedia::n_pressure_interval;
int  PorousMedia::it_pressure;
bool PorousMedia::do_any_diffuse;
int  PorousMedia::do_cpl_advect;
Real PorousMedia::ic_chem_relax_dt;
int  PorousMedia::nGrowHYP;
int  PorousMedia::nGrowMG;
int  PorousMedia::nGrowEIGEST;
bool PorousMedia::do_constant_vel;
Real PorousMedia::be_cn_theta_trac;
bool PorousMedia::do_output_flow_time_in_years;
bool PorousMedia::do_output_chemistry_time_in_years;
bool PorousMedia::do_output_transport_time_in_years;

int  PorousMedia::richard_solver_verbose;

//
// Init to steady
//
bool PorousMedia::do_richard_init_to_steady;
int  PorousMedia::richard_init_to_steady_verbose;
int  PorousMedia::steady_min_iterations;
int  PorousMedia::steady_min_iterations_2;
int  PorousMedia::steady_max_iterations;
int  PorousMedia::steady_limit_iterations;
Real PorousMedia::steady_time_step_reduction_factor;
Real PorousMedia::steady_time_step_increase_factor;
Real PorousMedia::steady_time_step_increase_factor_2;
Real PorousMedia::steady_time_step_retry_factor_1;
Real PorousMedia::steady_time_step_retry_factor_2;
Real PorousMedia::steady_time_step_retry_factor_f;
int  PorousMedia::steady_max_consecutive_failures_1;
int  PorousMedia::steady_max_consecutive_failures_2;
Real PorousMedia::steady_init_time_step;
int  PorousMedia::steady_max_time_steps;
Real PorousMedia::steady_max_time_step_size;
Real PorousMedia::steady_max_psuedo_time;
int  PorousMedia::steady_max_num_consecutive_success;
Real PorousMedia::steady_extra_time_step_increase_factor;
int  PorousMedia::steady_max_num_consecutive_increases;
Real PorousMedia::steady_consecutive_increase_reduction_factor;
bool PorousMedia::steady_use_PETSc_snes;
bool PorousMedia::steady_abort_on_psuedo_timestep_failure;
int  PorousMedia::steady_limit_function_evals;
Real PorousMedia::steady_abs_tolerance;
Real PorousMedia::steady_rel_tolerance;
Real PorousMedia::steady_abs_update_tolerance;
Real PorousMedia::steady_rel_update_tolerance;
int  PorousMedia::steady_do_grid_sequence;
Array<Real> PorousMedia::steady_grid_sequence_new_level_dt_factor;
std::string PorousMedia::steady_record_file;

int  PorousMedia::richard_max_ls_iterations;
Real PorousMedia::richard_min_ls_factor;
Real PorousMedia::richard_ls_acceptance_factor;
Real PorousMedia::richard_ls_reduction_factor;
int  PorousMedia::richard_monitor_linear_solve;
int  PorousMedia::richard_monitor_line_search;
Real PorousMedia::richard_perturbation_scale_for_J;
int  PorousMedia::richard_use_fd_jac;
int  PorousMedia::richard_use_dense_Jacobian;
int  PorousMedia::richard_upwind_krel;
int  PorousMedia::richard_pressure_maxorder;
bool PorousMedia::richard_scale_solution_before_solve;
bool PorousMedia::richard_semi_analytic_J;
bool PorousMedia::richard_centered_diff_J;
Real PorousMedia::richard_variable_switch_saturation_threshold;
Real PorousMedia::richard_dt_thresh_pure_steady;

RichardSolver* PorousMedia::richard_solver;
NLScontrol* PorousMedia::richard_solver_control;
RSdata* PorousMedia::richard_solver_data;

PorousMedia::ExecutionMode PorousMedia::execution_mode;
Real PorousMedia::switch_time;

namespace
{
    static void PM_Setup_CleanUpStatics() 
    {
#ifdef ALQUIMIA_ENABLED
      Amanzi::AmanziChemistry::Chemistry_Engine *chemistry_engine = PorousMedia::GetChemistryEngine();
      delete chemistry_engine; chemistry_engine = 0;
#endif
    }
}

static Box grow_box_by_one (const Box& b) { return BoxLib::grow(b,1); }

//
// Components are  Interior, Inflow, Outflow, Symmetry, SlipWall, NoSlipWall.
//

static int scalar_bc[] =
  {
    INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, FOEXTRAP, SEEPAGE
    //INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_ODD, SEEPAGE
  };

static int tracer_bc[] =
  {
    //INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_EVEN, SEEPAGE
    INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, FOEXTRAP, SEEPAGE
  };

static int press_bc[] =
  {
    INT_DIR, FOEXTRAP, EXT_DIR, REFLECT_EVEN, FOEXTRAP, FOEXTRAP
  };

static int norm_vel_bc[] =
  {
    INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_ODD, EXT_DIR, EXT_DIR
  };

static int tang_vel_bc[] =
  {
    INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, HOEXTRAP, EXT_DIR
  };

static BCRec trac_bc; // Set in read_trac, used in variableSetUp

static
void
set_scalar_bc (BCRec&       bc,
               const BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  for (int i = 0; i < BL_SPACEDIM; i++)
    {
      bc.setLo(i,scalar_bc[lo_bc[i]]);
      bc.setHi(i,scalar_bc[hi_bc[i]]);
    }
}

static
void
set_tracer_bc (BCRec&       bc,
               const BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  for (int i = 0; i < BL_SPACEDIM; i++)
    {
      bc.setLo(i,tracer_bc[lo_bc[i]]);
      bc.setHi(i,tracer_bc[hi_bc[i]]);
    }
}

static
void
set_pressure_bc (BCRec&       bc,
                 const BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  for (int i = 0; i < BL_SPACEDIM; i++)
    {
      bc.setLo(i,press_bc[lo_bc[i]]);
      bc.setHi(i,press_bc[hi_bc[i]]);
    }
}

static
void
set_x_vel_bc (BCRec&       bc,
              const BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  bc.setLo(0,norm_vel_bc[lo_bc[0]]);
  bc.setHi(0,norm_vel_bc[hi_bc[0]]);
  bc.setLo(1,tang_vel_bc[lo_bc[1]]);
  bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#if (BL_SPACEDIM == 3)
  bc.setLo(2,tang_vel_bc[lo_bc[2]]);
  bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

static
void
set_y_vel_bc (BCRec&       bc,
              const BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  bc.setLo(0,tang_vel_bc[lo_bc[0]]);
  bc.setHi(0,tang_vel_bc[hi_bc[0]]);
  bc.setLo(1,norm_vel_bc[lo_bc[1]]);
  bc.setHi(1,norm_vel_bc[hi_bc[1]]);
#if (BL_SPACEDIM == 3)
  bc.setLo(2,tang_vel_bc[lo_bc[2]]);
  bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

#if (BL_SPACEDIM == 3)
static
void
set_z_vel_bc (BCRec&       bc,
              const BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  bc.setLo(0,tang_vel_bc[lo_bc[0]]);
  bc.setHi(0,tang_vel_bc[hi_bc[0]]);
  bc.setLo(1,tang_vel_bc[lo_bc[1]]);
  bc.setHi(1,tang_vel_bc[hi_bc[1]]);
  bc.setLo(2,norm_vel_bc[lo_bc[2]]);
  bc.setHi(2,norm_vel_bc[hi_bc[2]]);
}
#endif

typedef StateDescriptor::BndryFunc BndryFunc;
typedef ErrorRec::ErrorFunc ErrorFunc;

struct PMModel
{
  PMModel(PorousMedia::MODEL_ID id = PorousMedia::PM_INVALID) 
    : model(id) {}
  PorousMedia::MODEL_ID model;
};
static std::map<std::string,PMModel> available_models;

void 
PorousMedia::setup_list()
{
  // model list
  available_models["single-phase"] = PMModel(PM_SINGLE_PHASE);
  available_models["single-phase-solid"] = PMModel(PM_SINGLE_PHASE_SOLID);
  available_models["two-phase"] = PMModel(PM_TWO_PHASE);
  available_models["polymer"] = PMModel(PM_POLYMER);
  available_models["richards"] = PMModel(PM_RICHARDS);
  available_models["steady-saturated"] = PMModel(PM_STEADY_SATURATED);
  available_models["saturated"] = PMModel(PM_SATURATED);
}

void
PorousMedia::InitializeStaticVariables ()
{
  //
  // Set all default values for static variables here!!!
  //
  PorousMedia::num_state_type = -1;

  PorousMedia::kappadata = 0;
  PorousMedia::phidata   = 0;

  PorousMedia::do_source_term = false;

  PorousMedia::model        = PorousMedia::PM_INVALID;
  PorousMedia::nphases      = 0;
  PorousMedia::ncomps       = 0; 
  PorousMedia::ndiff        = 0;
  PorousMedia::idx_dominant = -1;

  PorousMedia::ntracers = 0; 
  PorousMedia::uninitialized_data = 1.0e30;
  PorousMedia::nminerals = 0; 
  PorousMedia::minerals.clear();
  PorousMedia::nsorption_sites = 0; 
  PorousMedia::sorption_sites.clear();
  PorousMedia::ncation_exchange = 0;
  PorousMedia::nsorption_isotherms = 0;
  PorousMedia::aux_chem_variables.clear();
  PorousMedia::using_sorption = false;
  PorousMedia::sorption_isotherm_ics.clear();
  PorousMedia::mineralogy_ics.clear();
  PorousMedia::surface_complexation_ics.clear();
  PorousMedia::cation_exchange_ics.clear();
  PorousMedia::solute_chem_ics.clear();
  PorousMedia::sorption_chem_ics.clear();
  PorousMedia::mineralogy_label_map.clear();
  PorousMedia::sorption_isotherm_label_map.clear();
  PorousMedia::surface_complexation_label_map.clear();
  PorousMedia::cation_exchange_label_map.clear();
  PorousMedia::sorption_chem_label_map.clear();
  
#ifdef MG_USE_FBOXLIB
  PorousMedia::richard_iter = 100;
#endif
  PorousMedia::wt_lo = 0;
  PorousMedia::wt_hi = 0;

  PorousMedia::temperature = 300;

  PorousMedia::verbose      = 0;
  PorousMedia::cfl          = 0.8;
  PorousMedia::init_shrink  = 1.0;
  PorousMedia::dt_init      = -1.0; // Ignore if < 0
  PorousMedia::dt_grow_max  = -1;
  PorousMedia::dt_shrink_max  = 10;
  PorousMedia::fixed_dt     = -1.0;
  PorousMedia::steady_richard_max_dt = -1; // Ignore if < 0
  PorousMedia::transient_richard_max_dt = -1; // Ignore if < 0
  PorousMedia::dt_cutoff    = 0.0;
  PorousMedia::gravity      = 9.807 / BL_ONEATM;
  PorousMedia::gravity_dir  = BL_SPACEDIM-1;
  PorousMedia::z_location   = 0;
  PorousMedia::initial_step = false;
  PorousMedia::initial_iter = false;
  PorousMedia::sum_interval = 1;
  PorousMedia::NUM_SCALARS  = 0;
  PorousMedia::NUM_STATE    = 0;
  PorousMedia::full_cycle   = 0;

  PorousMedia::be_cn_theta           = 0.5;
  PorousMedia::visc_tol              = 1.0e-10;  
  PorousMedia::visc_abs_tol          = 1.0e-10;  
  PorousMedia::def_harm_avg_cen2edge = true;

  PorousMedia::have_capillary = 0;
  PorousMedia::atmospheric_pressure_atm = 1;
  PorousMedia::saturation_threshold_for_vg_Kr = -1; // <0 bypasses smoothing
  PorousMedia::use_shifted_Kr_eval = 0; //

  PorousMedia::variable_scal_diff = 1; 

  PorousMedia::do_tracer_chemistry = false;
  PorousMedia::do_tracer_advection = false;
  PorousMedia::do_tracer_diffusion = false;
  PorousMedia::setup_tracer_transport = false;
  PorousMedia::advect_tracers     = false;
  PorousMedia::diffuse_tracers    = false;
  PorousMedia::tensor_tracer_diffusion = false;
  PorousMedia::do_full_strang     = 0;
  PorousMedia::n_chem_interval    = 0;
  PorousMedia::it_chem            = 0;
  PorousMedia::dt_chem            = 0;
  PorousMedia::max_grid_size_chem = 16;
  PorousMedia::no_initial_values  = true;
  PorousMedia::use_funccount      = false;

  PorousMedia::do_simple           = 0;
  PorousMedia::do_multilevel_full  = 1;
  PorousMedia::use_PETSc_snes_for_evolution = true;
  PorousMedia::do_reflux           = 1;
  PorousMedia::do_correct          = 0;
  PorousMedia::no_corrector        = 0;
  PorousMedia::do_kappa_refine     = 0;
  PorousMedia::n_pressure_interval = 0;
  PorousMedia::it_pressure         = 0;  
  PorousMedia::do_any_diffuse      = false;
  PorousMedia::do_cpl_advect       = 0;
  PorousMedia::do_richard_sat_solve = false;
  PorousMedia::execution_mode      = PorousMedia::INVALID;
  PorousMedia::switch_time         = 0;
  PorousMedia::ic_chem_relax_dt    = -1; // < 0 implies not done
  PorousMedia::solute_transport_limits_dt = false;
  PorousMedia::do_constant_vel = false;
  PorousMedia::nGrowHYP = 3;
  PorousMedia::nGrowMG = 1;
  PorousMedia::nGrowEIGEST = 1;
  PorousMedia::max_n_subcycle_transport = 10;
  PorousMedia::max_dt_iters_flow = 20;
  PorousMedia::verbose_chemistry = 0;
  PorousMedia::abort_on_chem_fail = true;
  PorousMedia::show_selected_runtimes = 0;
  PorousMedia::be_cn_theta_trac = 0.5;
  //PorousMedia::do_output_flow_time_in_years = true;
  PorousMedia::do_output_flow_time_in_years = false;
  PorousMedia::do_output_chemistry_time_in_years = true;
  PorousMedia::do_output_transport_time_in_years = false;

  PorousMedia::richard_solver_verbose = 2;

  PorousMedia::do_richard_init_to_steady = false;
  PorousMedia::richard_init_to_steady_verbose = 1;
  PorousMedia::steady_min_iterations = 10;
  PorousMedia::steady_min_iterations_2 = 2;
  PorousMedia::steady_max_iterations = 15;
  PorousMedia::steady_limit_iterations = 20;
  PorousMedia::steady_time_step_reduction_factor = 0.8;
  PorousMedia::steady_time_step_increase_factor = 1.6;
  PorousMedia::steady_time_step_increase_factor_2 = 10;
  PorousMedia::steady_time_step_retry_factor_1 = 0.2;
  PorousMedia::steady_time_step_retry_factor_2 = 0.01;
  PorousMedia::steady_time_step_retry_factor_f = 0.001;
  PorousMedia::steady_max_consecutive_failures_1 = 3;
  PorousMedia::steady_max_consecutive_failures_2 = 4;
  PorousMedia::steady_init_time_step = 1.e10;
  PorousMedia::steady_max_time_steps = 8000;
  PorousMedia::steady_max_time_step_size = 1.e20;
  PorousMedia::steady_max_psuedo_time = 1.e14;
  PorousMedia::steady_max_num_consecutive_success = 0;
  PorousMedia::steady_extra_time_step_increase_factor = 10.;
  PorousMedia::steady_max_num_consecutive_increases = 3;
  PorousMedia::steady_consecutive_increase_reduction_factor = 0.4;
  PorousMedia::steady_use_PETSc_snes = true;
  PorousMedia::steady_abort_on_psuedo_timestep_failure = false;
  PorousMedia::steady_limit_function_evals = 1e8;
  PorousMedia::steady_abs_tolerance = 1.e-10;
  PorousMedia::steady_rel_tolerance = 1.e-20;
  PorousMedia::steady_abs_update_tolerance = 1.e-12;
  PorousMedia::steady_rel_update_tolerance = -1;
  PorousMedia::steady_do_grid_sequence = 1;
  PorousMedia::steady_grid_sequence_new_level_dt_factor.resize(1,1);
  PorousMedia::steady_record_file.clear();

  PorousMedia::richard_max_ls_iterations = 10;
  PorousMedia::richard_min_ls_factor = 1.e-8;
  PorousMedia::richard_ls_acceptance_factor = 1.4;
  PorousMedia::richard_ls_reduction_factor = 0.1;
  PorousMedia::richard_monitor_linear_solve = 0;
  PorousMedia::richard_monitor_line_search = 0;
  PorousMedia::richard_perturbation_scale_for_J = 1.e-8;
  PorousMedia::richard_use_fd_jac = 1;
  PorousMedia::richard_use_dense_Jacobian = 0;
  PorousMedia::richard_upwind_krel = 1;
  PorousMedia::richard_pressure_maxorder = 3;
  PorousMedia::richard_scale_solution_before_solve = true;
  PorousMedia::richard_semi_analytic_J = false;
  PorousMedia::richard_centered_diff_J = true;
  PorousMedia::richard_variable_switch_saturation_threshold = -1;
  PorousMedia::richard_dt_thresh_pure_steady = -1;

  PorousMedia::echo_inputs    = 0;
  PorousMedia::richard_solver = 0;
  PorousMedia::richard_solver_control = 0;
  PorousMedia::richard_solver_data = 0;

  PorousMedia::chemistry_engine = 0;
}

std::pair<std::string,std::string>
SplitDirAndName(const std::string& orig)
{
    if (orig[orig.length()-1] == '/') {
        BoxLib::Abort(std::string("Invalid filename:" + orig).c_str());
    }
    vector<std::string> tokens = BoxLib::Tokenize(orig,std::string("/"));
    std::pair<std::string,std::string> result;
    int size = tokens.size();
    BL_ASSERT(tokens.size()>0);
    if (size>1) {
        for (int i=0; i<size-2; ++i) {
            result.first += tokens[i] + "/";
        }
        result.first += tokens[size-2];
    }
    else {
        result.first = ".";
    }
    result.second = tokens[size-1];
    return result;
}

void
PorousMedia::variableSetUp ()
{

  InitializeStaticVariables();
  ParmParse pproot;
  pproot.query("echo_inputs",echo_inputs);

  BL_ASSERT(desc_lst.size() == 0);

  for (int dir = 0; dir < BL_SPACEDIM; dir++)
  {
    phys_bc.setLo(dir,SlipWall);
    phys_bc.setHi(dir,SlipWall);
  }

  setup_list();
  std::string pp_dump_file = ""; 
  if (pproot.countval("dump_parmparse_table")) {
      pproot.get("dump_parmparse_table",pp_dump_file);
      std::ofstream ofs;
      std::pair<std::string,std::string> df = SplitDirAndName(pp_dump_file);
      if (ParallelDescriptor::IOProcessor()) {
          if (!BoxLib::UtilCreateDirectory(df.first, 0755)) {
              BoxLib::CreateDirectoryFailed(df.first);
          }

          ofs.open(pp_dump_file.c_str());
          if (ofs.fail()) {
              BoxLib::Abort(std::string("Cannot open pp dump file: "+pp_dump_file).c_str());
          }
          if (verbose>1)
          {
              std::cout << "\nDumping ParmParse table:\n";
          }

          // NOTE: Formatting useless since all data are strings at this point
          //
          // std::ios::fmtflags oflags = ofs.flags();
          // ofs.setf(std::ios::floatfield, std::ios::scientific);
          // int old_prec = ofs.precision(15);

          bool prettyPrint = false;
          ParmParse::dumpTable(ofs,prettyPrint);

          // ofs.flags(oflags);
          // ofs.precision(old_prec);

          if (!ofs.good())
              BoxLib::Error("Write of pp dump file failed");
          

          if (verbose>1)
          {
              std::cout << "... done dumping ParmParse table.\n" << '\n';
          }
          ofs.close();
      }
      ParallelDescriptor::Barrier();
  }

  read_params(); 
  BCRec bc;

  //
  // Set state variables Ids.
  //
  int num_gradn = ncomps;
  // NUM_SCALARS   = ncomps + 2; // Currently unused last 2 components
  NUM_SCALARS   = ncomps;

  if (ntracers > 0)
    NUM_SCALARS = NUM_SCALARS + ntracers;

  if (model == PM_POLYMER)
  {
    NUM_SCALARS = NUM_SCALARS + 2;
  }

  // add velocity and correction velocity
  NUM_STATE = NUM_SCALARS + BL_SPACEDIM + BL_SPACEDIM ;

  //
  // **************  DEFINE SCALAR VARIABLES  ********************
  //

  Array<BCRec>       bcs(ncomps);
  Array<std::string> names(ncomps);

  desc_lst.addDescriptor(State_Type,IndexType::TheCellType(),
			 StateDescriptor::Point,1,NUM_SCALARS,
			 &cell_cons_interp);

  set_scalar_bc(bc,phys_bc);
  for (int i = 0; i < ncomps; i++) 
  {
    bcs[i] = bc;
    names[i] = cNames[i];
  }

  desc_lst.setComponent(State_Type,
			0,
			names,
			bcs,
			BndryFunc(FORT_ONE_N_FILL,FORT_ALL_N_FILL));

  if (ntracers > 0)
  {
    Array<BCRec>       tbcs(ntracers);
    Array<std::string> tnames(ntracers);

    for (int i = 0; i < ntracers; i++) 
    {
      tbcs[i]   = trac_bc;
      tnames[i] = tNames[i];
    }

    desc_lst.setComponent(State_Type,
			  ncomps,
			  tnames,
			  tbcs,
			  BndryFunc(FORT_ONE_N_FILL,FORT_ALL_T_FILL));
  }

#if 0
  // Currently unused
  desc_lst.setComponent(State_Type,ncomps+ntracers,"Aux1",
			bc,BndryFunc(FORT_ENTHFILL));
  desc_lst.setComponent(State_Type,ncomps+ntracers+1,"Aux2",
			bc,BndryFunc(FORT_ADVFILL));
#endif

  if (model == PM_POLYMER) {
    desc_lst.setComponent(State_Type,ncomps+2,"s",
			  bc,BndryFunc(FORT_ONE_N_FILL));
    desc_lst.setComponent(State_Type,ncomps+3,"c",
			  bc,BndryFunc(FORT_ONE_N_FILL));
  }

  is_diffusive.resize(NUM_SCALARS,false);
  advectionType.resize(NUM_SCALARS,Conservative);
  diffusionType.resize(NUM_SCALARS,Laplacian_S);

  // For components.
  for (int i=0; i<ncomps; i++) 
    {
      advectionType[i] = Conservative;
      diffusionType[i] = Laplacian_S;
      is_diffusive[i] = false;
      if (visc_coef[i] > 0.0 && solid.compare(pNames[pType[i]])!=0)
	is_diffusive[i] = true;
    }

  // For tracers
  for (int i=0; i<ntracers; i++) 
    {
      advectionType[ncomps+i] = NonConservative;
      diffusionType[ncomps+i] = Laplacian_S;
      is_diffusive[ncomps+i] = false;
      if (diffuse_tracers)
	is_diffusive[ncomps+i] = true;
    }

  for (int i = ncomps+ntracers; i < NUM_SCALARS; i++)
    {
      advectionType[i] = NonConservative;
      diffusionType[i] = Laplacian_S;
      is_diffusive[i] = false;
    }

  if (do_tracer_chemistry && ntracers > 0)
  {
      // NOTE: aux_chem_variables is setup by RockManager and read_chem as data is
      // parsed in.  By the time we get here, we have figured out all the variables
      // for which we need to make space.

      int num_aux_chem_variables = aux_chem_variables.size();
      Array<BCRec> cbcs(num_aux_chem_variables);
      Array<std::string> tmp_aux(num_aux_chem_variables);
      for (std::map<std::string,int>::iterator it=aux_chem_variables.begin(); 
	   it!=aux_chem_variables.end(); ++it)
      {
	int i = it->second;
	tmp_aux[i] = it->first;
	cbcs[i] = bc;
	  
      }

      FORT_AUXPARAMS(&num_aux_chem_variables);

      desc_lst.addDescriptor(Aux_Chem_Type,IndexType::TheCellType(),
                             StateDescriptor::Point,0,num_aux_chem_variables,
                             &cell_cons_interp);
      desc_lst.setComponent(Aux_Chem_Type,0,tmp_aux,cbcs,
                            BndryFunc(FORT_ONE_A_FILL,FORT_ALL_A_FILL));

  }

  //
  // **************  DEFINE PRESSURE VARIABLE  ********************
  //

  desc_lst.addDescriptor(Press_Type,IndexType::TheCellType(),
			 StateDescriptor::Point,1,1,
			 &cell_cons_interp);
  set_pressure_bc(bc,pres_bc);
  desc_lst.setComponent(Press_Type,Pressure,"pressure",
			bc,BndryFunc(FORT_PRESFILL));

  //
  // **************  DEFINE VELOCITY VARIABLES  ********************

  desc_lst.addDescriptor(Vcr_Type,IndexType::TheCellType(),
			 StateDescriptor::Point,1,BL_SPACEDIM,
			 &cell_cons_interp);
  set_x_vel_bc(bc,phys_bc);
  desc_lst.setComponent(Vcr_Type,Xvcr,"x_vcorr",
			bc,BndryFunc(FORT_XVELFILL));
  set_y_vel_bc(bc,phys_bc);
  desc_lst.setComponent(Vcr_Type,Yvcr,"y_vcorr",
			bc,BndryFunc(FORT_YVELFILL));
#if (BL_SPACEDIM == 3)
  set_z_vel_bc(bc,phys_bc);
  desc_lst.setComponent(Vcr_Type,Zvcr,"z_vcorr",
			bc,BndryFunc(FORT_ZVELFILL));
#endif

#if defined(AMANZI)
  if (do_tracer_chemistry>0)
    {
      // add function count
      int nfunccountghost = 0;
      if (do_full_strang) nfunccountghost=1;
      desc_lst.addDescriptor(FuncCount_Type, IndexType::TheCellType(),
			     StateDescriptor::Point,nfunccountghost,1, &cell_cons_interp);
      desc_lst.setComponent(FuncCount_Type, 0, "FuncCount", 
			    bc, BndryFunc(FORT_ONE_A_FILL));
    }
#endif

  // "User defined" - although these must correspond to those in PorousMedia::derive
  IndexType regionIDtype(IndexType::TheCellType());
  int nCompRegion = 1;
  std::string amr_prefix = "amr";
  ParmParse pp(amr_prefix);
  int num_user_derives = pp.countval("user_derive_list");
  Array<std::string> user_derive_list(num_user_derives);
  pp.getarr("user_derive_list",user_derive_list,0,num_user_derives);
  for (int i=0; i<num_user_derives; ++i) {
    int nCompThis = (user_derive_list[i] == "Dispersivity" ? 2 : 1);
    derive_lst.add(user_derive_list[i], regionIDtype, nCompThis);
  }

  //
  // **************  DEFINE ERROR ESTIMATION QUANTITIES  *************
  //
  const RegionManager* region_manager = PMAmr::RegionManagerPtr();
  Array<std::string> refinement_indicators;
  pp.queryarr("refinement_indicators",refinement_indicators,0,pp.countval("refinement_indicators"));
  for (int i=0; i<refinement_indicators.size(); ++i)
  {
      std::string ref_prefix = amr_prefix + "." + refinement_indicators[i];
      ParmParse ppr(ref_prefix);
      Real min_time = 0; ppr.query("start_time",min_time);
      Real max_time = -1; ppr.query("end_time",max_time);
      int max_level = -1;  ppr.query("max_level",max_level);
      Array<std::string> region_names(1,"All"); 
      int nreg = ppr.countval("regions");
      if (nreg) {
          ppr.getarr("regions",region_names,0,nreg);
      }
      Array<const Region*> regions = region_manager->RegionPtrArray(region_names);
      if (ppr.countval("val_greater_than")) {
          Real value; ppr.get("val_greater_than",value);
          std::string field; ppr.get("field",field);
          err_list.add(field.c_str(),0,ErrorRec::Special,
                       PM_Error_Value(FORT_VALGTERROR,value,min_time,max_time,max_level,regions));
      }
      else if (ppr.countval("val_less_than")) {
          Real value; ppr.get("val_less_than",value);
          std::string field; ppr.get("field",field);
          err_list.add(field.c_str(),0,ErrorRec::Special,
                       PM_Error_Value(FORT_VALLTERROR,value,min_time,max_time,max_level,regions));
      }
      else if (ppr.countval("diff_greater_than")) {
          Real value; ppr.get("diff_greater_than",value);
          std::string field; ppr.get("field",field);
          err_list.add(field.c_str(),1,ErrorRec::Special,
                       PM_Error_Value(FORT_DIFFGTERROR,value,min_time,max_time,max_level,regions));
      }
      else if (ppr.countval("in_region")) {
          Real value; ppr.get("in_region",value);
          err_list.add("PMAMR_DUMMY",1,ErrorRec::Special,
                       PM_Error_Value(min_time,max_time,max_level,regions));
      }
      else {
          BoxLib::Abort(std::string("Unrecognized refinement indicator for " + refinement_indicators[i]).c_str());
      }
  }

  num_state_type = desc_lst.size();

  BoxLib::ExecOnFinalize(PM_Setup_CleanUpStatics);

}

void PorousMedia::read_prob()
{
  ParmParse pp;

  std::string exec_mode_in_str; 
  pp.get("execution_mode",exec_mode_in_str);
  if (exec_mode_in_str == "transient") {
    execution_mode = TRANSIENT;
  } else if (exec_mode_in_str == "init_to_steady") {
    execution_mode = INIT_TO_STEADY;
  } else if (exec_mode_in_str == "steady") {
    execution_mode = STEADY;
  } else {
    ParallelDescriptor::Barrier();
    if (ParallelDescriptor::IOProcessor()) {
      std::string str = "Unrecognized execution_mode: \"" + exec_mode_in_str + "\"";
      BoxLib::Abort(str.c_str());
    }
  }
  if (execution_mode==INIT_TO_STEADY) {
    pp.get("switch_time",switch_time);
  }

  pp.query("do_output_flow_time_in_years;",do_output_flow_time_in_years);
  pp.query("do_output_transport_time_in_years;",do_output_transport_time_in_years);
  pp.query("do_output_chemistry_time_in_years;",do_output_chemistry_time_in_years);

  // determine the model based on model_name
  ParmParse pb("prob");
  std::string model_name;
  pb.query("model_name",model_name);
  model = available_models[model_name].model;
  if (model == PM_INVALID) {
    BoxLib::Abort("Invalid model selected");
  }

  if (model_name=="steady-saturated") {
      solute_transport_limits_dt = true;
      do_multilevel_full = true;
      do_richard_init_to_steady = true;
      use_PETSc_snes_for_evolution = true;
  }

  pb.query("do_tracer_advection",do_tracer_advection);
  pb.query("do_tracer_diffusion",do_tracer_diffusion);
  if (do_tracer_advection || do_tracer_diffusion) {
      setup_tracer_transport = true; // NOTE: May want these data structures regardless...
  }

  if (setup_tracer_transport && 
      ( model==PM_STEADY_SATURATED
	|| model == PM_SATURATED
        || (execution_mode==INIT_TO_STEADY && switch_time<=0)
        || (execution_mode!=INIT_TO_STEADY) ) ) {
      advect_tracers = do_tracer_advection;
      diffuse_tracers = do_tracer_diffusion;
      react_tracers = do_tracer_chemistry;
  }
    
  // Verbosity
  pb.query("v",verbose);
  pb.query("richard_solver_verbose",richard_solver_verbose);
  pb.query("do_richard_sat_solve",do_richard_sat_solve);

  // Get timestepping parameters.  Some will be used to default values for int-to-steady solver
  pb.get("cfl",cfl);
  pb.query("init_shrink",init_shrink);
  pb.query("dt_init",dt_init);
  pb.query("dt_cutoff",dt_cutoff);
  pb.query("dt_grow_max",dt_grow_max);
  pb.query("dt_shrink_max",dt_shrink_max);
  pb.query("fixed_dt",fixed_dt);
  pb.query("steady_richard_max_dt",steady_richard_max_dt);
  pb.query("transient_richard_max_dt",transient_richard_max_dt);
  pb.query("sum_interval",sum_interval);
  pb.query("max_n_subcycle_transport",max_n_subcycle_transport);

  pb.query("max_dt_iters_flow",max_dt_iters_flow);
  pb.query("verbose_chemistry",verbose_chemistry);
  pb.query("show_selected_runtimes",show_selected_runtimes);
  pb.query("abort_on_chem_fail",abort_on_chem_fail);

  pb.query("richard_init_to_steady_verbose",richard_init_to_steady_verbose);
  pb.query("do_richard_init_to_steady",do_richard_init_to_steady);
  pb.query("steady_record_file",steady_record_file);
  pb.query("steady_min_iterations",steady_min_iterations);
  pb.query("steady_min_iterations_2",steady_min_iterations_2);
  pb.query("steady_max_iterations",steady_max_iterations);
  pb.query("steady_limit_iterations",steady_limit_iterations);
  pb.query("steady_time_step_reduction_factor",steady_time_step_reduction_factor);
  pb.query("steady_time_step_increase_factor_2",steady_time_step_increase_factor_2);
  pb.query("steady_time_step_increase_factor",steady_time_step_increase_factor);
  pb.query("steady_time_step_retry_factor_1",steady_time_step_retry_factor_1);
  pb.query("steady_time_step_retry_factor_2",steady_time_step_retry_factor_2);
  pb.query("steady_time_step_retry_factor_f",steady_time_step_retry_factor_f);
  pb.query("steady_max_consecutive_failures_1",steady_max_consecutive_failures_1);
  pb.query("steady_max_consecutive_failures_2",steady_max_consecutive_failures_2);
  pb.query("steady_init_time_step",steady_init_time_step);
  pb.query("steady_max_time_steps",steady_max_time_steps);
  pb.query("steady_max_time_step_size",steady_max_time_step_size);
  pb.query("steady_max_psuedo_time",steady_max_psuedo_time);
  pb.query("steady_max_num_consecutive_success",steady_max_num_consecutive_success);
  pb.query("steady_extra_time_step_increase_factor",steady_extra_time_step_increase_factor);
  pb.query("steady_max_num_consecutive_increases",steady_max_num_consecutive_increases);
  pb.query("steady_consecutive_increase_reduction_factor",steady_consecutive_increase_reduction_factor);
  pb.query("steady_use_PETSc_snes",steady_use_PETSc_snes);
  pb.query("steady_abort_on_psuedo_timestep_failure",steady_abort_on_psuedo_timestep_failure);
  pb.query("steady_limit_function_evals",steady_limit_function_evals);
  pb.query("steady_abs_tolerance",steady_abs_tolerance);
  pb.query("steady_rel_tolerance",steady_rel_tolerance);
  pb.query("steady_abs_update_tolerance",steady_abs_update_tolerance);
  pb.query("steady_rel_update_tolerance",steady_rel_update_tolerance);
  pb.query("steady_do_grid_sequence",steady_do_grid_sequence);
  int ndt = pb.countval("steady_grid_sequence_new_level_dt_factor");
  if (ndt > 0) {
      pb.getarr("steady_grid_sequence_new_level_dt_factor",steady_grid_sequence_new_level_dt_factor,0,ndt);
  }

  pb.query("richard_max_ls_iterations",richard_max_ls_iterations);
  pb.query("richard_min_ls_factor",richard_min_ls_factor);
  pb.query("richard_ls_acceptance_factor",richard_ls_acceptance_factor);
  pb.query("richard_ls_reduction_factor",richard_ls_reduction_factor);
  pb.query("richard_monitor_linear_solve",richard_monitor_linear_solve);
  pb.query("richard_monitor_line_search",richard_monitor_line_search);
  pb.query("richard_perturbation_scale_for_J",richard_perturbation_scale_for_J);
  pb.query("richard_use_fd_jac",richard_use_fd_jac);
  pb.query("richard_use_dense_Jacobian",richard_use_dense_Jacobian);
  pb.query("richard_upwind_krel",richard_upwind_krel);
  pb.query("richard_pressure_maxorder",richard_pressure_maxorder);
  pb.query("richard_scale_solution_before_solve",richard_scale_solution_before_solve);
  pb.query("richard_semi_analytic_J",richard_semi_analytic_J);
  pb.query("richard_centered_diff_J",richard_centered_diff_J);
  pb.query("richard_variable_switch_saturation_threshold",richard_variable_switch_saturation_threshold);
  richard_dt_thresh_pure_steady = 0.99*steady_init_time_step;
  pb.query("richard_dt_thresh_pure_steady",richard_dt_thresh_pure_steady);

  // Gravity are specified as m/s^2 in the input file
  // This is converted to the unit that is used in the code.
  if (pb.contains("gravity")) {
    pb.get("gravity",gravity);
    gravity /= BL_ONEATM;
  }
  pb.query("gravity_dir",gravity_dir);
  BL_ASSERT(gravity_dir>=0 && gravity_dir<3); // Note: can set this to 2 for a 2D problem
  if (BL_SPACEDIM<3 && gravity_dir>BL_SPACEDIM-1) {
    pb.query("z_location",z_location);
  }
  if (pb.countval("atmospheric_pressure_atm")) {
    pp.get("atmospheric_pressure_atm",atmospheric_pressure_atm);
    atmospheric_pressure_atm *= 1 / BL_ONEATM;
  }

  // Get algorithmic flags and options
  pb.query("full_cycle", full_cycle);
  //pb.query("algorithm", algorithm);
  pb.query("do_multilevel_full",  do_multilevel_full );
  use_PETSc_snes_for_evolution = do_multilevel_full;
  pb.query("use_PETSc_snes_for_evolution", use_PETSc_snes_for_evolution);
  pb.query("do_simple",  do_simple );
  pb.query("do_reflux",  do_reflux );
  pb.query("do_correct", do_correct);
  pb.query("do_cpl_advect", do_cpl_advect);
  pb.query("no_corrector",no_corrector);
  pb.query("do_kappa_refine",do_kappa_refine);
  pb.query("n_pressure_interval",n_pressure_interval);

  // Get solver tolerances
  pb.query("visc_tol",visc_tol);
  pb.query("visc_abs_tol",visc_abs_tol);
  pb.query("be_cn_theta",be_cn_theta);
  if (be_cn_theta > 1.0 || be_cn_theta < .5)
    BoxLib::Abort("PorousMedia::Must have be_cn_theta <= 1.0 && >= .5");   
  pb.query("be_cn_theta_trac",be_cn_theta_trac);
  if (be_cn_theta > 1.0 || be_cn_theta < 0)
    BoxLib::Abort("PorousMedia::Must have be_cn_theta_trac <= 1.0 && >= 0");   
  pb.query("harm_avg_cen2edge", def_harm_avg_cen2edge);

  // if capillary pressure flag is true, then we make sure 
  // the problem can handle capillary pressure.
  pb.query("have_capillary",have_capillary);
  if (have_capillary == 1) 
    {
      if (nphases != 2 && ncomps !=nphases) 
	{
	  if (ParallelDescriptor::IOProcessor())
	    {
	      std::cerr << "PorousMedia::read_prob: nphases != 2 && ncomps !=nphases "
			<< "although have_capillary == 1.\n ";
	      BoxLib::Abort("PorousMedia::read_prob()");
	    }
	}
    }
}

//
// Construct bc functions
//

const Material&
PorousMedia::find_material(const std::string& name)
{
    bool found=false;
    int iMat = -1;
    for (int i=0; i<materials.size() && !found; ++i)
    {
        const Material& mat = materials[i];
        if (name == mat.Name()) {
            found = true;
            iMat = i;
        }
    } 
    if (iMat < 0) {
        std::string m = "Named material not found " + name;
        BoxLib::Abort(m.c_str());
    }
    return materials[iMat];
}


struct PressToRhoSat
    : public ArrayTransform
{
    PressToRhoSat() {}
    virtual ArrayTransform* clone() const {return new PressToRhoSat(*this);}
    virtual Array<Real> transform(Real inData) const;
protected:
};

Array<Real>
PressToRhoSat::transform(Real aqueous_pressure) const
{
    // FIXME: Requires Water
    const Array<std::string>& cNames = PorousMedia::componentNames();
    const Array<Real>& density = PorousMedia::Density();

    int ncomps = cNames.size();
    int idx = -1;
    for (int j=0; j<ncomps; ++j) {
        if (cNames[j] == "Water") {
            idx = j;
        }
    }
    BL_ASSERT(idx>=0);

    Array<double> rhoSat(ncomps,0);
    rhoSat[idx] = density[idx] * 1; // Fully saturated...an assumption
    return rhoSat;
}

void  PorousMedia::read_comp()
{
  //
  // Read in parameters for phases and components
  //
  ParmParse pp("phase");

  // Get number and names of phases
  nphases = pp.countval("phases");
  pp.getarr("phases",pNames,0,nphases);
  for (int i = 0; i<nphases; i++) phase_list[pNames[i]] = i;

  // Build flattened list of components
  ndiff = 0;
  for (int i = 0; i<nphases; i++) {
      const std::string prefix("phase." + pNames[i]);
      ParmParse ppr(prefix.c_str());
      int p_nc = ppr.countval("comps");
      BL_ASSERT(p_nc==1); // An assumption all over the place...
      ncomps += p_nc;
      Array<std::string> p_cNames; ppr.getarr("comps",p_cNames,0,p_nc);
      for (int j=0; j<p_cNames.size(); ++j) {
          cNames.push_back(p_cNames[j]);
      }
      Real p_rho; ppr.get("density",p_rho); density.push_back(p_rho);
      // viscosity in units of kg/(m.s)
      //Real p_visc; ppr.get("viscosity",p_visc); muval.push_back(p_visc);
      Real p_visc; ppr.get("viscosity",p_visc); muval.push_back(p_visc);
      Real p_diff; ppr.get("diffusivity",p_diff); visc_coef.push_back(p_diff);

      // Tracer diffusion handled during tracer read
      if (visc_coef.back() > 0)
      {
	  do_any_diffuse = true;
	  is_diffusive[visc_coef.size()-1] = 1;
      }
      else {
          variable_scal_diff = 0;
      }
      ++ndiff;

      pType.push_back(phase_list[pNames[i]]);
  }

  ParmParse cp("comp");
  for (int i = 0; i<ncomps; i++) comp_list[cNames[i]] = i;
#if 0

  // Get the dominant component
  std::string domName;
  cp.query("dominant",domName);
  if (!domName.empty())
    idx_dominant = comp_list[domName];

  // Get the boundary conditions for the components
  Array<int> lo_bc(BL_SPACEDIM), hi_bc(BL_SPACEDIM);
  cp.getarr("lo_bc",lo_bc,0,BL_SPACEDIM);
  cp.getarr("hi_bc",hi_bc,0,BL_SPACEDIM);
  for (int i = 0; i < BL_SPACEDIM; i++)
    {
      phys_bc.setLo(i,lo_bc[i]);
      phys_bc.setHi(i,hi_bc[i]);
    }

  // Check phys_bc against possible periodic geometry: 
  //  if periodic, that boundary must be internal BC.
  if (Geometry::isAnyPeriodic())
    {      
      // Do idiot check.  Periodic means interior in those directions.
      for (int dir = 0; dir < BL_SPACEDIM; dir++)
        {
	  if (Geometry::isPeriodic(dir))
	    {
	      if (lo_bc[dir] != Interior)
		{
		  std::cerr << "PorousMedia::variableSetUp:periodic in direction "
			    << dir
			    << " but low BC is not Interior\n";
		  BoxLib::Abort("PorousMedia::read_comp()");
		}
	      if (hi_bc[dir] != Interior)
		{
		  std::cerr << "PorousMedia::variableSetUp:periodic in direction "
			    << dir
			    << " but high BC is not Interior\n";
		  BoxLib::Abort("PorousMedia::read_comp()");
		}
	    } 
        }
    }
  else
    {
      
      // Do idiot check.  If not periodic, should be no interior.
      for (int dir = 0; dir < BL_SPACEDIM; dir++)
        {
	  if (!Geometry::isPeriodic(dir))
	    {
	      if (lo_bc[dir] == Interior)
		{
		  std::cerr << "PorousMedia::variableSetUp:Interior bc in direction "
			    << dir
			    << " but not defined as periodic\n";
		  BoxLib::Abort("PorousMedia::read_comp()");
		}
	      if (hi_bc[dir] == Interior)
		{
		  std::cerr << "PorousMedia::variableSetUp:Interior bc in direction "
			    << dir
			    << " but not defined as periodic\n";
		  BoxLib::Abort("PorousMedia::read_comp()");
		}
	    }
        }
    }
#endif

  // Initial condition and boundary condition
  //
  // Component ics, bcs will be set all at once
  int n_ics = cp.countval("ic_labels");
  if (n_ics > 0)
  {
      Array<std::string> ic_names;
      cp.getarr("ic_labels",ic_names,0,n_ics);
      ic_array.resize(n_ics,PArrayManage);
      do_constant_vel = false;
      const RegionManager* region_manager = PMAmr::RegionManagerPtr();
      for (int i = 0; i<n_ics; i++)
      {
          const std::string& icname = ic_names[i];
	  const std::string prefix("comp.ics." + icname);
	  ParmParse ppr(prefix.c_str());
          
	  int n_ic_regions = ppr.countval("regions");
          Array<std::string> region_names;
	  ppr.getarr("regions",region_names,0,n_ic_regions);
	  Array<const Region*> ic_regions = region_manager->RegionPtrArray(region_names);

          std::string ic_type; ppr.get("type",ic_type);
	  BL_ASSERT(!do_constant_vel); // If this is ever set, it must be the only IC so we should never see this true here
          if (ic_type == "pressure")
          {
              int nPhase = pNames.size();
              Array<Real> vals(nPhase);
              
              int num_phases_reqd = nPhase;
              std::map<std::string,bool> phases_set;
              for (int j = 0; j<pNames.size(); j++)
              {
                  std::string val_name = "val";
                  ppr.get(val_name.c_str(),vals[0]);
                  phases_set[pNames[j]] = true;
              }
	      
	      // convert to atm
	      for (int j=0; j<vals.size(); ++j) {
		vals[j] = vals[j] / BL_ONEATM;
	      }
      
              int num_phases = phases_set.size();
              if (num_phases != num_phases_reqd) {
                  std::cerr << icname << ": Insufficient number of phases specified" << std::endl;
                  std::cerr << " ngiven, nreqd: " << num_phases << ", " << num_phases_reqd << std::endl;
                  std::cerr << " current model: " << model << std::endl;
                  BoxLib::Abort();
              }
              
              Array<Real> times(1,0);
              Array<std::string> forms(0);
              ic_array.set(i, new ArrayRegionData(icname,times,vals,forms,ic_regions,ic_type,1));
          }
          else if (ic_type == "linear_pressure")
          {
              int nPhase = pNames.size();
              if (nPhase!=1) {
                std::cerr << "Multiphase not currently surrported" << std::endl;
                BoxLib::Abort();
              }

              Real press_val;
              std::string val_name = "val";
              ppr.get(val_name.c_str(),press_val);
              press_val = press_val / BL_ONEATM;

              int ngrad = ppr.countval("grad");
              if (ngrad<BL_SPACEDIM) {
                std::cerr << "Insufficient number of components given for pressure gradient" << std::endl;
                BoxLib::Abort();
              }
              Array<Real> pgrad(BL_SPACEDIM);
              ppr.getarr("grad",pgrad,0,ngrad);
	      for (int j=0; j<pgrad.size(); ++j) {
                pgrad[j] = pgrad[j] / BL_ONEATM;
	      }

              int nref = ppr.countval("ref_coord");
              if (nref<BL_SPACEDIM) {
		if (ParallelDescriptor::IOProcessor()) {
		  std::cerr << "Insufficient number of components given for pressure refernce location" << std::endl;
		}
                BoxLib::Abort();
              }
              Array<Real> pref(BL_SPACEDIM);
              ppr.getarr("ref_coord",pref,0,nref);

              int ntmp = 2*BL_SPACEDIM+1;
              Array<Real> tmp(ntmp);
              tmp[0] = press_val;
              for (int j=0; j<BL_SPACEDIM; ++j) {
                tmp[1+j] = pgrad[j];
                tmp[1+j+BL_SPACEDIM] = pref[j];
              }
              ic_array.set(i, new RegionData(icname,ic_regions,ic_type,tmp));
          }
          else if (ic_type == "saturation")
          {
              Array<Real> vals(ncomps);
              for (int j = 0; j<cNames.size(); j++) {
                  ppr.get(cNames[j].c_str(),vals[j]);
                  vals[j] *= density[j];
              }
              std::string generic_type = "scalar";
              ic_array.set(i, new RegionData(icname,ic_regions,generic_type,vals));
          }
          else if (ic_type == "constant_velocity")
          {
	      if (model != PM_STEADY_SATURATED) {
	          if (ParallelDescriptor::IOProcessor()) {
		    std::cerr << "constant-velocity settings may only be used with steady-saturated flow" << std::endl;
		    BoxLib::Abort();
		  }
	      }
              Array<Real> vals(BL_SPACEDIM);
	      ppr.getarr("Velocity_Vector",vals,0,BL_SPACEDIM);
              std::string generic_type = "constant_velocity";
	      do_constant_vel = true;
              ic_array.set(i, new RegionData(icname,ic_regions,generic_type,vals));
          }
          else if (ic_type == "hydrostatic")
          {
              Array<Real> water_table_height(1); ppr.get("water_table_height",water_table_height[0]);
              Array<Real> times(1,0);
              Array<std::string> forms;
              ic_array.set(i, new ArrayRegionData(icname,times,water_table_height,
                                                  forms,ic_regions,ic_type,1));
          }
          else if (ic_type == "zero_total_velocity")
          {
	      Array<Real> vals(4);
              Array<Real> times(1,0);
              Array<std::string> forms;
	      ppr.get("aqueous_vol_flux",vals[0]);
              ppr.get("water_table_height",vals[1]);
              Real aqueous_ref_pres = 0; ppr.query("val",vals[2]);
              Real aqueous_pres_grad = 0; ppr.query("grad",vals[3]);
	      ic_array.set(i,new RegionData(icname,ic_regions,ic_type,vals));
          }
          else {
              BoxLib::Abort("Unsupported comp ic");
          }
      }
  }

  int n_bcs = cp.countval("bc_labels");
  if (n_bcs > 0)
  {
      rinflow_bc_lo.resize(BL_SPACEDIM,0); 
      rinflow_bc_hi.resize(BL_SPACEDIM,0); 
      inflow_bc_lo.resize(BL_SPACEDIM,0); 
      inflow_bc_hi.resize(BL_SPACEDIM,0); 

      bc_array.resize(n_bcs,PArrayManage);
      Array<std::string> bc_names;
      cp.getarr("bc_labels",bc_names,0,n_bcs);

      // default to no flow first.
      for (int j=0;j<BL_SPACEDIM;j++) {
	phys_bc.setLo(j,1);
	pres_bc.setLo(j,1);
	phys_bc.setHi(j,1);
	pres_bc.setHi(j,1);
      }

      const RegionManager* region_manager = PMAmr::RegionManagerPtr();
      for (int i = 0; i<n_bcs; i++)
      {
          int ibc = i;
          const std::string& bcname = bc_names[i];
	  const std::string prefix("comp.bcs." + bcname);
	  ParmParse ppr(prefix.c_str());
          
	  int n_bc_regions = ppr.countval("regions");
          Array<std::string> region_names;
	  ppr.getarr("regions",region_names,0,n_bc_regions);
	  Array<const Region*> bc_regions = region_manager->RegionPtrArray(region_names);
          std::string bc_type; ppr.get("type",bc_type);

          bool is_inflow = false;
          int component_bc = 1;
	  int pressure_bc  = 1;

          use_gauge_pressure[bcname] = false; // Default value

          if (bc_type == "pressure")
          {
              int nPhase = pNames.size();
              BL_ASSERT(nPhase==1); // FIXME
              Array<Real> vals, times;
              Array<std::string> forms;
              
              std::string val_name = "vals";
              int nv = ppr.countval(val_name.c_str());
              if (nv) {
                  ppr.getarr(val_name.c_str(),vals,0,nv);
                  times.resize(nv,0);
                  if (nv>1) {
                      ppr.getarr("times",times,0,nv);
                      ppr.getarr("forms",forms,0,nv-1);
                  }
              }
              
              // convert to atm
              for (int j=0; j<vals.size(); ++j) {
                vals[j] = vals[j] / BL_ONEATM;
              }
              
              is_inflow = false;
              if (model == PM_STEADY_SATURATED || model == PM_SATURATED) {
                component_bc = 2;
              } else {
                component_bc = 1;
              }
              pressure_bc = 2;

              if (model == PM_STEADY_SATURATED 
		  || model == PM_SATURATED 
		  || (model == PM_RICHARDS && !do_richard_sat_solve)) {
                bc_array.set(ibc, new RegionData(bcname,bc_regions,bc_type,vals));
              } else {
                PressToRhoSat p_to_sat;
                bc_array.set(ibc, new Transform_S_AR_For_BC(bcname,times,vals,forms,bc_regions,
                                                            bc_type,ncomps,p_to_sat));
              }
          }
          else if (bc_type == "pressure_head")
          {              
            Array<Real> vals;
            std::string val_name = "vals";
            int nv = ppr.countval(val_name.c_str());
            if (nv) {
              ppr.getarr(val_name.c_str(),vals,0,nv);
            }

            if (pp.countval("normalization")>0) {
              std::string norm_str; pp.get("normalization",norm_str);
              if (norm_str == "Absolute") {
                use_gauge_pressure[bcname] = false;
              } else if (norm_str == "Relative") {
                use_gauge_pressure[bcname] = true;
              } else {
                BoxLib::Abort("pressure_head BC normalization must be \"Absolute\" or \"Relative\"");
              }
            }

            is_inflow = false;
            if (model == PM_STEADY_SATURATED
		|| model == PM_SATURATED ) {
              component_bc = 2;
            } else {
              component_bc = 1;
            }
            pressure_bc = 2;

	    Array<Real> times(1,0);
	    Array<std::string> forms(0);
	    bc_array.set(ibc,new ArrayRegionData(bcname,times,vals,forms,bc_regions,bc_type,vals.size()));
          }
          else if (bc_type == "linear_pressure")
          {
	    Real val; ppr.get("val",val);
            int ng = ppr.countval("grad");
	    BL_ASSERT(ng>=BL_SPACEDIM);
	    Array<Real> grad(BL_SPACEDIM); ppr.getarr("grad",grad,0,BL_SPACEDIM);

            int nl = ppr.countval("loc");
	    BL_ASSERT(nl>=BL_SPACEDIM);
	    Array<Real> loc(BL_SPACEDIM); ppr.getarr("loc",loc,0,BL_SPACEDIM);
	    
            Array<Real> vals(2*BL_SPACEDIM+1);
	    vals[0] = val / BL_ONEATM;
	    for (int d=0; d<BL_SPACEDIM; ++d) {
	      vals[1+d] = grad[d] / BL_ONEATM;
	      vals[1+d+BL_SPACEDIM] = loc[d];
	    }

            is_inflow = false;
            if (model == PM_STEADY_SATURATED || model == PM_SATURATED) {
              component_bc = 2;
            } else {
              component_bc = 1;
            }
            pressure_bc = 2;

	    Array<Array<Real> > values(vals.size(),Array<Real>(1,0));
            for (int j=0; j<vals.size(); ++j) {
              values[j][0] = vals[j];
            }
	    Array<Array<Real> > times(vals.size(),Array<Real>(1,0));
	    Array<Array<std::string> > forms(vals.size(),Array<std::string>(0));
	    bc_array.set(ibc,new ArrayRegionData(bcname,times,values,forms,bc_regions,bc_type));
          }
          else if (bc_type == "zero_total_velocity")
          {
              Array<Real> vals, times;
              Array<std::string> forms;

              int nv = ppr.countval("aqueous_vol_flux");
              if (nv) {
                  ppr.getarr("aqueous_vol_flux",vals,0,nv); // "inward" flux
                  times.resize(nv,0);
                  if (nv>1) {
                      ppr.getarr("inflowtimes",times,0,nv);
                      ppr.getarr("inflowfncs",forms,0,nv-1);
                  }
              }
              else {
                  vals.resize(1,0);
                  times.resize(1,0);
                  forms.resize(0);
              }        

              // Work out sign of flux for this boundary
              int is_hi = -1;
              for (int j=0; j<bc_regions.size(); ++j)
              {
                  const std::string purpose = bc_regions[j]->purpose;
                  for (int k=0; k<7; ++k) {
                      if (purpose == PMAMR::RpurposeDEF[k]) {
			  if (k == 6) {
			    BoxLib::Abort(std::string("BC \""+bcname+"\" must be applied on a face region").c_str());
			  }
                          bool this_is_hi = (k>3);
                          if (is_hi < 0) {
                              is_hi = this_is_hi;
                          }
                          else {
                              if (this_is_hi != is_hi) {
                                  BoxLib::Abort("BC must apply to a single face only");
                              }
                          }
                      }
                  }
              }
              if (is_hi) {
                  for (int k=0; k<vals.size(); ++k) {
                      vals[k] = -vals[k];
                  }
              }

              is_inflow = true;
              component_bc = 1;
              pressure_bc = 1;
	      bc_array.set(ibc,new ArrayRegionData(bcname,times,vals,forms,bc_regions,bc_type,1));
          }
          else if (bc_type == "noflow")
          {
            Array<Real> vals(1,0), times(1,0);
            Array<std::string> forms(0);
            is_inflow = true;
            component_bc = 1;
            pressure_bc = 1;
            bc_array.set(ibc,new ArrayRegionData(bcname,times,vals,forms,bc_regions,bc_type,1));
          }
          else
          {
	    std::cout << bc_type << " not a valid bc_type " << std::endl;
	    BoxLib::Abort();
          }

          // Some clean up 
          std::set<std::string> o_set;

          for (int j=0; j<bc_regions.size(); ++j)
          {
              const std::string purpose = bc_regions[j]->purpose;
              int dir = -1, is_hi;
              for (int k=0; k<7; ++k) {
                  if (purpose == PMAMR::RpurposeDEF[k]) {
                      BL_ASSERT(k != 6);
                      dir = k%3;
                      is_hi = k>=3;
                  }
              }
              if (dir<0 || dir > BL_SPACEDIM) {
                  std::cout << "Bad region for boundary: \n" << bc_regions[j] << std::endl;
                  BoxLib::Abort();
              }

              if (o_set.find(purpose) == o_set.end())
              {
                  o_set.insert(purpose);

                  if (is_hi) {
                      rinflow_bc_hi[dir] = (is_inflow ? 1 : 0);
                      phys_bc.setHi(dir,component_bc);
                      pres_bc.setHi(dir,pressure_bc);
                  }
                  else {
                      rinflow_bc_lo[dir] = (is_inflow ? 1 : 0);
                      phys_bc.setLo(dir,component_bc);
                      pres_bc.setLo(dir,pressure_bc);
                  }
              }
              else {

                  bool is_consistent = true;
                  if (is_hi) {
                      is_consistent = ( (rinflow_bc_hi[dir] == is_inflow)
                                        && (phys_bc.hi()[dir] == component_bc)
                                        && (pres_bc.hi()[dir] == pressure_bc) );
                  }
                  else {
                      is_consistent = ( (rinflow_bc_lo[dir] == is_inflow)
                                        && (phys_bc.lo()[dir] == component_bc)
                                        && (pres_bc.lo()[dir] == pressure_bc) );
                  }

                  if (is_consistent) {
                      BoxLib::Abort("Inconconsistent type for boundary ");
                  }
              }
          }
      }
  }
}

using PMAMR::RlabelDEF;
void  PorousMedia::read_tracer()
{
  //
  // Read in parameters for tracers
  //
  ParmParse pp("tracer");

  // Get number of tracers
  ntracers = pp.countval("tracers");
  if (ntracers > 0)
  {
    tic_array.resize(ntracers);
    tbc_array.resize(ntracers);
      pp.getarr("tracers",tNames,0,ntracers);

      for (int i = 0; i<ntracers; i++)
      {
          const std::string prefix("tracer." + tNames[i]);
	  ParmParse ppr(prefix.c_str());
          if (do_tracer_chemistry>0  ||  do_tracer_advection  ||  do_tracer_diffusion) {
              setup_tracer_transport = true;
              std::string g="Total"; ppr.query("group",g); // FIXME: is this relevant anymore?
              group_map[g].push_back(i+ncomps);
          }
          else {
              setup_tracer_transport = false;
          }

          // Initial condition and boundary condition  
          Array<std::string> tic_names;
          int n_ic = ppr.countval("tinits");
          if (n_ic <= 0)
          {
              BoxLib::Abort("each tracer must be initialized");
          }
          ppr.getarr("tinits",tic_names,0,n_ic);
          tic_array[i].resize(n_ic,PArrayManage);
          
	  const RegionManager* region_manager = PMAmr::RegionManagerPtr();
          for (int n = 0; n<n_ic; n++)
          {
              const std::string prefixIC(prefix + "." + tic_names[n]);
              ParmParse ppri(prefixIC.c_str());
              int n_ic_region = ppri.countval("regions");
              Array<std::string> region_names;
              ppri.getarr("regions",region_names,0,n_ic_region);
	      Array<const Region*> tic_regions = region_manager->RegionPtrArray(region_names);
              std::string tic_type; ppri.get("type",tic_type);
              
              if (tic_type == "concentration")
              {
                  Real val = 0; ppri.query("val",val);
                  tic_array[i].set(n, new RegionData(tNames[i],tic_regions,tic_type,val));
              }
              else {
                  std::string m = "Tracer IC: \"" + tic_names[n] 
                      + "\": Unsupported tracer IC type: \"" + tic_type + "\"";
                  BoxLib::Abort(m.c_str());
              }
          }

          if (setup_tracer_transport)
          {
              Array<std::string> tbc_names;
              int n_tbc = ppr.countval("tbcs");
              ppr.getarr("tbcs",tbc_names,0,n_tbc);
              tbc_array[i].resize(n_tbc+2*BL_SPACEDIM,PArrayManage);

              // Explicitly build default BCs
              int tbc_cnt = 0;
              for (int n=0; n<BL_SPACEDIM; ++n) {
                tbc_array[i].set(tbc_cnt++, new RegionData(RlabelDEF[n] + "_DEFAULT",
                                                           region_manager->RegionPtrArray(Array<std::string>(1,RlabelDEF[n])),
                                                           std::string("noflow"),0));
                tbc_array[i].set(tbc_cnt++, new RegionData(RlabelDEF[n+3] + "_DEFAULT",
                                                           region_manager->RegionPtrArray(Array<std::string>(1,RlabelDEF[n+3])),
                                                           std::string("noflow"),0));
              }

              Array<int> orient_types(6,-1);
              for (int n = 0; n<n_tbc; n++)
              {
                  const std::string prefixTBC(prefix + "." + tbc_names[n]);
                  ParmParse ppri(prefixTBC.c_str());
                  
                  int n_tbc_region = ppri.countval("regions");
                  Array<std::string> tbc_region_names;
                  ppri.getarr("regions",tbc_region_names,0,n_tbc_region);

                  Array<const Region*> tbc_regions = region_manager->RegionPtrArray(tbc_region_names);
                  std::string tbc_type; ppri.get("type",tbc_type);

                  // When we get the BCs, we need to translate to AMR-standardized type id.  By
                  // convention, components are  Interior, Inflow, Outflow, Symmetry, SlipWall, NoSlipWall.
                  int AMR_BC_tID = -1;
                  if (tbc_type == "concentration")
                  {
                      Array<Real> times, vals;
                      Array<std::string> forms;
                      int nv = ppri.countval("vals");
                      if (nv) {
                          ppri.getarr("vals",vals,0,nv);
                          if (nv>1) {
                              ppri.getarr("times",times,0,nv);
                              ppri.getarr("forms",forms,0,nv-1);
                          }
                          else {
                              times.resize(1,0);
                          }
                      }
                      else {
                          vals.resize(1,0); // Default tracers to zero for all time
                          times.resize(1,0);
                          forms.resize(0);
                      }
                      int nComp = 1;
                      tbc_array[i].set(tbc_cnt++, new ArrayRegionData(tbc_names[n],times,vals,forms,tbc_regions,tbc_type,nComp));
                      AMR_BC_tID = 1; // Inflow
                  }
                  else if (tbc_type == "noflow")
                  {
                      Array<Real> val(1,0);
                      tbc_array[i].set(tbc_cnt++, new RegionData(tbc_names[n],tbc_regions,tbc_type,val));
                      AMR_BC_tID = 2;
                  }
                  else if (tbc_type == "outflow")
                  {
                      Array<Real> val(1,0);
                      tbc_array[i].set(tbc_cnt++, new RegionData(tbc_names[n],tbc_regions,tbc_type,val));
                      AMR_BC_tID = 3; // Outflow
                  }
                  else {
                      std::string m = "Tracer BC: \"" + tbc_names[n] 
                          + "\": Unsupported tracer BC type: \"" + tbc_type + "\"";
                      BoxLib::Abort(m.c_str());
                  }


                  for (int j=0; j<tbc_regions.size(); ++j)
                  {
                    const std::string purpose = tbc_regions[j]->purpose;
                    int dir = -1, is_hi, k;
                    for (int kt=0; kt<7 && dir<0; ++kt) {
                      if (purpose == PMAMR::RpurposeDEF[kt]) {
                        BL_ASSERT(kt != 6);
                        dir = kt%3;
                        is_hi = kt>=3;
                        k = kt;
                      }
                    }
                    if (dir<0 || dir > BL_SPACEDIM) {
                      std::cout << "Bad region for boundary: \n" << tbc_regions[j] << std::endl;
                      BoxLib::Abort();
                    }

                    if (orient_types[k] < 0) {
                      orient_types[k] = AMR_BC_tID;
                    } else {
                      if (orient_types[k] != AMR_BC_tID) {
                        BoxLib::Abort("BC for tracers must all be of same type on each side");
                      }
                    }
                  }
              }
              // Set the default BC type
              for (int k=0; k<orient_types.size(); ++k) {
                if (orient_types[k] < 0) orient_types[k] = 2;
              }

              BCRec phys_bc_trac;
              for (int i = 0; i < BL_SPACEDIM; i++) {
                phys_bc_trac.setLo(i,orient_types[i]);
                phys_bc_trac.setHi(i,orient_types[i+3]);
              }
              set_tracer_bc(trac_bc,phys_bc_trac);
          }
      }
      ndiff += ntracers;
  }
}

static
int loc_in_array(const std::string& val,const Array<std::string>& arr)
{
  int location = -1;
  for (int i=0; i<arr.size() && location<0; ++i) {
    if (val == arr[i]) location = i;
  }
  return location;
}


void  PorousMedia::read_source()
{
  //
  // Read in parameters for sources
  //
  ParmParse pp("source");
  ParmParse ppb("prob");
  ppb.query("do_source_term",do_source_term);

  int nsources = pp.countval("sources");
  if (nsources>0) {
    source_array.resize(nsources,PArrayManage);
    tsource_array.resize(nsources);
    Array<std::string> source_names(nsources);
    pp.getarr("sources",source_names,0,nsources);
    for (int i=0; i<nsources; ++i) {
      const std::string& source_name = source_names[i];
      const std::string prefix("source." + source_name);
      ParmParse pps(prefix.c_str());

      int n_src_regions = pps.countval("regions");
      Array<std::string> src_region_names; 
      pps.getarr("regions",src_region_names,0,n_src_regions);
      const RegionManager* region_manager = PMAmr::RegionManagerPtr();
      const Array<const Region*> source_regions = region_manager->RegionPtrArray(src_region_names);

      if (pps.countval("type")) {
	std::string source_type; pps.get("type",source_type);
	if (source_type == "uniform"
	    || source_type == "volume_weighted"
	    || source_type == "permeability_weighted"
	    || source_type == "point")
	  {
	    int nvars = pps.countval("vals");
	    BL_ASSERT(nvars>0);
	    Array<Real> vals; pps.getarr("vals",vals,0,nvars);

            if (source_type == "point") {
              BL_ASSERT(source_regions.size() == 1);
              BL_ASSERT(source_regions[0]->type=="point");
            }

	    source_array.set(i, new RegionData(source_name,source_regions,source_type,vals));
	  }
	else {
	  std::string m = "Source: \"" + source_names[i] 
	    + "\": Unsupported source type: \"" + source_type + "\"";
	  BoxLib::Abort(m.c_str());
	}
      }
      else {
	std::string m = "Source: \"" + source_names[i] 
	  + "\": Requires \"type\" specifier";
	BoxLib::Abort(m.c_str());
      }

      for (int ip=0; ip<pNames.size(); ++ip) {
	const std::string& pName = pNames[ip];
	const std::string p_prefix(prefix+"."+pName);
	ParmParse pps_p(p_prefix.c_str());
	
	for (int ic=0; ic<cNames.size(); ++ic) {
	  const std::string& cName = cNames[ic];
	  const std::string c_prefix(p_prefix+"."+cName);
	  ParmParse pps_c(c_prefix.c_str());

	  int ntracers_with_sources = pps_c.countval("tracers_with_sources");
	  if (ntracers_with_sources>0) {
	    Array<std::string> tracers_with_sources;
	    pps_c.getarr("tracers_with_sources",tracers_with_sources,0,ntracers_with_sources);
	    tsource_array[i].resize(ntracers, PArrayManage);
	    
	    for (int it=0; it<tracers_with_sources.size(); ++it) {
	      const std::string& tName = tracers_with_sources[it];
	      int t_pos = loc_in_array(tName,tNames);
	      if (t_pos>=0) {
		const std::string c_t_prefix(c_prefix+"."+tName);
		ParmParse pps_c_t(c_t_prefix.c_str());
	      
		if (pps_c_t.countval("type")) {
		  std::string tsource_type; pps_c_t.get("type",tsource_type);              
		  if (tsource_type == "uniform"
		      || tsource_type == "flow_weighted"
		      || tsource_type == "point")
		    {
		      int ntvars = pps_c_t.countval("vals");
		      BL_ASSERT(ntvars>0);
		      Array<Real> tvals; pps_c_t.getarr("vals",tvals,0,ntvars);
		      tsource_array[i].set(t_pos, new RegionData(source_name,source_regions,tsource_type,tvals));
		    }
		  else {
		    BoxLib::Abort(std::string("Source: \"" + source_names[i] + 
					      " \"" + cName + "\" Solute SOURCE: \"" + tName
					      + "\": Unsupported source type: \"" + tsource_type + "\"").c_str());
		  }
		} else {
		  BoxLib::Abort(std::string("Source: \"" + source_names[i] 
					    + "\": Requires \"type\" specifier for solute \""+tName+"\"").c_str());
		}
		if (pps_c_t.countval("Concentration_Units")) {
		  // FIXME: We do not currently do anything with this parameter
		}
	      }
	      else {
		BoxLib::Abort(std::string("Source: \"" + source_names[i]
					  + "\" contains unknown tracer: \""+tName+"\"").c_str());
	      }
	    }

	    // Set default source (uniform=0) for all tracers not set explicitly
	    const std::string default_tsource_type = "uniform";
	    const Array<Real> default_tsource_tvals(1,0);
	    for (int it=0; it<ntracers; ++it) {
	      if ( !(tsource_array[i].defined(it)) ) {
		tsource_array[i].set(it, new RegionData(source_name,source_regions,default_tsource_type,default_tsource_tvals));
	      }
	    }
	  }
	}
      }
    }
  }
}

void  PorousMedia::read_chem()
{

  ParmParse pp("prob");

  // get Chemistry stuff
  pp.query("do_chem",do_tracer_chemistry);
  pp.query("do_full_strang",do_full_strang);
  pp.query("n_chem_interval",n_chem_interval);
  pp.query("ic_chem_relax_dt",ic_chem_relax_dt);
  if (n_chem_interval > 0) 
    {
      do_full_strang = 0;
    }
      
  std::string chemistry_model = "Amanzi"; pp.query("chemistry_model",chemistry_model);

  // chemistry...
#ifdef AMANZI

  if (do_tracer_chemistry) {

#if ALQUIMIA_ENABLED
      const Teuchos::ParameterList& chemistry_parameter_list = PorousMedia::InputParameterList().sublist("Chemistry");
      chemistry_engine = new Amanzi::AmanziChemistry::Chemistry_Engine(chemistry_parameter_list);
      std::vector<std::string> primarySpeciesNames, mineralNames, siteNames, ionExchangeNames, isothermSpeciesNames;
      chemistry_engine->GetPrimarySpeciesNames(primarySpeciesNames);
      chemistry_engine->GetMineralNames(mineralNames);
      chemistry_engine->GetSurfaceSiteNames(siteNames);
      chemistry_engine->GetIonExchangeNames(ionExchangeNames);
      chemistry_engine->GetIsothermSpeciesNames(isothermSpeciesNames);
      int numPrimarySpecies = primarySpeciesNames.size();
      int numSorbedSpecies = chemistry_engine->NumSorbedSpecies();
      int numMinerals = mineralNames.size();
      int numSurfaceSites = chemistry_engine->NumSurfaceSites();
      int numIonExchangeSites = ionExchangeNames.size();
      int numIsothermSpecies = isothermSpeciesNames.size();
#else
      Amanzi::AmanziChemistry::SetupDefaultChemistryOutput();

      ParmParse pb("prob.amanzi");
      
      std::string verbose_chemistry_init = "silent"; pb.query("verbose_chemistry_init",verbose_chemistry_init);
      
      if (verbose_chemistry_init == "silent") {
	Amanzi::AmanziChemistry::chem_out->AddLevel("silent");
      }

      std::string fmt = "simple"; pb.query("Thermodynamic_Database_Format",fmt);
      pb.query("chem_database_file", amanzi_database_file);
      
      const std::string& activity_model_dh = Amanzi::AmanziChemistry::ActivityModelFactory::debye_huckel;
      const std::string& activity_model_ph = Amanzi::AmanziChemistry::ActivityModelFactory::pitzer_hwm;
      const std::string& activity_model_u  = Amanzi::AmanziChemistry::ActivityModelFactory::unit;
      std::string activity_model = activity_model_u; pp.query("Activity_Model",activity_model);
      
      Real tolerance=1.5e-12; pp.query("Tolerance",tolerance);
      int max_num_Newton_iters = 150; pp.query("Maximum_Newton_Iterations",max_num_Newton_iters);
      std::string outfile=""; pp.query("Output_File_Name",outfile);
      bool use_stdout = true; pp.query("Use_Standard_Out",use_stdout);
      int num_aux = pp.countval("Auxiliary_Data");
      if (num_aux>0) {
	Array<std::string> tmpaux(num_aux);
	aux_chem_variables.clear();
	pp.getarr("Auxiliary_Data",tmpaux,0,num_aux);
	for (int i=0;i<num_aux;i++)
	  aux_chem_variables[tmpaux[i]] = i;
      }

      ICParmPair solute_chem_options;
      solute_chem_options["Free_Ion_Guess"] = 1.e-9;
      solute_chem_options["Activity_Coefficient"] = 1;
      for (int k=0; k<tNames.size(); ++k) {
        for (ICParmPair::const_iterator it=solute_chem_options.begin(); it!=solute_chem_options.end(); ++it) {
          const std::string& str = it->first;
          bool found = false;
          for (int i=0; i<materials.size(); ++i) {
            const std::string& rname = materials[i].Name();
            const std::string prefix("tracer."+tNames[i]+".Initial_Condition");
            ParmParse pprs(prefix.c_str());
            solute_chem_ics[rname][tNames[k]][str] = it->second; // set to default value
            pprs.query(str.c_str(),solute_chem_ics[rname][tNames[k]][str]);
            const std::string label = str+"_"+tNames[k];

            if (aux_chem_variables.find(label) == aux_chem_variables.end())
            {
              solute_chem_label_map[tNames[k]][str] = aux_chem_variables.size();
              aux_chem_variables[label]=aux_chem_variables.size()-1;
            }
          }
        }
      }

      // TODO: add secondary species activity coefficients


      // TODO: here down goes into seperate function init_chem()
      
      //
      // In order to thread the AMANZI chemistry, we had to give each thread 
      // its own chemSolve and components object.
      //
      int tnum = 1;
#ifdef _OPENMP
      tnum = omp_get_max_threads();
#endif
      chemSolve.resize(tnum);
      components.resize(tnum);
      parameters.resize(tnum);
      
      
      for (int ithread = 0; ithread < tnum; ithread++)
        {
	  chemSolve.set(ithread, new Amanzi::AmanziChemistry::SimpleThermoDatabase());
	  
	  parameters[ithread] = chemSolve[ithread].GetDefaultParameters();
	  parameters[ithread].thermo_database_file = amanzi_database_file;
	  parameters[ithread].activity_model_name = activity_model;
	  parameters[ithread].porosity   = 0.25; 
	  parameters[ithread].saturation = 0;
	  parameters[ithread].volume     = 1.0;
	  parameters[ithread].water_density = density[0];
	  
	  // minimal initialization of the chemistry arrays to the
	  // correct size based on the xml input. Remaining arrays
	  // will be sized by chemistry
	  components[ithread].total.resize(ntracers,1.0e-40);
	  components[ithread].free_ion.resize(ntracers, 1.0e-9);
	  
	  components[ithread].mineral_volume_fraction.resize(nminerals, 0.0);
	  
	  if (using_sorption) { 
	    components[ithread].total_sorbed.resize(ntracers, 1.0e-40);
	  }

	  chemSolve[ithread].verbosity(Amanzi::AmanziChemistry::kTerse);
	  
	  // initialize the chemistry objects
	  chemSolve[ithread].Setup(components[ithread], parameters[ithread]);
	  // let chemistry finish resizing the arrays
	  chemSolve[ithread].CopyBeakerToComponents(&(components[ithread]));
	  if (ParallelDescriptor::IOProcessor() && ithread == 0) {
	    chemSolve[ithread].Display();
	    chemSolve[ithread].DisplayComponents(components[ithread]);
	  }
	}  // for(threads)
      
      // Verify that amr and chemistry agree on the names and ordering of the tracers
      std::vector<std::string> chem_names;
      chemSolve[0].GetPrimaryNames(&chem_names);
      BL_ASSERT(chem_names.size() == tNames.size());
      for (int i = 0; i < chem_names.size(); ++i) {
	if (chem_names.at(i) != tNames[i]) {
	  if (ParallelDescriptor::IOProcessor()) {
	    std::stringstream message;
	    message << "PM_setup::read_chem():\n"
		    << "  chemistry and amr do not agree on the name of tracer " << i << ".\n"
		    << "  chemistry : " << chem_names.at(i) << "\n"
		    << "  amr : " << tNames[i] << "\n";
	    BoxLib::Warning(message.str().c_str());
	  }
	}
      }

      // TODO: not needed for 2012 demo... request additional
      // secondary storage data from chemistry object (secondary
      // species activity coefficients)
      if (components[0].secondary_activity_coeff.size() > 0) {
	// allocate additional storage for secondary activity coeffs
      }
     
#endif
    }
#endif

  pp.query("use_funccount",use_funccount);
  pp.query("max_grid_size_chem",max_grid_size_chem);
  BL_ASSERT(max_grid_size_chem > 0);
}


void PorousMedia::read_params()
{
  // problem-specific
  read_prob();

  PMAmr::SetRegionManagerPtr(new RegionManager());
  RegionManager* region_manager = PMAmr::RegionManagerPtr();

  if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
    std::cout << "Read geometry." << std::endl;

  if (echo_inputs && ParallelDescriptor::IOProcessor()) {
      std::cout << "The Regions: " << std::endl;
      const Array<const Region*> regions = region_manager->RegionPtrArray();
      for (int i=0; i<regions.size(); ++i) {
	std::cout << *(regions[i]) << std::endl;
      }
  }

  // components and phases
  read_comp();
  if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
    std::cout << "Read components."<< std::endl;
  
  // chem requires the number of tracers and rocks be setup before we
  // can do anything, but read_tracer depends on do_tracer_chemistry
  // already being set. We'll query that now and do the remaining
  // chemistry after everything else has been read
  ParmParse pp("prob");
  pp.query("do_chem",do_tracer_chemistry);

  // tracers
  read_tracer();
  if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
    std::cout << "Read tracers."<< std::endl;

  // chemistry. Needs to come after tracers (and rock?) have been setup.
  if (verbose > 1 && ParallelDescriptor::IOProcessor())
    std::cout << "Read chemistry."<< std::endl;
  read_chem();

  // source
  read_source();
  if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
    std::cout << "Read sources."<< std::endl;
  
  int model_int = Model();
  FORT_INITPARAMS(&ncomps,&nphases,&model_int,density.dataPtr(),
		  muval.dataPtr(),pType.dataPtr(),
		  &gravity,&gravity_dir);
    
  if (ntracers > 0)
    FORT_TCRPARAMS(&ntracers);

}

