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
#include <POROUS_F.H>
#include <WritePlotfile.H>

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
#include "simple_thermo_database.hh"
#include "activity_model_factory.hh"
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
// Add 2 if do_chem>0 later.
//
int PorousMedia::num_state_type;
//
// Region.
//
std::string    PorousMedia::surf_file;
PArray<Region> PorousMedia::regions;
PArray<Material> PorousMedia::materials;
//
// Rock
//
MultiFab*   PorousMedia::kappadata;
MultiFab*   PorousMedia::phidata;
bool        PorousMedia::porosity_from_fine;
bool        PorousMedia::permeability_from_fine;
Real        PorousMedia::saturation_threshold_for_vg_Kr;
int         PorousMedia::use_shifted_Kr_eval;
DataServices* PorousMedia::phi_dataServices;
DataServices* PorousMedia::kappa_dataServices;
std::string PorousMedia::phi_plotfile_varname;
Array<std::string> PorousMedia::kappa_plotfile_varnames;
//
// Source.
//
bool          PorousMedia::do_source_term;
Array<Source> PorousMedia::source_array;
//
// Phases and components.
//
Array<std::string>  PorousMedia::pNames;
Array<std::string>  PorousMedia::cNames;
Array<int >         PorousMedia::pType;
Array<Real>         PorousMedia::density;
PArray<RegionData>  PorousMedia::ic_array;
PArray<RegionData>  PorousMedia::bc_array;
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
bool PorousMedia::do_tracer_transport;
bool PorousMedia::setup_tracer_transport;
int  PorousMedia::transport_tracers;
bool PorousMedia::diffuse_tracers;
bool PorousMedia::solute_transport_limits_dt;

//
// Chemistry flag.
//
int  PorousMedia::do_chem;
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
std::string PorousMedia::amanzi_database_file;
std::string PorousMedia::amanzi_activity_model;

PArray<amanzi::chemistry::SimpleThermoDatabase>    PorousMedia::chemSolve(PArrayManage);
Array<amanzi::chemistry::Beaker::BeakerComponents> PorousMedia::components;
Array<amanzi::chemistry::Beaker::BeakerParameters> PorousMedia::parameters;
#endif
//
// Internal switches.
//
int  PorousMedia::do_simple;
int  PorousMedia::do_multilevel_full;
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
bool PorousMedia::richard_subgrid_krel;
Real PorousMedia::richard_variable_switch_saturation_threshold;
Real PorousMedia::richard_dt_thresh_pure_steady;

RichardSolver* PorousMedia::richard_solver;

PorousMedia::ExecutionMode PorousMedia::execution_mode;
Real PorousMedia::switch_time;

namespace
{
    static void PM_Setup_CleanUpStatics() 
    {
	DataServices* phids = PorousMedia::PhiData(); delete phids; phids=0;
    }
}

static Box grow_box_by_one (const Box& b) { return BoxLib::grow(b,1); }

//
// Components are  Interior, Inflow, Outflow, Symmetry, SlipWall, NoSlipWall.
//

static int scalar_bc[] =
  {
    //    INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, FOEXTRAP, SEEPAGE
    INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_ODD, SEEPAGE
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
  PorousMedia::phi_dataServices = 0;
  PorousMedia::kappa_dataServices = 0;

  PorousMedia::porosity_from_fine     = false;
  PorousMedia::permeability_from_fine = false;

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
  PorousMedia::initial_step = false;
  PorousMedia::initial_iter = false;
  PorousMedia::sum_interval = -1;
  PorousMedia::NUM_SCALARS  = 0;
  PorousMedia::NUM_STATE    = 0;
  PorousMedia::full_cycle   = 0;

  PorousMedia::be_cn_theta           = 0.5;
  PorousMedia::visc_tol              = 1.0e-10;  
  PorousMedia::visc_abs_tol          = 1.0e-10;  
  PorousMedia::def_harm_avg_cen2edge = true;

  PorousMedia::have_capillary = 0;

  PorousMedia::saturation_threshold_for_vg_Kr = -1; // <0 bypasses smoothing
  PorousMedia::use_shifted_Kr_eval = 0; //

  PorousMedia::variable_scal_diff = 1; 

  PorousMedia::do_chem            = 0;
  PorousMedia::do_tracer_transport = false;
  PorousMedia::setup_tracer_transport = false;
  PorousMedia::transport_tracers  = 0;
  PorousMedia::diffuse_tracers    = false;
  PorousMedia::do_full_strang     = 0;
  PorousMedia::n_chem_interval    = 0;
  PorousMedia::it_chem            = 0;
  PorousMedia::dt_chem            = 0;
  PorousMedia::max_grid_size_chem = 16;
  PorousMedia::no_initial_values  = true;
  PorousMedia::use_funccount      = false;

  PorousMedia::do_simple           = 0;
  PorousMedia::do_multilevel_full  = 1;
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
  PorousMedia::nGrowHYP = 3;
  PorousMedia::nGrowMG = 1;
  PorousMedia::nGrowEIGEST = 1;
  PorousMedia::max_n_subcycle_transport = 10;
  PorousMedia::max_dt_iters_flow = 20;
  PorousMedia::verbose_chemistry = 0;
  PorousMedia::abort_on_chem_fail = true;
  PorousMedia::show_selected_runtimes = 0;

  PorousMedia::richard_solver_verbose = 2;

  PorousMedia::do_richard_init_to_steady = false;
  PorousMedia::richard_init_to_steady_verbose = 1;
  PorousMedia::steady_min_iterations = 10;
  PorousMedia::steady_min_iterations_2 = 2;
  PorousMedia::steady_max_iterations = 15;
  PorousMedia::steady_limit_iterations = 20;
  PorousMedia::steady_time_step_reduction_factor = 0.8;
  PorousMedia::steady_time_step_increase_factor = 1.8;
  PorousMedia::steady_time_step_increase_factor_2 = 10;
  PorousMedia::steady_time_step_retry_factor_1 = 0.05;
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
  PorousMedia::richard_subgrid_krel = false;
  PorousMedia::richard_variable_switch_saturation_threshold = -1;
  PorousMedia::richard_dt_thresh_pure_steady = -1;

  PorousMedia::echo_inputs    = 0;
  PorousMedia::richard_solver = 0;
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

  is_diffusive.resize(NUM_SCALARS);
  advectionType.resize(NUM_SCALARS);
  diffusionType.resize(NUM_SCALARS);

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

  if (do_chem && ntracers > 0)
  {
      // NOTE: aux_chem_variables is setup by read_rock and read_chem as data is
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
  if (do_chem>0)
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

  // "User defined" - atthough these must correspond to those in PorousMedia::derive
  IndexType regionIDtype(IndexType::TheCellType());
  int nCompRegion = 1;
  std::string amr_prefix = "amr";
  ParmParse pp(amr_prefix);
  int num_user_derives = pp.countval("user_derive_list");
  Array<std::string> user_derive_list(num_user_derives);
  pp.getarr("user_derive_list",user_derive_list,0,num_user_derives);
  for (int i=0; i<num_user_derives; ++i) {
      derive_lst.add(user_derive_list[i], regionIDtype, nCompRegion);
  }

  //
  // **************  DEFINE ERROR ESTIMATION QUANTITIES  *************
  //
  //err_list.add("gradn",1,ErrorRec::Special,ErrorFunc(FORT_ADVERROR));

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
      PArray<Region> regions = build_region_PArray(region_names);
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

//
//  Read input file
//
void PorousMedia::read_geometry()
{
  //
  // Get geometry-related parameters.  
  // Note: 1. The domain size and periodity information are read in 
  //          automatically.  This function deals primarily with region
  //          definition.
  //       2. regions defined in PorousMedia.H as PArray<Region>
  //
  ParmParse pp("geometry");

  Array<Real> problo, probhi;
  pp.getarr("prob_lo",problo,0,BL_SPACEDIM);
  pp.getarr("prob_hi",probhi,0,BL_SPACEDIM);
  Region::domlo = problo;
  Region::domhi = probhi;
  
  Real geometry_eps = -1; pp.get("geometry_eps",geometry_eps);
  Region::geometry_eps = geometry_eps;

  // set up  1+2*BL_SPACEDIM default regions
  bool generate_default_regions = true; pp.query("generate_default_regions",generate_default_regions);
  int nregion_DEF = 0;
  regions.clear();
  if (generate_default_regions) {
      nregion_DEF = 1 + 2*BL_SPACEDIM;
      regions.resize(nregion_DEF,PArrayManage);
      regions.set(0, new   AllRegion());
      regions.set(1, new AllBCRegion(0,0));
      regions.set(2, new AllBCRegion(0,1));
      regions.set(3, new AllBCRegion(1,0));
      regions.set(4, new AllBCRegion(1,1));
#if BL_SPACEDIM == 3
      regions.set(5, new AllBCRegion(2,0));
      regions.set(6, new AllBCRegion(2,1));
#endif
  }

  // Get parameters for each user defined region 
  int nregion = nregion_DEF;

  int nregion_user = pp.countval("regions");

  if (!generate_default_regions  && nregion_user==0) {
      BoxLib::Abort("Default regions not generated and none provided.  Perhaps omitted regions list?");
  }
  if (nregion_user)
    {
      std::string r_purpose, r_type;
      Array<std::string> r_name;
      pp.getarr("regions",r_name,0,nregion_user);
      nregion += nregion_user;
      regions.resize(nregion,PArrayManage);

      for (int j=0; j<nregion_user; ++j)
      {
          const std::string prefix("geometry." + r_name[j]);
          ParmParse ppr(prefix.c_str());
	  ppr.get("purpose",r_purpose);
	  ppr.get("type",r_type);      

	  if (r_type == "point")
          {
	      Array<Real> coor;
	      ppr.getarr("coordinate",coor,0,BL_SPACEDIM);
              regions.set(nregion_DEF+j, new PointRegion(r_name[j],r_purpose,coor));
	    }
	  else if (r_type == "box" || r_type == "surface")
	    {
	      Array<Real> lo_coor,hi_coor;
	      ppr.getarr("lo_coordinate",lo_coor,0,BL_SPACEDIM);
	      ppr.getarr("hi_coordinate",hi_coor,0,BL_SPACEDIM);
              regions.set(nregion_DEF+j, new BoxRegion(r_name[j],r_purpose,lo_coor,hi_coor));
	    }
	  else if (r_type == "color_function")
          {
              int color_value; ppr.get("color_value",color_value);
              std::string color_file; ppr.get("color_file",color_file);
              ColorFunctionRegion* cfr = new ColorFunctionRegion(r_name[j],r_purpose,color_file,color_value);
	      regions.set(nregion_DEF+j, cfr);
          }
          else {
              std::string m = "region type not supported \"" + r_type + "\"";
              BoxLib::Abort(m.c_str());
          }
	}
      pp.query("surf_file",surf_file);
    }
}

//using AmanziS::WritePlotfile;
void
WriteMaterialPltFile(int max_level,const Array<int>& n_cell,const Array<int>& fRatio,
		     const Real* problo, const Real* probhi, MultiFab* data,
		     const std::string& filename, int nGrow, const Array<int>& harmDir,
                     const Array<std::string>& names)
{
  int nLevs = max_level + 1;
  Array<MultiFab*> mlData(nLevs);
  mlData[max_level] = (MultiFab*) data; // will not change this one

  Array<Box> pd(nLevs), gpd(nLevs);
  Array<Array<Real> > dxLevel(nLevs,Array<Real>(BL_SPACEDIM));

  pd[0] = Box(IntVect::TheZeroVector(),
	      IntVect(n_cell.dataPtr())-IntVect::TheUnitVector());

  int fineRatio = 1;
  for (int lev=0; lev<=max_level; ++lev) {
    if (lev>0) {
      pd[lev] = Box(pd[lev-1]).refine(fRatio[lev-1]);
      fineRatio *= fRatio[lev-1];
    }
    for (int d=0; d<BL_SPACEDIM; ++d) {
      dxLevel[lev][d] = (Real)(probhi[d]-problo[d])/pd[lev].length(d);
    }
    gpd[lev] = Box(pd[lev]).grow(fineRatio*nGrow);
  }
  int nGrowFINE = fineRatio * nGrow;

  Real gplo[BL_SPACEDIM], gphi[BL_SPACEDIM];
  for (int d=0; d<BL_SPACEDIM; ++d) {
    gplo[d] = problo[d] - dxLevel[0][d]*nGrow;
    gphi[d] = probhi[d] + dxLevel[0][d]*nGrow;
  }

  int max_size = 64;
  int ratioToFinest = 1;
  int nComp = harmDir.size(); BL_ASSERT(data->nComp()>=nComp);
  
  for (int lev=max_level-1; lev>=0; --lev) {
    ratioToFinest *= fRatio[lev];
    
    BoxArray bac(gpd[lev]);
    bac.maxSize(max_size);
    BoxArray baf = BoxArray(bac).refine(ratioToFinest);
    BoxArray bafg = BoxArray(baf).grow(nGrowFINE);
    MultiFab mffine_ng(bafg,nComp,0);      
    mffine_ng.copy(*mlData[max_level],0,0,nComp); // parallel copy
    
    mlData[lev] = new MultiFab(bac,nComp,nGrow);    
    for (MFIter mfi(*mlData[lev]); mfi.isValid(); ++mfi) {
      const int* lo    = mfi.validbox().loVect();
      const int* hi    = mfi.validbox().hiVect();
      const FArrayBox& fine = mffine_ng[mfi];
      FArrayBox& crse       = (*mlData[lev])[mfi];
      
      for (int d=0; d<nComp; ++d) {
        FORT_CRSENMAT (fine.dataPtr(d), ARLIM(fine.loVect()), ARLIM(fine.hiVect()),
                       crse.dataPtr(d), ARLIM(crse.loVect()), ARLIM(crse.hiVect()),
                       lo, hi, &ratioToFinest, &(harmDir[d]));
      }
    }    
  }

  std::string pfversion = "MaterialData-0.1";
  Real t = 0;
  int coordSys = 0;
  bool plt_verbose = false;
  bool isCartGrid = false;
  Real vfeps = 1.e-12;
  int levelSteps = 0;
  WritePlotfile(pfversion,mlData,t,gplo,gphi,fRatio,gpd,dxLevel,coordSys,
		filename,names,plt_verbose,isCartGrid,&vfeps,&levelSteps);

  for (int lev=0; lev<max_level-1; ++lev) {
    delete mlData[lev];
  }
}

static DataServices*
OpenMaterialDataPltFile(const std::string& filename,
                        int                max_level,
                        const Array<int>&  ratio,
                        const Array<int>&  n_cell,
                        int                nGrowMAX)
{
  DataServices::SetBatchMode();
  Amrvis::FileType fileType(Amrvis::NEWPLT);
  DataServices* ret = new DataServices(filename, fileType);
  if (!ret->AmrDataOk())
    DataServices::Dispatch(DataServices::ExitRequest, NULL);
  AmrData& amrData = ret->AmrDataRef();

  // Verify kappa data is compatible with current run
  bool is_compatible = amrData.FinestLevel()>=max_level;
  BL_ASSERT(ratio.size()<max_level);
  for (int lev=0; lev<max_level && is_compatible; ++lev) {
    is_compatible &= amrData.RefRatio()[lev] == ratio[lev];
  }
  if (is_compatible) {
    Box probDomain=Box(IntVect::TheZeroVector(),
                       IntVect(n_cell.dataPtr())-IntVect::TheUnitVector());
    probDomain.grow(nGrowMAX);
    is_compatible &= amrData.ProbDomain()[0].contains(probDomain);
  }
  if (!is_compatible) {
    delete ret;
    ret = 0;
  }
}

void
PorousMedia::read_rock(int do_chem)
{
    //
    // Get parameters related to rock
    //
    ParmParse pp("rock");
    int nrock = pp.countval("rock");
    if (nrock <= 0) {
        BoxLib::Abort("At least one rock type must be defined.");
    }
    Array<std::string> r_names;  pp.getarr("rock",r_names,0,nrock);

    materials.clear();
    materials.resize(nrock,PArrayManage);
    Array<std::string> material_regions;
    for (int i = 0; i<nrock; i++)
    {
        const std::string& rname = r_names[i];
        const std::string prefix("rock." + rname);
        ParmParse ppr(prefix.c_str());
        
        static Property::CoarsenRule arith_crsn = Property::Arithmetic;
        static Property::CoarsenRule harm_crsn = Property::ComponentHarmonic;
        static Property::RefineRule pc_refine = Property::PiecewiseConstant;

        Real rdensity = -1; // ppr.get("density",rdensity); // not actually used anywhere

        Real rDeff = -1;
        if (ppr.countval("effective_diffusion_coefficient.val")) {
          ppr.get("effective_diffusion_coefficient.val",rDeff);
          diffuse_tracers = true;
        }
        std::string Deff_str = "Deff";
        Property* Deff_func = new ConstantProperty(Deff_str,rDeff,harm_crsn,pc_refine);

        Property* phi_func = 0;
        std::string phi_str = "porosity";
        Array<Real> rpvals(1), rptimes;
        Array<std::string> rpforms;
        if (ppr.countval("porosity.vals")) {
          ppr.getarr("porosity.vals",rpvals,0,ppr.countval("porosity.vals"));
          int nrpvals = rpvals.size();
          if (nrpvals>1) {
            ppr.getarr("porosity.times",rptimes,0,nrpvals);
            ppr.getarr("porosity.forms",rpforms,0,nrpvals-1);
            TabularFunction pft(rptimes,rpvals,rpforms);
            phi_func = new TabularInTimeProperty(phi_str,pft,arith_crsn,pc_refine);
          }
          else {
            phi_func = new ConstantProperty(phi_str,rpvals[0],arith_crsn,pc_refine);
          }
        } else if (ppr.countval("porosity")) {
          ppr.get("porosity",rpvals[0]); // FIXME: For backward compatibility
          phi_func = new ConstantProperty(phi_str,rpvals[0],arith_crsn,pc_refine);
        } else {
          BoxLib::Abort(std::string("No porosity function specified for rock: \""+rname).c_str());
        }


        Property* kappa_func = 0;
        std::string kappa_str = "permeability";
        Array<Real> rvpvals(1), rhpvals(1), rvptimes(1), rhptimes(1);
        Array<std::string> rvpforms, rhpforms;

        Array<Real> rperm_in(2);
        if (ppr.countval("permeability")) {
          ppr.getarr("permeability",rperm_in,0,2);
          rhpvals[0] = rperm_in[0];
          rvpvals[0] = rperm_in[1];
        }
        else {

          int nrvpvals = ppr.countval("permeability.vertical.vals");
          int nrhpvals = ppr.countval("permeability.horizontal.vals");
          if (nrvpvals>0 && nrhpvals>0) {
            ppr.getarr("permeability.vertical.vals",rvpvals,0,nrvpvals);
            if (nrvpvals>1) {
              ppr.getarr("permeability.vertical.times",rvptimes,0,nrvpvals);
              ppr.getarr("permeability.vertical.forms",rvpforms,0,nrvpvals-1);
            }

            ppr.getarr("permeability.horizontal.vals",rhpvals,0,nrhpvals);
            if (nrhpvals>1) {
              ppr.getarr("permeability.horizontal.times",rhptimes,0,nrhpvals);
              ppr.getarr("permeability.horizontal.forms",rhpforms,0,nrhpvals-1);
            }

          } else {
            BoxLib::Abort(std::string("No permeability function specified for rock: \""+rname).c_str());
          }
        }

        // The permeability is specified in mDa.  
        // This needs to be multiplied with 1e-10 to be consistent 
        // with the other units in the code.  What this means is that
        // we will be evaluating the darcy velocity as:
        //
        //  u_Darcy [m/s] = ( kappa [X . mD] / mu [Pa.s] ).Grad(p) [atm/m]
        //
        // where X is the factor necessary to have this formula be dimensionally
        // consistent.  X here is 1.e-10, and can be combined with kappa for the 
        // the moment because no other derived quantities depend directly on the 
        // value of kappa  (NOTE: We will have to know that this is done however
        // if kappa is used as a diagnostic or in some way for a derived quantity).
        //
        for (int j=0; j<rhpvals.size(); ++j) {
          rhpvals[j] *= 1.e-10;
        }
        for (int j=0; j<rvpvals.size(); ++j) {
          rvpvals[j] *= 1.e-10;
        }

        // Define Property functions for Material Property server.  Eventually, these
        // will replace "Rock", but not yet
        if (rvpvals.size()>1 || rhpvals.size()>1) {
          Array<TabularFunction> pft(2);
          pft[0] = TabularFunction(rhptimes,rhpvals,rhpforms);
          pft[1] = TabularFunction(rvptimes,rvpvals,rvpforms);
          kappa_func = new TabularInTimeProperty(kappa_str,pft,harm_crsn,pc_refine);
        }
        else {
          Array<Real> vals(2); vals[0] = rhpvals[0]; vals[1] = rvpvals[0];
          kappa_func = new ConstantProperty(kappa_str,vals,harm_crsn,pc_refine);
        }

        // Set old-style values
	Array<Real> rpermeability(BL_SPACEDIM,rvpvals[0]);
	for (int j=0;j<BL_SPACEDIM-1;j++) rpermeability[j] = rhpvals[0];
	// rpermeability will always be of size BL_SPACEDIM

        // relative permeability: include kr_coef, sat_residual
        int rkrType = 0;  ppr.query("kr_type",rkrType);
        Array<Real> rkrParam;
        if (rkrType > 0) {
            ppr.getarr("kr_param",rkrParam,0,ppr.countval("kr_param"));
        }

        Array<Real> krPt(rkrParam.size()+1);
        krPt[0] = Real(rkrType);
        for (int j=0; j<rkrParam.size(); ++j) {
          krPt[j+1] = rkrParam[j];
        }
        std::string krel_str = "relative_permeability";
        Property* krel_func = new ConstantProperty(krel_str,krPt,arith_crsn,pc_refine);

        // capillary pressure: include cpl_coef, sat_residual, sigma
        int rcplType = 0;  ppr.query("cpl_type", rcplType);
        Array<Real> rcplParam;
        if (rcplType > 0) {
            ppr.getarr("cpl_param",rcplParam,0,ppr.countval("cpl_param"));
        }
        Array<Real> cplPt(rcplParam.size()+1);
        cplPt[0] = Real(rcplType);
        for (int j=0; j<rcplParam.size(); ++j) {
          cplPt[j+1] = rcplParam[j];
        }
        std::string cpl_str = "capillary_pressure";
        Property* cpl_func = new ConstantProperty(cpl_str,cplPt,arith_crsn,pc_refine);

        Array<std::string> region_names;
        ppr.getarr("regions",region_names,0,ppr.countval("regions"));
        PArray<Region> rregions = build_region_PArray(region_names);
        for (int j=0; j<region_names.size(); ++j) {
            material_regions.push_back(region_names[j]);
        }

        if (ppr.countval("porosity_dist_param")>0) {
          BoxLib::Abort("porosity_dist_param not currently supported");
        }

        std::string porosity_dist="uniform"; ppr.query("porosity_dist",porosity_dist);
        Array<Real> rporosity_dist_param;
        if (porosity_dist!="uniform") {
          BoxLib::Abort("porosity_dist != uniform not currently supported");
          ppr.getarr("porosity_dist_param",rporosity_dist_param,
                     0,ppr.countval("porosity_dist_param"));
        }
        
        std::string permeability_dist="uniform"; ppr.get("permeability_dist",permeability_dist);
        Array<Real> rpermeability_dist_param;
        if (permeability_dist != "uniform")
        {
          ppr.getarr("permeability_dist_param",rpermeability_dist_param,
                     0,ppr.countval("permeability_dist_param"));
        }

        std::vector<Property*> properties;
        properties.push_back(phi_func);
        properties.push_back(kappa_func);
        properties.push_back(Deff_func);
        properties.push_back(krel_func);
        properties.push_back(cpl_func);
        materials.set(i,new Material(rname,rregions,properties));
        delete phi_func;
        delete kappa_func;
        delete Deff_func;
        delete krel_func;
        delete cpl_func;
    }

    // Read rock parameters associated with chemistry
    using_sorption = false;
    aux_chem_variables.clear();

    if (do_chem>0)
      {
        ParmParse ppm("mineral");
        nminerals = ppm.countval("minerals");
        minerals.resize(nminerals);
        if (nminerals>0) {
	  ppm.getarr("minerals",minerals,0,nminerals);
        }

        ParmParse pps("sorption_site");
        nsorption_sites = pps.countval("sorption_sites");
        sorption_sites.resize(nsorption_sites);
        if (nsorption_sites>0) {
	  pps.getarr("sorption_sites",sorption_sites,0,nsorption_sites);
        }

	ICParmPair sorption_isotherm_options;
	sorption_isotherm_options[          "Kd"] = 0;
	sorption_isotherm_options[  "Langmuir_b"] = 0;
	sorption_isotherm_options["Freundlich_n"] = 1;
	
	for (int k=0; k<tNames.size(); ++k) {
	  for (ICParmPair::const_iterator it=sorption_isotherm_options.begin();
	       it!=sorption_isotherm_options.end(); ++it) {
	    const std::string& str = it->first;
	    bool found = false;
	    for (int i=0; i<nrock; ++i) {
	      const std::string prefix("rock."+r_names[i]+".Sorption_Isotherms."+tNames[k]);
	      ParmParse pprs(prefix.c_str());
	      if (pprs.countval(str.c_str())) {
		pprs.get(str.c_str(),sorption_isotherm_ics[r_names[i]][tNames[k]][str]);
		found = true;
	      }
	    }
	    
	    if (found) {
              using_sorption = true;
              nsorption_isotherms = ntracers;
	      for (int i=0; i<nrock; ++i) {
		if (sorption_isotherm_ics[r_names[i]][tNames[k]].count(str) == 0) {
		  sorption_isotherm_ics[r_names[i]][tNames[k]][str] = it->second; // set to default value
		}
	      }
	      const std::string label = str+"_"+tNames[k];
	      if (aux_chem_variables.find(label) == aux_chem_variables.end()) {
		sorption_isotherm_label_map[tNames[k]][str] = aux_chem_variables.size();
		aux_chem_variables[label]=aux_chem_variables.size()-1;
	      }
	    }
	  }
	}

	ICParmPair cation_exchange_options;
	cation_exchange_options["Cation_Exchange_Capacity"] = 0;
	{
	  for (ICParmPair::const_iterator it=cation_exchange_options.begin(); it!=cation_exchange_options.end(); ++it) {
	    const std::string& str = it->first;
	    bool found = false;
	    for (int i=0; i<nrock; ++i) {
	      const std::string prefix("rock."+r_names[i]);
	      ParmParse pprs(prefix.c_str());
	      if (pprs.countval(str.c_str())) {
		pprs.get(str.c_str(),cation_exchange_ics[r_names[i]]);
		found = true;
	      }
	    }
	    
	    if (found) {
              using_sorption = true;
              ncation_exchange = 1;
	      for (int i=0; i<nrock; ++i) {
		if (cation_exchange_ics.count(r_names[i]) == 0) {
		  cation_exchange_ics[r_names[i]] = it->second; // set to default value
		}
	      }

	      const std::string label = str;
	      if (aux_chem_variables.find(label) == aux_chem_variables.end())  {
		cation_exchange_label_map[str] = aux_chem_variables.size();
		aux_chem_variables[label]=aux_chem_variables.size()-1;
	      }
	      //std::cout << "****************** cation_exchange_ics[" << r_names[i] << "] = " 
	      //	  << cation_exchange_ics[r_names[i]] << std::endl;
	    }
	  }
	}

	ICParmPair mineralogy_options;
	mineralogy_options[      "Volume_Fraction"] = 0;
	mineralogy_options["Specific_Surface_Area"] = 0;
	for (int k=0; k<minerals.size(); ++k) {
	  for (ICParmPair::const_iterator it=mineralogy_options.begin(); it!=mineralogy_options.end(); ++it) {
	    const std::string& str = it->first;
	    bool found = false;
	    for (int i=0; i<nrock; ++i) {
              const std::string prefix("rock."+r_names[i]+".mineralogy."+minerals[k]);
	      ParmParse pprs(prefix.c_str());
	      if (pprs.countval(str.c_str())) {
		pprs.get(str.c_str(),mineralogy_ics[r_names[i]][minerals[k]][str]);
		found = true;
	      }
	    }
	    
	    if (found) {
              using_sorption = true;
	      for (int i=0; i<nrock; ++i) {
		if (mineralogy_ics[r_names[i]][minerals[k]].count(str) == 0) {
		  mineralogy_ics[r_names[i]][minerals[k]][str] = it->second; // set to default value
		}
	      }
		//std::cout << "****************** mineralogy_ics[" << r_names[i] << "][" << minerals[k] 
		//	  << "][" << str << "] = " << mineralogy_ics[r_names[i]][minerals[k]][str] 
		//	  << std::endl;

	      const std::string label = str+"_"+minerals[k];
	      if (aux_chem_variables.find(label) == aux_chem_variables.end()) {
		mineralogy_label_map[minerals[k]][str] = aux_chem_variables.size();
		aux_chem_variables[label]=aux_chem_variables.size()-1;
	      }
	    }
	  }
	}

	ICParmPair complexation_options;
	complexation_options["Site_Density"] = 0;
	for (int k=0; k<sorption_sites.size(); ++k) {
	  for (ICParmPair::const_iterator it=complexation_options.begin(); it!=complexation_options.end(); ++it) {
	    const std::string& str = it->first;
	    bool found = false;
	    for (int i=0; i<nrock; ++i) {
	      const std::string prefix("rock."+r_names[i]+".Surface_Complexation_Sites."+sorption_sites[k]);
	      ParmParse pprs(prefix.c_str());
	      if (pprs.countval(str.c_str())) {
		pprs.get(str.c_str(),surface_complexation_ics[r_names[i]][sorption_sites[k]][str]);
		found = true;
	      }
	    }
	    
	    if (found) {
              using_sorption = true;
	      for (int i=0; i<nrock; ++i) {
		if (surface_complexation_ics[r_names[i]][sorption_sites[k]].count(str) == 0) {
		  surface_complexation_ics[r_names[i]][sorption_sites[k]][str] = it->second; // set to default value
		}
	      }
	      //std::cout << "****************** surface_complexation_ics[" << r_names[i] << "][" << sorption_sites[k] 
	      //	  << "][" << str << "] = " << sorption_isotherm_ics[r_names[i]][sorption_sites[k]][str] 
	      //	  << std::endl;
	      
	      const std::string label = str+"_"+sorption_sites[k];
	      if (aux_chem_variables.find(label) == aux_chem_variables.end()) {
		surface_complexation_label_map[sorption_sites[k]][str] = aux_chem_variables.size();
		aux_chem_variables[label]=aux_chem_variables.size()-1;
	      }
	    }
	  }
	}

        if (using_sorption) 
        {
            ICParmPair sorption_chem_options; // these are domain-wide, specified per solute
            sorption_chem_options["Total_Sorbed"] = 1.e-40;
            for (int k=0; k<tNames.size(); ++k) {
                for (ICParmPair::const_iterator it=sorption_chem_options.begin(); it!=sorption_chem_options.end(); ++it) {
                    const std::string& str = it->first;
                    const std::string prefix("tracer."+tNames[k]+".Initial_Condition."+str);
                    ParmParse pprs(prefix.c_str());
                    sorption_chem_ics[tNames[k]][str] = it->second; // set to default value
                    pprs.query(str.c_str(),sorption_chem_ics[tNames[k]][str]);                      
                    //std::cout << "****************** sorption_chem_ics[" << tNames[k] 
                    //              << "][" << str << "] = " << sorption_chem_ics[tNames[k]][str] << std::endl;
                    const std::string label = str+"_"+tNames[k];
		    if (aux_chem_variables.find(label) == aux_chem_variables.end()){
		      sorption_chem_label_map[tNames[k]][str] = aux_chem_variables.size();
		      aux_chem_variables[label]=aux_chem_variables.size()-1;
		    }
                }
            }
        }
    }
    
    pp.query("Use_Shifted_Kr_Eval",use_shifted_Kr_eval);
    pp.query("Saturation_Threshold_For_Kr",saturation_threshold_for_vg_Kr);

    if (use_shifted_Kr_eval!=1 && saturation_threshold_for_vg_Kr>1) {
        if (ParallelDescriptor::IOProcessor()) {
            std::cout << "WARNING: Reducing Saturation_Threshold_For_vg_Kr to 1!" << std::endl;
        }
        saturation_threshold_for_vg_Kr = 1;
    }

    FORT_KR_INIT(&saturation_threshold_for_vg_Kr,
		 &use_shifted_Kr_eval);

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
  }

  pb.query("do_tracer_transport",do_tracer_transport);
  if (do_tracer_transport) {
      setup_tracer_transport = true; // NOTE: May want these data structures regardless...
  }

  if (setup_tracer_transport && 
      ( model==PM_STEADY_SATURATED
        || (execution_mode==INIT_TO_STEADY && switch_time<=0)
        || (execution_mode!=INIT_TO_STEADY && do_tracer_transport) ) ) {
      transport_tracers = true;
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
  pb.query("richard_subgrid_krel",richard_subgrid_krel);
  pb.query("richard_variable_switch_saturation_threshold",richard_variable_switch_saturation_threshold);
  richard_dt_thresh_pure_steady = 0.99*steady_init_time_step;
  pb.query("richard_dt_thresh_pure_steady",richard_dt_thresh_pure_steady);

  // Gravity are specified as m/s^2 in the input file
  // This is converted to the unit that is used in the code.
  if (pb.contains("gravity")) {
    pb.get("gravity",gravity);
    gravity /= BL_ONEATM;
  }

  // Get algorithmic flags and options
  pb.query("full_cycle", full_cycle);
  //pb.query("algorithm", algorithm);
  pb.query("do_multilevel_full",  do_multilevel_full );
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
    BoxLib::Abort("PorousMedia::read_prob():Must have be_cn_theta <= 1.0 && >= .5");   
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

PArray<Region>
PorousMedia::build_region_PArray(const Array<std::string>& region_names)
{
    PArray<Region> ret(region_names.size(), PArrayNoManage);
    for (int i=0; i<region_names.size(); ++i)
    {
        const std::string& name = region_names[i];
        bool found = false;
        for (int j=0; j<regions.size() && !found; ++j)
        {
            Region& r = regions[j];
            if (regions[j].name == name) {
                found = true;
                ret.set(i,&r);
            }
        }
        if (!found) {
            std::string m = "Named region not found " + name;
            BoxLib::Error(m.c_str());
        }
    }
    return ret;
}

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
      for (int i = 0; i<n_ics; i++)
      {
          const std::string& icname = ic_names[i];
	  const std::string prefix("comp.ics." + icname);
	  ParmParse ppr(prefix.c_str());
          
	  int n_ic_regions = ppr.countval("regions");
          Array<std::string> region_names;
	  ppr.getarr("regions",region_names,0,n_ic_regions);
          PArray<Region> ic_regions = build_region_PArray(region_names);

          std::string ic_type; ppr.get("type",ic_type);          
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
                std::cerr << "Insufficient number of components given for pressure refernce location" << std::endl;
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
	phys_bc.setLo(j,4);
	pres_bc.setLo(j,4);
	phys_bc.setHi(j,4);
	pres_bc.setHi(j,4);
      }	  

      for (int i = 0; i<n_bcs; i++)
      {
          const std::string& bcname = bc_names[i];
	  const std::string prefix("comp.bcs." + bcname);
	  ParmParse ppr(prefix.c_str());
          
	  int n_bc_regions = ppr.countval("regions");
          Array<std::string> region_names;
	  ppr.getarr("regions",region_names,0,n_bc_regions);
          const PArray<Region> bc_regions = build_region_PArray(region_names);
          std::string bc_type; ppr.get("type",bc_type);

          bool is_inflow = false;
          int component_bc = 4;
	  int pressure_bc  = 4;

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
              if (model == PM_STEADY_SATURATED) {
                component_bc = 2;
              } else {
                component_bc = 1;
              }
              pressure_bc = 2;

              if (model == PM_STEADY_SATURATED 
		  || (model == PM_RICHARDS && !do_richard_sat_solve)) {
                bc_array.set(i, new RegionData(bcname,bc_regions,bc_type,vals));
              } else {
                PressToRhoSat p_to_sat;
                bc_array.set(i, new Transform_S_AR_For_BC(bcname,times,vals,forms,bc_regions,
                                                          bc_type,ncomps,p_to_sat));
              }
          }
          else if (bc_type == "pressure_head")
          {              
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

            is_inflow = false;
            if (model == PM_STEADY_SATURATED) {
              component_bc = 2;
            } else {
              component_bc = 1;
            }
            pressure_bc = 2;
            bc_array.set(i, new RegionData(bcname,bc_regions,bc_type,vals[0]));//Fixme, support t-dependent
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
            if (model == PM_STEADY_SATURATED) {
              component_bc = 2;
            } else {
              component_bc = 1;
            }
            pressure_bc = 2;
            bc_array.set(i, new RegionData(bcname,bc_regions,bc_type,vals));//Fixme, support t-dependent
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
                  const std::string purpose = bc_regions[j].purpose;
                  for (int k=0; k<7; ++k) {
                      if (purpose == PMAMR::RpurposeDEF[k]) {
                          BL_ASSERT(k != 6);
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
	      bc_array.set(i,new ArrayRegionData(bcname,times,vals,forms,bc_regions,bc_type,1));
          }
          else if (bc_type == "noflow")
          {
              is_inflow = false;
              component_bc = 4;
              pressure_bc = 4;

              Array<Real> val(1,0);
              bc_array.set(i, new RegionData(bcname,bc_regions,bc_type,val));
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
              const std::string purpose = bc_regions[j].purpose;
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


void  PorousMedia::read_tracer(int do_chem)
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
          if (do_chem>0  ||  do_tracer_transport) {
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
          
          for (int n = 0; n<n_ic; n++)
          {
              const std::string prefixIC(prefix + "." + tic_names[n]);
              ParmParse ppri(prefixIC.c_str());
              int n_ic_region = ppri.countval("regions");
              Array<std::string> region_names;
              ppri.getarr("regions",region_names,0,n_ic_region);
              const PArray<Region> tic_regions = build_region_PArray(region_names);
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
              if (n_tbc <= 0)
              {
                  BoxLib::Abort("each tracer requires boundary conditions");
              }
              ppr.getarr("tbcs",tbc_names,0,n_tbc);
              tbc_array[i].resize(n_tbc,PArrayManage);
              

              Array<int> orient_types(6,-1);
              for (int n = 0; n<n_tbc; n++)
              {
                  const std::string prefixTBC(prefix + "." + tbc_names[n]);
                  ParmParse ppri(prefixTBC.c_str());
                  
                  int n_tbc_region = ppri.countval("regions");
                  Array<std::string> tbc_region_names;
                  ppri.getarr("regions",tbc_region_names,0,n_tbc_region);

                  const PArray<Region> tbc_regions = build_region_PArray(tbc_region_names);
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
                      tbc_array[i].set(n, new ArrayRegionData(tbc_names[n],times,vals,forms,tbc_regions,tbc_type,nComp));
                      AMR_BC_tID = 1; // Inflow
                  }
                  else if (tbc_type == "noflow")
                  {
                      Array<Real> val(1,0);
                      tbc_array[i].set(n, new RegionData(tbc_names[n],tbc_regions,tbc_type,val));
                      AMR_BC_tID = 4; // Noflow
                  }
                  else if (tbc_type == "outflow")
                  {
                      Array<Real> val(1,0);
                      tbc_array[i].set(n, new RegionData(tbc_names[n],tbc_regions,tbc_type,val));
                      AMR_BC_tID = 2; // Outflow
                  }
                  else {
                      std::string m = "Tracer BC: \"" + tbc_names[n] 
                          + "\": Unsupported tracer BC type: \"" + tbc_type + "\"";
                      BoxLib::Abort(m.c_str());
                  }

                  // Determine which boundary this bc is for, ensure that a unique type has been
                  // specified for each boundary
                  for (int k=0; k<tbc_region_names.size(); ++k) {
                    int iorient = -1;
                    for (int j=0; j<6; ++j) {
                      if (tbc_region_names[k] == PMAMR::RlabelDEF[j]) {
                        iorient = j;
                      }
                    }
                    if (iorient<0) {
                      BoxLib::Abort("BC givien for tracers on region that is not on boundary");
                    }
                    if (orient_types[iorient] < 0) {
                      orient_types[iorient] = AMR_BC_tID;
                    } else {
                      if (orient_types[iorient] != AMR_BC_tID) {
                        BoxLib::Abort("BC for tracers must all be of same type on each side");
                      }
                    }
                  }
              }

              // Set the default BC type = SlipWall (noflow)
              for (int k=0; k<orient_types.size(); ++k) {
                if (orient_types[k] < 0) orient_types[k] = 4;
              }

              BCRec phys_bc_trac;
              for (int i = 0; i < BL_SPACEDIM; i++) {
                phys_bc_trac.setLo(i,orient_types[i]);
                phys_bc_trac.setHi(i,orient_types[i+3]);
              }
              set_tracer_bc(trac_bc,phys_bc_trac);
          }
      }
      if (diffuse_tracers) {
        ndiff += ntracers;
      }
  }
}

void  PorousMedia::read_source()
{
  //
  // Read in parameters for sources
  //
  ParmParse pp("source");

  // determine number of sources 
  pp.query("do_source",do_source_term);

  if (do_source_term) {
      BoxLib::Abort("Sources no longer supported");
  }

#if 0
  int nsource = pp.countval("source");
  pp.query("nsource",nsource);
  if (pp.countval("source") != nsource) 
    {
      std::cerr << "Number of sources specified and listed "
		<< "do not match.\n";
      BoxLib::Abort("read_source()");
    }
  if (do_source_term > 0 && nsource > 0)
    {
      source_array.resize(nsource);
      Array<std::string> sname(nsource);
      pp.getarr("source",sname,0,nsource);

      // Get parameters for each source
      // influence function:0=constant,1=linear,2=spherical,3=exponential
      std::string buffer;
      for (int i=0; i<nsource; i++)
	{
          const std::string prefix("source." + sname[i]);
	  ParmParse ppr(prefix.c_str());
	  source_array[i].name = sname[i];
	  ppr.get("var_type",source_array[i].var_type);
	  ppr.get("var_id",source_array[i].var_id);
	  if (source_array[i].var_type == "comp")
	    {
	      source_array[i].id.resize(1);
	      for (int j=0; j<cNames.size();j++)
		{
                    if (source_array[i].var_id == cNames[j]) {
                        source_array[i].id[0] = j;
                    }
		}
	    }
	  else if (source_array[i].var_type == "tracer")
	    {
	      if (source_array[i].var_id == "ALL")
		{
		  source_array[i].id.resize(ntracers);
		  for (int j=0;j<ntracers;j++)
		    source_array[i].id[j] = j ;
		}
	      else
		{
		  source_array[i].id.resize(1);
		  for (int j=0; j<ntracers;j++)
		    {
                      if (source_array[i].var_id == tNames[j]) {
                          source_array[i].id[0] = j;
                      }
		    }
		}
	    }

	  ppr.get("regions",buffer);
          bool region_set=false;
	  for (int j=0; j<region_array.size();j++)
          {
	      if (buffer==region_array[j]->name)
              {
                  source_array[i].region = j;
                  region_set = true;
              }
          }
          BL_ASSERT(region_set);
	  ppr.get("dist_type",buffer);
	  if (!buffer.compare("constant"))
	      source_array[i].dist_type = 0;
	  else if (!buffer.compare("linear"))
	      source_array[i].dist_type = 1;
	  else if (!buffer.compare("quadratic"))
	      source_array[i].dist_type = 2;
	  else if (!buffer.compare("exponential"))
	      source_array[i].dist_type = 3;

	  ppr.getarr("val",source_array[i].val_param,
		      0,ppr.countval("val"));
	  if (ppr.countval("val")< source_array[i].id.size())
	    std::cout << "Number of values does not match the number of var_id.\n" ;
	  if (source_array[i].dist_type != 0)
	    ppr.getarr("dist_param",source_array[i].dist_param,
			0,ppr.countval("dist_param"));
	 
	}
    }
#endif
}

void  PorousMedia::read_chem()
{

  ParmParse pp("prob");

  // get Chemistry stuff
  pp.query("do_chem",do_chem);
  pp.query("do_full_strang",do_full_strang);
  pp.query("n_chem_interval",n_chem_interval);
  pp.query("ic_chem_relax_dt",ic_chem_relax_dt);
  if (n_chem_interval > 0) 
    {
      do_full_strang = 0;
    }
      
#ifdef AMANZI

  // get input file name, create SimpleThermoDatabase, process
  if (do_chem>0)
    {
      amanzi::chemistry::SetupDefaultChemistryOutput();

      ParmParse pb("prob.amanzi");
      
      std::string verbose_chemistry_init = "silent"; pb.query("verbose_chemistry_init",verbose_chemistry_init);
      
      if (verbose_chemistry_init == "silent") {
	amanzi::chemistry::chem_out->AddLevel("silent");
      }

      std::string fmt = "simple"; pb.query("Thermodynamic_Database_Format",fmt);
      pb.query("chem_database_file", amanzi_database_file);
      
      const std::string& activity_model_dh = amanzi::chemistry::ActivityModelFactory::debye_huckel;
      const std::string& activity_model_ph = amanzi::chemistry::ActivityModelFactory::pitzer_hwm;
      const std::string& activity_model_u  = amanzi::chemistry::ActivityModelFactory::unit;
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

      if (do_chem)
      {
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
                      //std::cout << "****************** solute_chem_ics[" << rname << "][" << tNames[k] 
                      //          << "][" << str << "] = " << solute_chem_ics[rname][tNames[k]][str] << std::endl;
                      const std::string label = str+"_"+tNames[k];
		      
		      if (aux_chem_variables.find(label) == aux_chem_variables.end())
		      {
			solute_chem_label_map[tNames[k]][str] = aux_chem_variables.size();
			aux_chem_variables[label]=aux_chem_variables.size()-1;
		      }
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
	  chemSolve.set(ithread, new amanzi::chemistry::SimpleThermoDatabase());
	  
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

	  chemSolve[ithread].verbosity(amanzi::chemistry::kTerse);
	  
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

  // geometry
  read_geometry();
  if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
    std::cout << "Read geometry." << std::endl;

  if (echo_inputs && ParallelDescriptor::IOProcessor()) {
      std::cout << "The Regions: " << std::endl;
      for (int i=0; i<regions.size(); ++i) {
          std::cout << regions[i] << std::endl;
      }
  }

  // components and phases
  read_comp();
  if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
    std::cout << "Read components."<< std::endl;
  
  // chem requires the number of tracers and rocks be setup before we
  // can do anything, but read_tracer and read_rock depend on do_chem
  // already being set. We'll query that now and do the remaining
  // chemistry after everything else has been read
  ParmParse pp("prob");
  pp.query("do_chem",do_chem);

  // tracers
  read_tracer(do_chem);
  if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
    std::cout << "Read tracers."<< std::endl;

  // rock
  read_rock(do_chem);
  if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
    std::cout << "Read rock."<< std::endl;

  // FIXME
  if (echo_inputs && ParallelDescriptor::IOProcessor()) {
      std::cout << "The Materials: " << std::endl;
      for (int i=0; i<materials.size(); ++i) {
        //std::cout << materials[i] << std::endl;
      }
  }

  // chemistry. Needs to come after tracers (and rock?) have been setup.
  if (verbose > 1 && ParallelDescriptor::IOProcessor())
    std::cout << "Read chemistry."<< std::endl;
  read_chem();

  // source
  //if (verbose > 1 && ParallelDescriptor::IOProcessor()) 
  //  std::cout << "Read sources."<< std::endl;
  //read_source();

  int model_int = Model();
  FORT_INITPARAMS(&ncomps,&nphases,&model_int,density.dataPtr(),
		  muval.dataPtr(),pType.dataPtr(),
		  &gravity);
    
  if (ntracers > 0)
    FORT_TCRPARAMS(&ntracers);

}

