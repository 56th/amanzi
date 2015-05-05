#ifndef AMANZI_BDF1_TIME_INTEGRATOR_HH_
#define AMANZI_BDF1_TIME_INTEGRATOR_HH_

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_VerboseObjectParameterListHelpers.hpp"

#include "errors.hh"
#include "VerboseObject.hh"
#include "Solver.hh"
#include "SolverFactory.hh"

#include "BDF1_State.hh"
#include "SolutionHistory.hh"
#include "BDF1_SolverFnBase.hh"

namespace Amanzi {

template<class Vector, class VectorSpace>
class BDF1_TI {
 public:
  // Create the BDF Dae solver object, the nonlinear problem must be
  // defined in a class that derives from the virtual base class FnBase.
  BDF1_TI(BDFFnBase<Vector>& fn, Teuchos::ParameterList& plist,
          const Teuchos::RCP<const Vector>& initvector);

  // initializes the state
  void SetInitialState(const double h,
                       const Teuchos::RCP<Vector>& x,
                       const Teuchos::RCP<Vector>& xdot);

  // After a successful step, this method commits the new
  // solution to the solution history
  void CommitSolution(const double h, const Teuchos::RCP<Vector>& u);

  // Computes a step and returns true whan it fails.
  bool TimeStep(double dt, double& dt_next, const Teuchos::RCP<Vector>& x);

  // Reset the memory of the time integrator
  void Reset();

  // returns the most recent time
  double time();

  // Report statistics
  int number_nonlinear_steps() { return state_->solve_itrs; }
  void ReportStatistics_();

 protected:
  Teuchos::RCP<TimestepController> ts_control;  // timestep controller
  Teuchos::RCP<BDF1_State<Vector> > state_;

  Teuchos::RCP<AmanziSolvers::Solver<Vector,VectorSpace> > solver_;
  Teuchos::RCP<BDF1_SolverFnBase<Vector> > solver_fn_;
  Teuchos::RCP<BDFFnBase<Vector> > fn_;

  Teuchos::ParameterList plist_;
  Teuchos::RCP<VerboseObject> vo_;

  Teuchos::RCP<Vector> udot_prev_, udot_;  // for error estimate 

 private:
  double tol_solver_;  // reference solver's tolerance
};


/* ******************************************************************
* Constructor
****************************************************************** */
template<class Vector,class VectorSpace>
BDF1_TI<Vector, VectorSpace>::BDF1_TI(BDFFnBase<Vector>& fn,
                     Teuchos::ParameterList& plist,
                     const Teuchos::RCP<const Vector>& initvector) :
    plist_(plist) {
  fn_ = Teuchos::rcpFromRef(fn);

  // update the verbose options
  vo_ = Teuchos::rcp(new VerboseObject("TI::BDF1", plist_));

  // Create the state.
  state_ = Teuchos::rcp(new BDF1_State<Vector>());
  state_->InitializeFromPlist(plist_, initvector);

  // Set up the nonlinear solver
  // -- initialized the SolverFnBase interface
  solver_fn_ = Teuchos::rcp(new BDF1_SolverFnBase<Vector>(plist_, fn_));

  AmanziSolvers::SolverFactory<Vector,VectorSpace> factory;
  solver_ = factory.Create(plist_);

  solver_->Init(solver_fn_, initvector->Map());

  // Allocate memory for adaptive timestep controll
  udot_ = Teuchos::rcp(new Vector(*initvector));
  udot_prev_ = Teuchos::rcp(new Vector(*initvector));

  // timestep controller
  TimestepControllerFactory<Vector> fac;
  ts_control = fac.Create(plist, udot_, udot_prev_);

  // misc internal parameters
  tol_solver_=solver_->tolerance();
}


/* ******************************************************************
* Initialize miscaleneous parameters.
****************************************************************** */
template<class Vector,class VectorSpace>
void BDF1_TI<Vector,VectorSpace>::SetInitialState(const double t,
        const Teuchos::RCP<Vector>& x,
        const Teuchos::RCP<Vector>& xdot) {
  // set a clean initial state for when the time integrator is reinitialized
  state_->uhist->FlushHistory(t, *x, *xdot);
  state_->seq = 0;
  state_->pc_lag = 0;
}


/* ******************************************************************
* Record solution to the history.
****************************************************************** */
template<class Vector,class VectorSpace>
void BDF1_TI<Vector,VectorSpace>::CommitSolution(const double h, const Teuchos::RCP<Vector>& u) {
  double t = h + state_->uhist->MostRecentTime();

  // record the solution for later use when computing an initial guess
  // for the nonlinear solver
  state_->uhist->RecordSolution(t, *u);

  // record some information about this time step
  state_->hlast = h;
  state_->seq++;
  state_->hmin = std::min<double>(h, state_->hmin);
  state_->hmax = std::max<double>(h, state_->hmax);
}


/* ******************************************************************
* Returns most recent time.
****************************************************************** */
template<class Vector,class VectorSpace>
double BDF1_TI<Vector,VectorSpace>::time() {
  return state_->uhist->MostRecentTime();
}


/* ******************************************************************
* Implementation of implicit Euler time step.
****************************************************************** */
template<class Vector,class VectorSpace>
bool BDF1_TI<Vector,VectorSpace>::TimeStep(double dt, double& dt_next, const Teuchos::RCP<Vector>& u) {
  // initialize the output stream
  Teuchos::OSTab tab = vo_->getOSTab();

  // print info about the time step
  double tlast = state_->uhist->MostRecentTime();
  double tnew = tlast + dt;

  if (vo_->os_OK(Teuchos::VERB_HIGH)) {
    *vo_->os() << "step " << state_->seq << " T = " << tlast
               << " [sec]  dT = " << dt << std::endl;
    *vo_->os() << "preconditioner lag is " << state_->pc_lag
               << " out of " << state_->maxpclag << std::endl;
  }

  // u at the start of the time step
  Teuchos::RCP<Vector> u0 = Teuchos::rcp(new Vector(*u));

  // Predicted solution (initial value for the nonlinear solver)
  if (state_->extrapolate_guess) {
    if (state_->uhist->history_size() > 1) {
      state_->uhist->InterpolateSolution(tnew, *u);
      fn_->ChangedSolution();

      if (fn_->IsAdmissible(u)) {
    	bool changed = fn_->ModifyPredictor(dt, u0, u);
    	if (changed) fn_->ChangedSolution();
      } else {
	*u = *u0;
	fn_->ChangedSolution();
      }
    }
  }

  // Set up the solver fn.
  solver_fn_->SetTimes(tlast, tnew);
  solver_fn_->SetPreviousTimeSolution(u0);

  // Set up tolerance due to damping.
  double factor = state_->tol_multiplier;
  double tol = tol_solver_  * factor;
  solver_->set_tolerance(tol);

  if (factor > 1.0) {
    if (vo_->os_OK(Teuchos::VERB_MEDIUM)) {
      *vo_->os() << "modified tolerance=" << tol << std::endl;
    }
  }

  // Solve the nonlinear BCE system.
  int ierr, code, itr;
  try {
    ierr = solver_->Solve(u);
    itr = solver_->num_itrs();
    code = solver_->returned_code();
  } catch (const Errors::CutTimeStep& e) {
    ierr = 1;
    itr = -1;  // This should not be summed up into the global counter.
    code = AmanziSolvers::SOLVER_INTERNAL_EXCEPTION;
  }

  if (ierr == 0) {
    if (vo_->os_OK(Teuchos::VERB_HIGH)) {
      *vo_->os() << "success: " << itr << " nonlinear itrs" 
                 << " error=" << solver_->residual() << std::endl;
    }
  } else {
    if (vo_->os_OK(Teuchos::VERB_HIGH)) {
      *vo_->os() << vo_->color("red") << "step failed with error code " << code << vo_->reset() << std::endl;
    }
    *u = *u0;
  }

  // update the next timestep size
  if (ierr != 0) itr = -1;
  dt_next = ts_control->get_timestep(dt, itr);

  // update the preconditioner lag and tolerance multiplier
  if (ierr != 0) {
    state_->pc_lag = 0;
  } else {
    if (dt_next > dt) {
      state_->pc_lag = std::min(state_->pc_lag + 1, state_->maxpclag);
      state_->tol_multiplier= std::max(1.0, factor * state_->tol_multiplier_damp);
    } else if (dt_next < dt) {
      state_->pc_lag = std::max(state_->pc_lag - 1, 0);
    }
  }
  solver_->set_pc_lag(state_->pc_lag);

  // update performance statistics
  state_->solve_itrs += solver_->num_itrs();
  state_->pc_updates += solver_->pc_updates();
  state_->pc_calls += solver_->pc_calls();

  if (ierr != 0) {
    state_->failed_solve++;
  } else {
    state_->hmax = std::max(state_->hmax, dt);
    state_->hmin = std::min(state_->hmin, dt);

    if (state_->uhist->history_size() > 1) {
      *udot_prev_ = *udot_;
      double tmp = 1.0 / dt;
      *udot_ = *u;
      udot_->Update(-tmp, *u0, tmp);
    }

    ReportStatistics_();
  }
  return (ierr != 0);  // Returns true when it fails.
}


/* ******************************************************************
* Report statistics.
****************************************************************** */
template<class Vector,class VectorSpace>
void BDF1_TI<Vector,VectorSpace>::ReportStatistics_()
{
  if (vo_->os_OK(Teuchos::VERB_HIGH)) {
    std::ostringstream oss;
    oss.flush();
    oss.setf(std::ios::scientific, std::ios::floatfield);

    oss << "TS:" << std::right << state_->seq;
    oss << " FS:" << state_->failed_solve;
    oss << " NS:" << state_->solve_itrs;
    oss << " PC:" << state_->pc_updates << " " << state_->pc_calls;

    oss << " dt:";
    oss.precision(4);
    oss.width(10);
    oss << state_->hmin << " " << state_->hmax;

    *vo_->os() << oss.str() << std::endl;
  }  
}

}  // namespace Amanzi

#endif
