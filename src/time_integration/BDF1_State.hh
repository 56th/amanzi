/*
  Authors: Marcus Berndt
           Ethan Coon (ecoon@lanl.gov)

  This class is based on Neil Carlson's BDF2_DAE module
  that is part of LANL's Truchas code.
*/

#ifndef AMANZI_BDF1STATE_HH_
#define AMANZI_BDF1STATE_HH_

#include <limits>

#include "errors.hh"
#include "dbc.hh"

#include "SolutionHistory.hh"

#include "TimestepController.hh"
#include "TimestepControllerFactory.hh"

namespace Amanzi {

template<class Vector>
struct BDF1_State {
  BDF1_State() {
    maxpclag = 0;
    extrapolate_guess = true;

    seq = -1;
    failed_solve = 0;
    solve_itrs = 0;

    uhist_size = 2;

    hmax = std::numeric_limits<double>::min();
    hmin = std::numeric_limits<double>::max();
  }

  // Parameters and control
  int maxpclag;  // maximum iterations that the preconditioner can be lagged
  bool extrapolate_guess;  // extrapolate forward in time or use previous
                           // step as initial guess for nonlinear solver

  // Solution history
  Teuchos::RCP<SolutionHistory<Vector> > uhist;

  // internal counters and state
  int uhist_size;  // extrapolation order for initial guess
  double hlast;  // last step size
  double hpc;  // step size built into the current preconditioner
  int pc_lag;  // counter for how many iterations the preconditioner has been lagged

  // performance counters
  int seq;  // number of steps taken
  int failed_solve;  // number of completely failed BCE steps
  double hmin, hmax;  // minimum and maximum dt used on a successful step

  // performane of nonlinear solver
  int solve_itrs;

  void InitializeFromPlist(Teuchos::ParameterList&, const Teuchos::RCP<const Vector>&);
};


/* ******************************************************************
* Initiazition of fundamental parameters
****************************************************************** */
template<class Vector>
void BDF1_State<Vector>::InitializeFromPlist(Teuchos::ParameterList& plist,
        const Teuchos::RCP<const Vector>& initvec) {
  
  //std::cout<<plist<<std::endl;
  // preconditioner lag control
  maxpclag = plist.get<int>("max preconditioner lag iterations", 0);

  // forward time extrapolation
  extrapolate_guess = plist.get<bool>("extrapolate initial guess", true);

  // solution history object
  double t0 = plist.get<double>("initial time", 0.0);
  uhist = Teuchos::rcp(new SolutionHistory<Vector>(uhist_size, t0, *initvec));

}

}  // namespace Amanzi

#endif

