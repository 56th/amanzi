/*
This is the Nonlinear Solver component of the Amanzi code. 
License: BSD
Authors: Ethan Coon (ecoon@lanl.gov)
         Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_SOLVER_DEFS_HH_
#define AMANZI_SOLVER_DEFS_HH_

namespace Amanzi {
namespace AmanziSolvers {

enum ConvergenceMonitor {
     SOLVER_MONITOR_UPDATE = 0,
     SOLVER_MONITOR_PCED_RESIDUAL = 1,
     SOLVER_MONITOR_RESIDUAL = 2
};

enum BacktrackMonitor {
  BT_MONITOR_ENORM,   // accept decrease in the ENORM
  BT_MONITOR_L2,      // accept decrease in the Linf of the ConvergenceMonitor (residual)
  BT_MONITOR_EITHER   // accept decrease in either of the above
};

const int SOLVER_CONTINUE = 1;
const int SOLVER_CONVERGED = 0;

const int SOLVER_MAX_ITERATIONS = -1;
const int SOLVER_OVERFLOW = -2;
const int SOLVER_STAGNATING = -3;
const int SOLVER_DIVERGING = -4;
const int SOLVER_INADMISSIBLE_SOLUTION = -5;
const int SOLVER_INTERNAL_EXCEPTION = -6;
const int SOLVER_BAD_SEARCH_DIRECTION = -7;

const double BACKTRACKING_GOOD_REDUCTION = 0.5;
const int BACKTRACKING_USED = 1;
const int BACKTRACKING_MAX_ITERATIONS = 4;
const int BACKTRACKING_ROUNDOFF_PROBLEM = 8;

}  // namespace AmanziSolvers
}  // namespace Amanzi
 
#endif

