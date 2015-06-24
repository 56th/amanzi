#include <iostream>
#include "UnitTest++.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Epetra_MpiComm.h"
#include "Epetra_Vector.h"

#include "solver_fnbase1.hh"
#include "SolverNKA.hh"
#include "SolverNKA_BT.hh"
#include "SolverNKA_BT_ATS.hh"
#include "SolverNewton.hh"
#include "SolverJFNK.hh"
#include "SolverAA.hh"

using namespace Amanzi;

SUITE(SOLVERS) {
// data structures for testing
struct test_data {
  Epetra_MpiComm *comm;
  Teuchos::RCP<Epetra_Map> map;
  Teuchos::RCP<Epetra_Vector> vec;

  test_data() {
    comm = new Epetra_MpiComm(MPI_COMM_SELF);
    map = Teuchos::rcp(new Epetra_Map(5, 0, *comm));
    vec = Teuchos::rcp(new Epetra_Vector(*map));
  }

  ~test_data() { delete comm; }
};


/* ******************************************************************/
TEST_FIXTURE(test_data, NKA_SOLVER_EXACT_JACOBIAN) {
  std::cout << "NKA nonlinear solver, exact Jacobian..." << std::endl;

  // create the function class
  Teuchos::RCP<NonlinearProblem> fn = Teuchos::rcp(new NonlinearProblem(1.0, 1.0, true));

  // create the SolverState
  Teuchos::ParameterList plist;
  plist.set("nonlinear tolerance", 1e-8);
  plist.set("diverged tolerance", 1e10);
  plist.set("limit iterations", 10);
  plist.set("max du growth factor", 1e5);
  plist.set("max divergent iterations", 3);
  plist.set("max nka vectors", 1);
  plist.sublist("VerboseObject").set("Verbosity Level", "extreme");

  // create the Solver
  Teuchos::RCP<AmanziSolvers::SolverNKA<Epetra_Vector, Epetra_BlockMap> > nka =
      Teuchos::rcp(new AmanziSolvers::SolverNKA<Epetra_Vector, Epetra_BlockMap>(plist));
  nka->Init(fn, *map);

  // initial guess
  Teuchos::RCP<Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*vec));
  (*u)[0] = -0.9;
  (*u)[1] =  0.9; 

  // solve
  nka->Solve(u);
  CHECK_CLOSE(0.0, (*u)[0], 1.0e-6);
  CHECK_CLOSE(0.0, (*u)[1], 1.0e-6);
};


/* ******************************************************************/
TEST_FIXTURE(test_data, NKA_SOLVER_INEXACT_JACOBIAN) {
  std::cout << std::endl 
            << "NKA nonlinear solver, inexact Jacobian..." << std::endl;

  // create the function class
  Teuchos::RCP<NonlinearProblem> fn = Teuchos::rcp(new NonlinearProblem(1.0, 1.0, false));

  // create the SolverState
  Teuchos::ParameterList plist;
  plist.set("nonlinear tolerance", 1e-8);
  plist.set("diverged tolerance", 1e10);
  plist.set("limit iterations", 20);
  plist.set("max du growth factor", 1e5);
  plist.set("max divergent iterations", 3);
  plist.set("max nka vectors", 2);
  plist.sublist("VerboseObject").set("Verbosity Level", "high");

  // create the Solver
  Teuchos::RCP<AmanziSolvers::SolverNKA<Epetra_Vector, Epetra_BlockMap> > nka =
      Teuchos::rcp(new AmanziSolvers::SolverNKA<Epetra_Vector, Epetra_BlockMap>(plist));
  nka->Init(fn, *map);

  // initial guess
  Teuchos::RCP<Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*vec));
  (*u)[0] = -0.9;
  (*u)[1] =  0.9;

  // solve
  nka->Solve(u);
  CHECK_CLOSE(0.0, (*u)[0], 1.e-6);
  CHECK_CLOSE(0.0, (*u)[1], 1.e-6);
};


/* ******************************************************************/
TEST_FIXTURE(test_data, NEWTON_SOLVER) {
  std::cout << std::endl << "Newton nonlinear solver..." << std::endl;

  // create the function class
  Teuchos::RCP<NonlinearProblem> fn = Teuchos::rcp(new NonlinearProblem(1.0, 1.0, true));

  // create the SolverState
  Teuchos::ParameterList plist;
  plist.set("nonlinear tolerance", 1e-6);
  plist.set("diverged tolerance", 1e10);
  plist.set("limit iterations", 15);
  plist.set("max du growth factor", 1e5);
  plist.set("max divergent iterations", 3);
  plist.sublist("VerboseObject").set("Verbosity Level", "high");

  // create the Solver
  Teuchos::RCP<AmanziSolvers::SolverNewton<Epetra_Vector, Epetra_BlockMap> > newton =
      Teuchos::rcp(new AmanziSolvers::SolverNewton<Epetra_Vector, Epetra_BlockMap>(plist));
  newton->Init(fn, *map);

  // initial guess
  Teuchos::RCP<Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*vec));
  (*u)[0] = -0.9;
  (*u)[1] =  0.9;

  // solve
  newton->Solve(u);
  CHECK_CLOSE(0.0, (*u)[0], 1.0e-6);
  CHECK_CLOSE(0.0, (*u)[1], 1.0e-6);
};


/* ******************************************************************/
TEST_FIXTURE(test_data, JFNK_SOLVER) {
  std::cout << std::endl << "JFNK nonlinear solver..." << std::endl;

  // create the function class
  Teuchos::RCP<NonlinearProblem> fn = Teuchos::rcp(new NonlinearProblem(1.0, 1.0, false));

  // create the SolverState
  Teuchos::ParameterList plist;
  plist.sublist("nonlinear solver").set("solver type", "Newton");
  plist.sublist("nonlinear solver").sublist("Newton parameters").sublist("VerboseObject")
        .set("Verbosity Level", "extreme");
  plist.sublist("nonlinear solver").sublist("Newton parameters").set("nonlinear tolerance", 1e-6);
  plist.sublist("nonlinear solver").sublist("Newton parameters").set("diverged tolerance", 1e10);
  plist.sublist("nonlinear solver").sublist("Newton parameters").set("limit iterations", 15);
  plist.sublist("nonlinear solver").sublist("Newton parameters").set("max du growth factor", 1e5);
  plist.sublist("nonlinear solver").sublist("Newton parameters").set("max divergent iterations", 3);
  plist.sublist("JF matrix parameters");
  plist.sublist("linear operator").set("iterative method", "gmres");
  plist.sublist("linear operator").sublist("gmres parameters").set("size of Krylov space", 2);
  plist.sublist("linear operator").sublist("VerboseObject").set("Verbosity Level", "extreme");

  // create the Solver
  Teuchos::RCP<AmanziSolvers::SolverJFNK<Epetra_Vector, Epetra_BlockMap> > jfnk =
      Teuchos::rcp(new AmanziSolvers::SolverJFNK<Epetra_Vector, Epetra_BlockMap>(plist));
  jfnk->Init(fn, *map);

  // initial guess
  Teuchos::RCP<Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*vec));
  (*u)[0] = -0.9;
  (*u)[1] =  0.9;

  // solve
  jfnk->Solve(u);
  CHECK_CLOSE(0.0, (*u)[0], 1.0e-6);
  CHECK_CLOSE(0.0, (*u)[1], 1.0e-6);
};


/* ******************************************************************/
TEST_FIXTURE(test_data, NKA_BT_SOLVER) {
  std::cout << std::endl << "NKA with backtracking..." << std::endl;

  // create the function class
  Teuchos::RCP<NonlinearProblem> fn = Teuchos::rcp(new NonlinearProblem(1.0, 1.0, true));

  // create the SolverState
  Teuchos::ParameterList plist;
  plist.set("nonlinear tolerance", 1e-6);
  plist.set("diverged tolerance", 1e10);
  plist.set("limit iterations", 15);
  plist.set("max du growth factor", 1e5);
  plist.set("max divergent iterations", 3);
  plist.sublist("VerboseObject").set("Verbosity Level", "high");

  // create the Solver
  Teuchos::RCP<AmanziSolvers::SolverNKA_BT<Epetra_Vector, Epetra_BlockMap> > nka_bt =
      Teuchos::rcp(new AmanziSolvers::SolverNKA_BT<Epetra_Vector, Epetra_BlockMap>(plist));
  nka_bt->Init(fn, *map);

  // initial guess
  Teuchos::RCP<Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*vec));
  (*u)[0] = -0.9;
  (*u)[1] =  0.9;

  // solve
  nka_bt->Solve(u);
  CHECK_CLOSE(0.0, (*u)[0], 1.0e-6);
  CHECK_CLOSE(0.0, (*u)[1], 1.0e-6);
};

/* ******************************************************************/
TEST_FIXTURE(test_data, NKA_BT_ATS_SOLVER) {
  std::cout << std::endl << "NKA with backtracking, ATS custom..." << std::endl;

  // create the function class
  Teuchos::RCP<NonlinearProblem> fn = Teuchos::rcp(new NonlinearProblem(1.0, 1.0, true));

  // create the SolverState
  Teuchos::ParameterList plist;
  plist.set("nonlinear tolerance", 1e-6);
  plist.set("diverged tolerance", 1e10);
  plist.set("limit iterations", 15);
  plist.set("max du growth factor", 1e5);
  plist.set("max divergent iterations", 3);
  plist.sublist("VerboseObject").set("Verbosity Level", "high");

  // create the Solver
  Teuchos::RCP<AmanziSolvers::SolverNKA_BT_ATS<Epetra_Vector, Epetra_BlockMap> > nka_bt =
      Teuchos::rcp(new AmanziSolvers::SolverNKA_BT_ATS<Epetra_Vector, Epetra_BlockMap>(plist));
  nka_bt->Init(fn, *map);

  // initial guess
  Teuchos::RCP<Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*vec));
  (*u)[0] = -0.9;
  (*u)[1] =  0.9;

  // solve
  nka_bt->Solve(u);

  std::cout<<"Solution "<<(*u)[0]<<" "<<(*u)[1]<<"\n";

  CHECK_CLOSE(0.0, (*u)[0], 1.0e-6);
  CHECK_CLOSE(0.0, (*u)[1], 1.0e-6);
};

TEST_FIXTURE(test_data, AA_SOLVER) {
  std::cout << std::endl << "AA solver...." << std::endl;

  // create the function class
  Teuchos::RCP<NonlinearProblem> fn = Teuchos::rcp(new NonlinearProblem(1.0, 1.0, true));

  // create the SolverState
  Teuchos::ParameterList plist;
  plist.set("nonlinear tolerance", 1e-7);
  plist.set("diverged tolerance", 1e10);
  plist.set("limit iterations", 15);
  plist.set("max du growth factor", 1e5);
  plist.set("max divergent iterations", 3);
  plist.set("max aa vectors", 4);
  plist.set("relaxation parameter", 1.);
  plist.sublist("VerboseObject").set("Verbosity Level", "high");

  // create the Solver
  Teuchos::RCP<AmanziSolvers::SolverAA<Epetra_Vector, Epetra_BlockMap> > aa  =
      Teuchos::rcp(new AmanziSolvers::SolverAA<Epetra_Vector, Epetra_BlockMap>(plist));
  aa->Init(fn, *map);

  // initial guess
  Teuchos::RCP<Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*vec));
  (*u)[0] = -0.95;
  (*u)[1] =  0.15;
  (*u)[2] = -0.51;
  (*u)[3] =  0.35;
  (*u)[4] = -0.54;


  // solve
  aa->Solve(u);
  CHECK_CLOSE(0.0, (*u)[0], 1.0e-6);
  CHECK_CLOSE(0.0, (*u)[1], 1.0e-6);
};


}  // SUITE



