/*
  Factory for nonlinear solvers.

  License: BSD
  Authors: Ethan Coon (ecoon@lanl.gov)

*/

#ifndef AMANZI_SOLVER_FACTORY_HH_
#define AMANZI_SOLVER_FACTORY_HH_

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "errors.hh"
#include "Solver.hh"

namespace Amanzi {
namespace AmanziSolvers {

template<class Vector,class VectorSpace>
struct SolverFactory {
 public:
  Teuchos::RCP<Solver<Vector,VectorSpace> >
  Create(const std::string& name,
         const Teuchos::ParameterList& solver_list);

  Teuchos::RCP<Solver<Vector,VectorSpace> >
  Create(Teuchos::ParameterList& solver_list);
};

} // namespace
} // namespace

#include "SolverNKA.hh"
#include "SolverNKA_BT.hh"
#include "SolverNKA_BT_ATS.hh"
#include "SolverNewton.hh"
#include "SolverJFNK.hh"
#include "SolverContinuation.hh"

namespace Amanzi {
namespace AmanziSolvers {

/* ******************************************************************
* Initialization of the Solver
****************************************************************** */
template<class Vector,class VectorSpace>
Teuchos::RCP<Solver<Vector, VectorSpace> >
SolverFactory<Vector,VectorSpace>::Create(
    const std::string& name, const Teuchos::ParameterList& solver_list)
{
  if (solver_list.isSublist(name)) {
    Teuchos::ParameterList slist = solver_list.sublist(name);
    return Create(slist);
  } else {
    std::stringstream estream;
    estream << "SolverFactory: nonexistent solver sublist \"" << name << "\"";
    Errors::Message msg(estream.str());
    Exceptions::amanzi_throw(msg);
  }
}


/* ******************************************************************
* Initialization of the solver
****************************************************************** */
template<class Vector,class VectorSpace>
Teuchos::RCP<Solver<Vector, VectorSpace> >
SolverFactory<Vector, VectorSpace>::Create(Teuchos::ParameterList& slist)
{
  if (slist.isParameter("solver type")) {
    std::string type = slist.get<std::string>("solver type");

    if (type == "nka") {
      if (!slist.isSublist("nka parameters")) {
        Errors::Message msg("SolverFactory: missing sublist \"nka parameters\"");
        Exceptions::amanzi_throw(msg);
      }
      Teuchos::ParameterList nka_list = slist.sublist("nka parameters");
      Teuchos::RCP<Solver<Vector,VectorSpace> > solver =
          Teuchos::rcp(new SolverNKA<Vector,VectorSpace>(nka_list));
      return solver;
    } else if (type == "Newton") {
      if (!slist.isSublist("Newton parameters")) {
        Errors::Message msg("SolverFactory: missing sublist \"Newton parameters\"");
        Exceptions::amanzi_throw(msg);
      }
      Teuchos::ParameterList newton_list = slist.sublist("Newton parameters");
      Teuchos::RCP<Solver<Vector,VectorSpace> > solver =
          Teuchos::rcp(new SolverNewton<Vector,VectorSpace>(newton_list));
      return solver;
    } else if (type == "nka_bt") {
      if (!slist.isSublist("nka_bt parameters")) {
        Errors::Message msg("SolverFactory: missing sublist \"nka_bt parameters\"");
        Exceptions::amanzi_throw(msg);
      }
      Teuchos::ParameterList nka_list = slist.sublist("nka_bt parameters");
      Teuchos::RCP<Solver<Vector,VectorSpace> > solver =
          Teuchos::rcp(new SolverNKA_BT<Vector,VectorSpace>(nka_list));
      return solver;
    } else if (type == "nka_bt_ats") {
      if (!slist.isSublist("nka_bt_ats parameters")) {
        Errors::Message msg("SolverFactory: missing sublist \"nka_bt_ats parameters\"");
        Exceptions::amanzi_throw(msg);
      }
      Teuchos::ParameterList nka_list = slist.sublist("nka_bt_ats parameters");
      Teuchos::RCP<Solver<Vector,VectorSpace> > solver =
          Teuchos::rcp(new SolverNKA_BT_ATS<Vector,VectorSpace>(nka_list));
      return solver;
    } else if (type == "JFNK") {
      if (!slist.isSublist("JFNK parameters")) {
        Errors::Message msg("SolverFactory: missing sublist \"JFNK parameters\"");
        Exceptions::amanzi_throw(msg);
      }
      Teuchos::ParameterList jfnk_list = slist.sublist("JFNK parameters");
      Teuchos::RCP<Solver<Vector,VectorSpace> > solver =
          Teuchos::rcp(new SolverJFNK<Vector,VectorSpace>(jfnk_list));
      return solver;
    } else if (type == "continuation") {
      if (!slist.isSublist("continuation parameters")) {
        Errors::Message msg("SolverFactory: missing sublist \"continuation parameters\"");
        Exceptions::amanzi_throw(msg);
      }
      Teuchos::ParameterList cont_list = slist.sublist("continuation parameters");
      Teuchos::RCP<Solver<Vector,VectorSpace> > solver =
          Teuchos::rcp(new SolverContinuation<Vector,VectorSpace>(cont_list));
      return solver;
    } else {
      Errors::Message msg("SolverFactory: wrong value of parameter `\"solver type`\"");
      Exceptions::amanzi_throw(msg);
    }
  } else {
    Errors::Message msg("SolverFactory: parameter `\"solver type`\" is missing");
    Exceptions::amanzi_throw(msg);
  }
  return Teuchos::null;
}

}  // namespace AmanziSolvers
}  // namespace Amanzi

#endif
