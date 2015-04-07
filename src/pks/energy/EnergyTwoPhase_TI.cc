/*
  This is the energy component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Ethan Coon

  EnergyBase is a BDFFnBase
*/

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "boundary_function.hh"
#include "FieldEvaluator.hh"
#include "EnergyTwoPhase_PK.hh"

namespace Amanzi {
namespace Energy {

/* ******************************************************************
* Computes the non-linear functional g = g(t,u,udot)
****************************************************************** */
void EnergyTwoPhase_PK::Functional(
    double t_old, double t_new, Teuchos::RCP<TreeVector> u_old,
    Teuchos::RCP<TreeVector> u_new, Teuchos::RCP<TreeVector> g)
{
  Teuchos::OSTab tab = vo_->getOSTab();
  double h = t_new - t_old;  // get timestep

  // update BCs and conductivity
  temperature_eval_->SetFieldAsChanged(S_.ptr());
  UpdateSourceBoundaryData(t_old, t_new, *u_new->Data());
  UpdateConductivityData(S_.ptr());

  // assemble residual for diffusion operator
  op_matrix_->Init();
  op_matrix_diff_->UpdateMatrices(Teuchos::null, solution.ptr());
  op_matrix_diff_->ApplyBCs(true);

  op_matrix_->ComputeNegativeResidual(*u_new->Data(), *g->Data());

  // add accumulation term
  double dt = t_new - t_old;

  // update the energy at the new time.
  S_->GetFieldEvaluator(energy_key_)->HasFieldChanged(S_.ptr(), passwd_);

  const Epetra_MultiVector& e1 = *S_->GetFieldData(energy_key_)->ViewComponent("cell");
  const Epetra_MultiVector& e0 = *S_->GetFieldData(prev_energy_key_)->ViewComponent("cell");
  Epetra_MultiVector& g_c = *g->Data()->ViewComponent("cell");

  int nsize = g_c.MyLength();
  for (int i = 0; i < nsize; ++i) {
    g_c[0][i] += (e1[0][i] - e0[0][i]) / dt;
  }

  // advect tmp = molar_density_liquid * enthalpy 
  S_->GetFieldEvaluator(enthalpy_key_)->HasFieldChanged(S_.ptr(), passwd_);
  const CompositeVector& enthalpy = *S_->GetFieldData(enthalpy_key_);
  const CompositeVector& n_l = *S_->GetFieldData("molar_density_liquid");

  const CompositeVector& flux = *S_->GetFieldData("darcy_flux");
  op_matrix_advection_->Setup(flux);
  op_matrix_advection_->UpdateMatrices(flux);

  CompositeVector tmp(enthalpy);
  tmp.Multiply(1.0, tmp, n_l, 0.0);

  CompositeVector g_adv(g->Data()->Map());
  op_advection_->Apply(tmp, g_adv);
  g->Data()->Update(1.0, g_adv, 1.0);
}


/* ******************************************************************
* Update the preconditioner at time t and u = up
****************************************************************** */
void EnergyTwoPhase_PK::UpdatePreconditioner(
    double t, Teuchos::RCP<const TreeVector> up, double dt)
{
  Teuchos::OSTab tab = vo_->getOSTab();
  if (vo_->os_OK(Teuchos::VERB_EXTREME)) {
    *vo_->os() << "updating preconditioner, T=" << t << std::endl;
  }

  // update BCs and conductivity
  UpdateSourceBoundaryData(t, t + dt, *up->Data());
  UpdateConductivityData(S_.ptr());

  // assemble residual for diffusion operator
  op_preconditioner_->Init();
  op_preconditioner_diff_->UpdateMatrices(Teuchos::null, up->Data().ptr());
  op_preconditioner_diff_->ApplyBCs(true);

  // update with accumulation terms
  // update the accumulation derivatives, dE/dT
  S_->GetFieldEvaluator(energy_key_)->HasFieldDerivativeChanged(S_.ptr(), passwd_, "temperature");
  CompositeVector& dEdT = *S_->GetFieldData("denergy_dtemperature", energy_key_);

  if (dt > 0.0) {
    op_acc_->AddAccumulationTerm(*up->Data().ptr(), dEdT, dt, "cell");
  }

  // finalize preconditioner
  op_preconditioner_->AssembleMatrix();
  op_preconditioner_->InitPreconditioner(preconditioner_name_, *preconditioner_list_);
}


/* ******************************************************************
* TBW
****************************************************************** */
double EnergyTwoPhase_PK::ErrorNorm(Teuchos::RCP<const TreeVector> u,
                                    Teuchos::RCP<const TreeVector> du)
{
  Teuchos::OSTab tab = vo_->getOSTab();

  // Relative error in cell-centered temperature
  const Epetra_MultiVector& uc = *u->Data()->ViewComponent("cell", false);
  const Epetra_MultiVector& duc = *du->Data()->ViewComponent("cell", false);

  int cell_bad;
  double error_t(0.0);
  double ref_temp(273.0);
  for (int c = 0; c < ncells_owned; c++) {
    double tmp = fabs(duc[0][c]) / (fabs(uc[0][c] - ref_temp) + ref_temp);
    if (tmp > error_t) {
      error_t = tmp;
      cell_bad = c;
    } 
  }

  // Cell error is based upon error in energy conservation relative to
  // a characteristic energy
  double error_e(0.0);
  /*
  S_->GetFieldEvaluator(energy_key_)->HasFieldChanged(S_.ptr(), passwd_);
  const Epetra_MultiVector& energy = *S_->GetFieldData(energy_key_)->ViewComponent("cell", false);

  for (int c = 0; c != ncells_owned; ++c) {
    double tmp = std::abs(h*res_c[0][c]) / (atol_ * cv[0][c]*2.e6 + rtol_* std::abs(energy[0][c]));
    if (tmp > error_e) {
      error_e = tmp;
      cell_bad = c;
    }
  }
  */

  // Face error is mismatch in flux??


  double error = std::max(error_t, error_e);

#ifdef HAVE_MPI
  double buf = error;
  du->Data()->Comm().MaxAll(&buf, &error, 1);  // find the global maximum
#endif

  return error;
}

}  // namespace Energy
}  // namespace Amanzi

