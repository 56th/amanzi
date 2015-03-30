/*
  This is the flow component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)

  Field evaluator for total volumetric water content which is the 
  conserved quantity in Richards's equation.

  Constant water density.
*/

#include "CommonDefs.hh"
#include "VWContentEvaluator_ConstDensity.hh"

namespace Amanzi {
namespace Flow {

/* ******************************************************************
* Constructor.
****************************************************************** */
VWContentEvaluator_ConstDensity::VWContentEvaluator_ConstDensity(Teuchos::ParameterList& plist) :
    VWContentEvaluator(plist) {};


/* ******************************************************************
* Initialization.
****************************************************************** */
void VWContentEvaluator_ConstDensity::Init_()
{
  my_key_ = std::string("water_content");

  dependencies_.insert(std::string("porosity"));
  dependencies_.insert(std::string("saturation_liquid"));

  vapor_phase_ = plist_.get<bool>("water vapor phase", false);
  if (vapor_phase_) {
    dependencies_.insert(std::string("molar_density_gas"));
    dependencies_.insert(std::string("molar_fraction_gas"));
  }
}


/* ******************************************************************
* Required member: field calculation.
****************************************************************** */
void VWContentEvaluator_ConstDensity::EvaluateField_(
    const Teuchos::Ptr<State>& S, const Teuchos::Ptr<CompositeVector>& result)
{
  S->GetFieldEvaluator("saturation_liquid")->HasFieldChanged(S.ptr(), "flow");
  const Epetra_MultiVector& s_l = *S->GetFieldData("saturation_liquid")->ViewComponent("cell");
  const Epetra_MultiVector& phi = *S->GetFieldData("porosity")->ViewComponent("cell");

  double rho = *S->GetScalarData("fluid_density");
  double n_l = rho / CommonDefs::MOLAR_MASS_H2O;

  Epetra_MultiVector& result_v = *result->ViewComponent("cell");

  if (vapor_phase_) {
    const Epetra_MultiVector& n_g = *S->GetFieldData("molar_density_gas")->ViewComponent("cell");
    const Epetra_MultiVector& mlf_g = *S->GetFieldData("molar_fraction_gas")->ViewComponent("cell");
    
    int ncells = result->size("cell", false);
    for (int c = 0; c != ncells; ++c) {
      result_v[0][c] = phi[0][c] * (s_l[0][c] * n_l + (1.0 - s_l[0][c]) * n_g[0][c] * mlf_g[0][c]);
    }
  } else {
    int ncells = result->size("cell", false);
    for (int c = 0; c != ncells; ++c) {
      result_v[0][c] = phi[0][c] * s_l[0][c] * n_l;
    }
  }      
}


/* ******************************************************************
* Required member: field calculation.
****************************************************************** */
void VWContentEvaluator_ConstDensity::EvaluateFieldPartialDerivative_(
    const Teuchos::Ptr<State>& S,
    Key wrt_key, const Teuchos::Ptr<CompositeVector>& result)
{
  const Epetra_MultiVector& s_l = *S->GetFieldData("saturation_liquid")->ViewComponent("cell");
  const Epetra_MultiVector& phi = *S->GetFieldData("porosity")->ViewComponent("cell");

  double rho = *S->GetScalarData("fluid_density");
  double n_l = rho / CommonDefs::MOLAR_MASS_H2O;

  Epetra_MultiVector& result_v = *result->ViewComponent("cell");

  if (vapor_phase_) {
    const Epetra_MultiVector& n_g = *S->GetFieldData("molar_density_gas")->ViewComponent("cell");
    const Epetra_MultiVector& mlf_g = *S->GetFieldData("molar_fraction_gas")->ViewComponent("cell");

    int ncells = result->size("cell", false);
    if (wrt_key == "porosity") {
      for (int c = 0; c != ncells; ++c) {
        result_v[0][c] = (s_l[0][c] * n_l + (1.0 - s_l[0][c]) * n_g[0][c] * mlf_g[0][c]);
      }
    } else if (wrt_key == "saturation_liquid") {
      for (int c = 0; c != ncells; ++c) {
        result_v[0][c] = phi[0][c] * n_l;
      }
    } else if (wrt_key == "molar_density_gas") {
      for (int c = 0; c != ncells; ++c) {
        result_v[0][c] = phi[0][c] * (1.0 - s_l[0][c]) * mlf_g[0][c];
      }
    } else if (wrt_key == "molar_fraction_gas") {
      for (int c = 0; c != ncells; ++c) {
        result_v[0][c] = phi[0][c] * (1.0 - s_l[0][c]) * n_g[0][c];
      }
    } else {
      ASSERT(0);
    }
    
  } else {
    int ncells = result->size("cell", false);
    if (wrt_key == "porosity") {
      for (int c = 0; c != ncells; ++c) {
        result_v[0][c] = s_l[0][c] * n_l;
      }
    } else if (wrt_key == "saturation_liquid") {
      for (int c = 0; c != ncells; ++c) {
        result_v[0][c] = phi[0][c] * n_l;
      }
    } else {
      ASSERT(0);
    }
  }
}

}  // namespace Flow
}  // namespace Amanzi
