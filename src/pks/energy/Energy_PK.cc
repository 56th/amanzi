/*
  This is the energy component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"

#include "GMVMesh.hh"
#include "Mesh.hh"
#include "mfd3d.hh"
#include "primary_variable_field_evaluator.hh"
#include "State.hh"

#include "Energy_BC_Factory.hh"
#include "Energy_PK.hh"

namespace Amanzi {
namespace Energy {

/* ******************************************************************
* Default constructor for Energy PK.
****************************************************************** */
Energy_PK::Energy_PK(const Teuchos::RCP<Teuchos::ParameterList>& glist,
                     Teuchos::RCP<State> S) :
    glist_(glist),
    vo_(NULL),
    passwd_("thermal")
{
  S_ = S;
  mesh_ = S->GetMesh();
  dim = mesh_->space_dimension();

  ncells_owned = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED);
  ncells_wghost = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::USED);

  nfaces_owned = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::OWNED);
  nfaces_wghost = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::USED);

  energy_key_ = "energy";
  prev_energy_key_ = "prev_energy";
  enthalpy_key_ = "enthalpy";
  conductivity_key_ = "thermal_conductivity";
}


/* ******************************************************************
* Construction of PK global variables.
****************************************************************** */
void Energy_PK::Setup()
{
  // require first-requested state variables
  if (!S_->HasField("atmospheric_pressure")) {
    S_->RequireScalar("atmospheric_pressure", passwd_);
  }

  // require primary state variables
  std::vector<std::string> names(2);
  names[0] = "cell";
  names[1] = "face";
 
  std::vector<AmanziMesh::Entity_kind> locations(2);
  locations[0] = AmanziMesh::CELL;
  locations[1] = AmanziMesh::FACE;
 
  std::vector<int> ndofs(2, 1);
  
  if (!S_->HasField("temperature")) {
    S_->RequireField("temperature", passwd_)->SetMesh(mesh_)->SetGhosted(true)
      ->SetComponents(names, locations, ndofs);

    Teuchos::ParameterList elist;
    elist.set<std::string>("evaluator name", "temperature");
    temperature_eval_ = Teuchos::rcp(new PrimaryVariableFieldEvaluator(elist));
    S_->SetFieldEvaluator("temperature", temperature_eval_);
  }

  // conserved quantity from the last time step.
  if (!S_->HasField("prev_energy")) {
    S_->RequireField("prev_energy", passwd_)->SetMesh(mesh_)->SetGhosted(true)
      ->SetComponent("cell", AmanziMesh::CELL, 1);
    S_->GetField("prev_energy", passwd_)->set_io_vis(false);
  }

  // Fields for energy as independent PK
  if (!S_->HasField("darcy_flux")) {
    S_->RequireField("darcy_flux", passwd_)->SetMesh(mesh_)->SetGhosted(true)
      ->SetComponent("face", AmanziMesh::FACE, 1);
  }
}


/* ******************************************************************
* Basic initialization of energy classes.
****************************************************************** */
void Energy_PK::Initialize()
{
  Teuchos::RCP<Teuchos::ParameterList> pk_list = Teuchos::sublist(glist_, "PKs", true);
  Teuchos::RCP<Teuchos::ParameterList> ep_list = Teuchos::sublist(pk_list, "Energy", true);

  // Create BCs objects.
  bc_model_.resize(nfaces_wghost, 0);
  bc_submodel_.resize(nfaces_wghost, 0);
  bc_value_.resize(nfaces_wghost, 0.0);
  bc_mixed_.resize(nfaces_wghost, 0.0);

  Teuchos::RCP<Teuchos::ParameterList>
      bc_list = Teuchos::rcp(new Teuchos::ParameterList(ep_list->sublist("boundary conditions", true)));
  EnergyBCFactory bc_factory(mesh_, bc_list);

  bc_temperature = bc_factory.CreateTemperature(bc_submodel_);
  bc_flux = bc_factory.CreateEnergyFlux(bc_submodel_);

  op_bc_ = Teuchos::rcp(new Operators:: BCs(Operators::OPERATOR_BC_TYPE_FACE, bc_model_, bc_value_, bc_mixed_));
}


/* ****************************************************************
* This completes initialization of missed fields in the state.
* This is useful for unit tests.
**************************************************************** */
void Energy_PK::InitializeFields_()
{
  Teuchos::OSTab tab = vo_->getOSTab();

  if (S_->HasField(prev_energy_key_)) {
    if (!S_->GetField(prev_energy_key_, passwd_)->initialized()) {
      temperature_eval_->SetFieldAsChanged(S_.ptr());
      S_->GetFieldEvaluator(energy_key_)->HasFieldChanged(S_.ptr(), passwd_);

      const CompositeVector& e1 = *S_->GetFieldData(energy_key_);
      CompositeVector& e0 = *S_->GetFieldData(prev_energy_key_, passwd_);
      e0 = e1;

      S_->GetField(prev_energy_key_, passwd_)->set_initialized();

      if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
          *vo_->os() << "initilized prev_energy to previous energy" << std::endl;  
    }
  }

  if (S_->GetField("darcy_flux")->owner() == passwd_) {
    if (!S_->GetField("darcy_flux", passwd_)->initialized()) {
      S_->GetFieldData("darcy_flux", passwd_)->PutScalar(0.0);
      S_->GetField("darcy_flux", passwd_)->set_initialized();

      if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
          *vo_->os() << "initilized darcy_flux to default value 0.0" << std::endl;  
    }
  }
}


/* ******************************************************************
* Transfer part of the internal data needed by energy PK in the next
* time step.
****************************************************************** */
void Energy_PK::CommitStep(double t_old, double t_new)
{
  // energy -> prev_prev_energy
  S_->GetFieldEvaluator(energy_key_)->HasFieldChanged(S_.ptr(), passwd_);
  const Epetra_MultiVector& e = *S_->GetFieldData(energy_key_)->ViewComponent("cell");
  Epetra_MultiVector& e_prev = *S_->GetFieldData(prev_energy_key_, passwd_)->ViewComponent("cell");
  e_prev = e;
}


/* ******************************************************************
* TBW.
****************************************************************** */
bool Energy_PK::UpdateConductivityData(const Teuchos::Ptr<State>& S)
{
  bool update = S->GetFieldEvaluator(conductivity_key_)->HasFieldChanged(S, passwd_);
  if (update) {
    const Epetra_MultiVector& conductivity = *S->GetFieldData(conductivity_key_)->ViewComponent("cell");
    WhetStone::Tensor Ktmp(dim, 1);

    K.clear();
    for (int c = 0; c < ncells_owned; c++) {
      Ktmp(0, 0) = conductivity[0][c];
      K.push_back(Ktmp);
    } 
  }
  return update;
}


/* ******************************************************************
* A wrapper for updating boundary conditions.
****************************************************************** */
void Energy_PK::UpdateSourceBoundaryData(double t_old, double t_new, const CompositeVector& u)
{
  bc_temperature->Compute(t_new);
  bc_flux->Compute(t_new);

  ComputeBCs(u);
}


/* ******************************************************************
* Add a boundary marker to used faces.
* WARNING: we can skip update of ghost boundary faces, b/c they 
* should be always owned. 
****************************************************************** */
void Energy_PK::ComputeBCs(const CompositeVector& u)
{
  const Epetra_MultiVector& u_cell = *u.ViewComponent("cell");
  
  for (int n = 0; n < bc_model_.size(); n++) {
    bc_model_[n] = Operators::OPERATOR_BC_NONE;
    bc_value_[n] = 0.0;
    bc_mixed_[n] = 0.0;
  }

  EnergyBoundaryFunction::Iterator bc;
  for (bc = bc_temperature->begin(); bc != bc_temperature->end(); ++bc) {
    int f = bc->first;
    bc_model_[f] = Operators::OPERATOR_BC_DIRICHLET;
    bc_value_[f] = bc->second;
  }

  for (bc = bc_flux->begin(); bc != bc_flux->end(); ++bc) {
    int f = bc->first;
    bc_model_[f] = Operators::OPERATOR_BC_NEUMANN;
    bc_value_[f] = bc->second;
  }

  dirichlet_bc_faces_ = 0;
  for (int f = 0; f < nfaces_owned; ++f) {
    if (bc_model_[f] == Operators::OPERATOR_BC_DIRICHLET) dirichlet_bc_faces_++;
  }
  int flag_essential_bc = (dirichlet_bc_faces_ > 0) ? 1 : 0;

  // verify that the algebraic problem is consistent
#ifdef HAVE_MPI
  int flag = flag_essential_bc;
  mesh_->get_comm()->MaxAll(&flag, &flag_essential_bc, 1);  // find the global maximum
#endif
  if (! flag_essential_bc && vo_->getVerbLevel() >= Teuchos::VERB_LOW) {
    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "WARNING: no essential boundary conditions, solver may fail" << std::endl;
  }
}

}  // namespace Energy
}  // namespace Amanzi

