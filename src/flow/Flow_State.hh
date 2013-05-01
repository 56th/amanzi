/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */
/* -------------------------------------------------------------------------
Amanzi Flow

License: see COPYRIGHT
Author: Ethan Coon

Interface layer between Flow and State, this is a harness for
accessing the new state-dev from the old Flow PK.

 ------------------------------------------------------------------------- */

#ifndef AMANZI_FLOW_STATE_NEW_HH_
#define AMANZI_FLOW_STATE_NEW_HH_

#include "PK_State.hh"

namespace Amanzi {
namespace AmanziFlow {

class Flow_State : public PK_State {

public:

  explicit Flow_State(Teuchos::RCP<AmanziMesh::Mesh> mesh);
  explicit Flow_State(Teuchos::RCP<State> S);
  explicit Flow_State(State& S);
  Flow_State(const Flow_State& other, PKStateConstructMode mode);

  virtual void Initialize();

  // access methods
  Teuchos::RCP<AmanziGeometry::Point> gravity();

  // const methods
  Teuchos::RCP<const double> fluid_density() const { return S_->GetScalarData("fluid_density"); }
  Teuchos::RCP<const double> fluid_viscosity() const {
    return S_->GetScalarData("fluid_viscosity"); }
  Teuchos::RCP<const Epetra_Vector> pressure() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("pressure")->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<const Epetra_Vector> lambda() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("pressure")->ViewComponent("face", ghosted_))(0)); }
  Teuchos::RCP<const Epetra_Vector> darcy_flux() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("darcy_flux")->ViewComponent("face", ghosted_))(0)); }
  Teuchos::RCP<const Epetra_MultiVector> darcy_velocity() const {
    return S_->GetFieldData("darcy_velocity")->ViewComponent("face", ghosted_); }

  Teuchos::RCP<const Epetra_Vector> vertical_permeability() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("permeability")->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<const Epetra_Vector> horizontal_permeability() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("permeability")->ViewComponent("cell", ghosted_))(1)); }
  Teuchos::RCP<const Epetra_Vector> porosity() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("porosity")->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<const Epetra_Vector> water_saturation() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("water_saturation")->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<const Epetra_Vector> prev_water_saturation() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("prev_water_saturation")->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<const Epetra_Vector> specific_storage() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("specific_storage")->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<const Epetra_Vector> specific_yield() const {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("specific_yield")->ViewComponent("cell", ghosted_))(0)); }

  // non-const pointers
  Teuchos::RCP<double> fluid_density() { return S_->GetScalarData("fluid_density", name_); }
  Teuchos::RCP<double> fluid_viscosity() {
    return S_->GetScalarData("fluid_viscosity", name_); }
  Teuchos::RCP<Epetra_Vector> pressure() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("pressure", name_)->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<Epetra_Vector> lambda() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("pressure", name_)->ViewComponent("face", ghosted_))(0)); }
  Teuchos::RCP<Epetra_Vector> darcy_flux() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("darcy_flux", name_)->ViewComponent("face", ghosted_))(0)); }
  Teuchos::RCP<Epetra_MultiVector> darcy_velocity() {
    return S_->GetFieldData("darcy_velocity", name_)->ViewComponent("face", ghosted_); }

  Teuchos::RCP<Epetra_Vector> vertical_permeability() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("permeability", name_)->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<Epetra_Vector> horizontal_permeability() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("permeability", name_)->ViewComponent("cell", ghosted_))(1)); }
  Teuchos::RCP<Epetra_Vector> porosity() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("porosity", name_)->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<Epetra_Vector> water_saturation() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("water_saturation", name_)->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<Epetra_Vector> prev_water_saturation() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("prev_water_saturation", name_)->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<Epetra_Vector> specific_storage() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("specific_storage", name_)->ViewComponent("cell", ghosted_))(0)); }
  Teuchos::RCP<Epetra_Vector> specific_yield() {
    return Teuchos::rcpFromRef(*(*S_->GetFieldData("specific_yield", name_)->ViewComponent("cell", ghosted_))(0)); }

  // non-const refs
  double ref_fluid_density() { return *fluid_density(); }
  double ref_fluid_viscosity() { return *fluid_viscosity(); }
  Epetra_Vector& ref_pressure() { return *pressure(); }
  Epetra_Vector& ref_lambda() { return *lambda(); }
  Epetra_Vector& ref_darcy_flux() { return *darcy_flux(); }
  Epetra_MultiVector& ref_darcy_velocity() { return *darcy_velocity(); }
  const AmanziGeometry::Point& ref_gravity() { return *gravity(); }

  Epetra_Vector& ref_vertical_permeability() { return *vertical_permeability(); }
  Epetra_Vector& ref_horizontal_permeability() { return *horizontal_permeability(); }
  Epetra_Vector& ref_porosity() { return *porosity(); }
  Epetra_Vector& ref_water_saturation() { return *water_saturation(); }
  Epetra_Vector& ref_prev_water_saturation() { return *prev_water_saturation(); }

  Epetra_Vector& ref_specific_storage() { return *specific_storage(); }
  Epetra_Vector& ref_specific_yield() { return *specific_yield(); }

  // miscaleneous
  double get_time() { return (S_ == Teuchos::null) ? -1.0 : S_->time(); }

  // debug routines
  void set_fluid_density(double rho);
  void set_fluid_viscosity(double mu);
  void set_porosity(double phi);
  void set_pressure_hydrostatic(double z0, double p0);
  void set_permeability(double Kh, double Kv);
  void set_permeability(double Kh, double Kv, const string region);
  void set_gravity(double g);
  void set_specific_storage(double ss);

 protected:
  void Construct_();

 private:
  Flow_State(const Flow_State& other);

};

} // namespace
} // namespace

#endif
