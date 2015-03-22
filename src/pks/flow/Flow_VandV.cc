/*
  This is the flow component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <set>

#include "errors.hh"
#include "OperatorDefs.hh"

#include "Flow_PK.hh"

namespace Amanzi {
namespace Flow {

/* ******************************************************************
* TODO: Verify that a BC has been applied to every boundary face.
* Right now faces without BC are considered no-mass-flux.
****************************************************************** */
void Flow_PK::VV_ValidateBCs() const
{
  // Create sets of the face indices belonging to each BC type.
  std::set<int> pressure_faces, head_faces, flux_faces;
  FlowBoundaryFunction::Iterator bc;
  for (bc = bc_pressure->begin(); bc != bc_pressure->end(); ++bc) pressure_faces.insert(bc->first);
  for (bc = bc_head->begin(); bc != bc_head->end(); ++bc) head_faces.insert(bc->first);
  for (bc = bc_flux->begin(); bc != bc_flux->end(); ++bc) flux_faces.insert(bc->first);

  std::set<int> overlap;
  std::set<int>::iterator overlap_end;
  int local_overlap, global_overlap;

  // Check for overlap between pressure and static head BC.
  std::set_intersection(pressure_faces.begin(), pressure_faces.end(),
                        head_faces.begin(), head_faces.end(),
                        std::inserter(overlap, overlap.end()));
  local_overlap = overlap.size();
  mesh_->get_comm()->SumAll(&local_overlap, &global_overlap, 1);  // this will over count ghost faces

  if (global_overlap != 0) {
    Errors::Message msg;
    std::stringstream s;
    s << global_overlap;
    msg << "Flow PK: static head BC overlap Dirichlet BC on "
        << s.str().c_str() << " faces\n";
    Exceptions::amanzi_throw(msg);
  }

  // Check for overlap between pressure and flux BC.
  overlap.clear();
  std::set_intersection(pressure_faces.begin(), pressure_faces.end(),
                        flux_faces.begin(), flux_faces.end(),
                        std::inserter(overlap, overlap.end()));
  local_overlap = overlap.size();
  mesh_->get_comm()->SumAll(&local_overlap, &global_overlap, 1);  // this will over count ghost faces

  if (global_overlap != 0) {
    Errors::Message msg;
    std::stringstream s;
    s << global_overlap;
    msg << "Flow PK: flux BC overlap Dirichlet BC on "
        << s.str().c_str() << " faces\n";
    Exceptions::amanzi_throw(msg);
  }

  // Check for overlap between static head and flux BC.
  overlap.clear();
  std::set_intersection(head_faces.begin(), head_faces.end(),
                        flux_faces.begin(), flux_faces.end(),
                        std::inserter(overlap, overlap.end()));
  local_overlap = overlap.size();
  mesh_->get_comm()->SumAll(&local_overlap, &global_overlap, 1);  // this will over count ghost faces

  if (global_overlap != 0) {
    Errors::Message msg;
    std::stringstream s;
    s << global_overlap;
    msg << "Flow PK: flux BC overlap static head BC on "
        << s.str().c_str() << " faces\n";
    Exceptions::amanzi_throw(msg);
  }
}


/* *******************************************************************
* Reports water balance.
******************************************************************* */
void Flow_PK::VV_ReportWaterBalance(const Teuchos::Ptr<State>& S) const
{
  const Epetra_MultiVector& phi = *S->GetFieldData("porosity")->ViewComponent("cell", false);
  const Epetra_MultiVector& flux = *S->GetFieldData("darcy_flux")->ViewComponent("face", true);
  const Epetra_MultiVector& ws = *S->GetFieldData("saturation_liquid")->ViewComponent("cell", false);

  double mass_bc_dT = WaterVolumeChangePerSecond(bc_model, flux) * rho_ * dt_;

  double mass_amanzi = 0.0;
  for (int c = 0; c < ncells_owned; c++) {
    mass_amanzi += ws[0][c] * rho_ * phi[0][c] * mesh_->cell_volume(c);
  }

  double mass_amanzi_tmp = mass_amanzi, mass_bc_tmp = mass_bc_dT;
  mesh_->get_comm()->SumAll(&mass_amanzi_tmp, &mass_amanzi, 1);
  mesh_->get_comm()->SumAll(&mass_bc_tmp, &mass_bc_dT, 1);

  mass_bc += mass_bc_dT;

  Teuchos::OSTab tab = vo_->getOSTab();
  *vo_->os() << "reservoir water mass=" << mass_amanzi 
             << " [kg], total influx=" << mass_bc << " [kg]" << std::endl;
}

 
/* *******************************************************************
* Calculate flow out of the current seepage face.
******************************************************************* */
void Flow_PK::VV_ReportSeepageOutflow(const Teuchos::Ptr<State>& S) const
{
  const Epetra_MultiVector& flux = *S->GetFieldData("darcy_flux")->ViewComponent("face");

  int dir, f, c;
  double tmp, outflow(0.0);
  FlowBoundaryFunction::Iterator bc;

  for (bc = bc_seepage->begin(); bc != bc_seepage->end(); ++bc) {
    f = bc->first;
    if (f < nfaces_owned) {
      c = BoundaryFaceGetCell(f);
      const AmanziGeometry::Point& normal = mesh_->face_normal(f, false, c, &dir);
      tmp = flux[0][f] * dir;
      if (tmp > 0.0) outflow += tmp;
    }
  }

  tmp = outflow;
  mesh_->get_comm()->SumAll(&tmp, &outflow, 1);

  outflow *= rho_;
  seepage_mass_ += outflow * dt_;

  if (MyPID == 0 && bc_seepage->global_size() > 0) {
    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "seepage face: flow=" << outflow << " [kg/s]," 
               << " total=" << seepage_mass_ << " [kg]" << std::endl;
  }
}


/* *******************************************************************
* Calculates best least square fit for data (h[i], error[i]).                       
******************************************************************* */
void Flow_PK::VV_PrintHeadExtrema(const CompositeVector& pressure) const
{
  double hmin(1.4e+9), hmax(-1.4e+9);  // diameter of the Sun
  double rho_g = rho_ * fabs(gravity_[dim - 1]);
  for (int f = 0; f < nfaces_owned; f++) {
    if (bc_model[f] == Operators::OPERATOR_BC_DIRICHLET) {
      double z = mesh_->face_centroid(f)[dim - 1]; 
      double h = z + (bc_value[f] - atm_pressure_) / rho_g;
      hmax = std::max(hmax, h);
      hmin = std::min(hmin, h);
    }
  }
  double tmp = hmin;  // global extrema
  mesh_->get_comm()->MinAll(&tmp, &hmin, 1);
  tmp = hmax;
  mesh_->get_comm()->MaxAll(&tmp, &hmax, 1);

  Teuchos::OSTab tab = vo_->getOSTab();
  *vo_->os() << "boundary head (BCs): min=" << hmin << " max=" << hmax << " [m]" << std::endl;

  // process cell-based quantaties
  const Epetra_MultiVector& pcells = *pressure.ViewComponent("cell");
  double vmin(1.4e+9), vmax(-1.4e+9);
  for (int c = 0; c < ncells_owned; c++) {
    double z = mesh_->cell_centroid(c)[dim - 1];              
    double h = z + (pcells[0][c] - atm_pressure_) / rho_g;
    vmax = std::max(vmax, h);
    vmin = std::min(vmin, h);
  }
  tmp = vmin;  // global extrema
  mesh_->get_comm()->MinAll(&tmp, &vmin, 1);
  tmp = vmax;
  mesh_->get_comm()->MaxAll(&tmp, &vmax, 1);
  *vo_->os() << "domain head (cells): min=" << vmin << " max=" << vmax << " [m]" << std::endl;

  // process face-based quantaties (if any)
  if (pressure.HasComponent("face")) {
    const Epetra_MultiVector& pface = *pressure.ViewComponent("face");

    for (int f = 0; f < nfaces_owned; f++) {
      double z = mesh_->face_centroid(f)[dim - 1];              
      double h = z + (pface[0][f] - atm_pressure_) / rho_g;
      vmax = std::max(vmax, h);
      vmin = std::min(vmin, h);
    }
    tmp = vmin;  // global extrema
    mesh_->get_comm()->MinAll(&tmp, &vmin, 1);
    tmp = vmax;
    mesh_->get_comm()->MaxAll(&tmp, &vmax, 1);
    *vo_->os() << "domain head (cells + faces): min=" << vmin << " max=" << vmax << " [m]" << std::endl;
  }
}

 
/* ****************************************************************
* Find string for the preconditoner.
**************************************************************** */
void Flow_PK::OutputTimeHistory(
    const Teuchos::ParameterList& plist, std::vector<dt_tuple>& dT_history)
{
  if (plist.isParameter("plot time history") && 
      vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM) {
    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "saving time history in file flow_dt_history.txt..." << std::endl;

    char file_name[30];
    sprintf(file_name, "flow_dt_history_%d.txt", ti_phase_counter++);

    std::ofstream ofile;
    ofile.open(file_name);

    for (double n = 0; n < dT_history.size(); n++) {
      ofile << std::setprecision(10) << dT_history[n].first / FLOW_YEAR << " " << dT_history[n].second << std::endl;
    }
    ofile.close();
  }
}


/* *******************************************************************
* Calculates best least square fit for data (h[i], error[i]).                       
******************************************************************* */
double bestLSfit(const std::vector<double>& h, const std::vector<double>& error)
{
  double a = 0.0, b = 0.0, c = 0.0, d = 0.0, tmp1, tmp2;

  int n = h.size();
  for (int i = 0; i < n; i++) {
    tmp1 = log(h[i]);
    tmp2 = log(error[i]);
    a += tmp1;
    b += tmp2;
    c += tmp1 * tmp1;
    d += tmp1 * tmp2;
  }

  return (a * b - n * d) / (a * a - n * c);
}


}  // namespace Flow
}  // namespace Amanzi

