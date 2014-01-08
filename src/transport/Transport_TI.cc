/*
This is the transport component of the Amanzi code. 

Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
Amanzi is released under the three-clause BSD License. 
The terms of use and "as is" disclaimer for this license are 
provided Reconstruction.cppin the top-level COPYRIGHT file.

Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <algorithm>

#include "Transport_PK.hh"

namespace Amanzi {
namespace AmanziTransport {

/* ******************************************************************* 
 * Routine takes a parallel overlapping vector C and returns a parallel
 * overlapping vector F(C).
 ****************************************************************** */
void Transport_PK::fun(const double t, const Epetra_Vector& component, Epetra_Vector& f_component)
{
  // transport routines need an RCP pointer
  Teuchos::RCP<const Epetra_Vector> component_rcp(&component, false);

  lifting.ResetField(mesh_, component_rcp);
  lifting.CalculateCellGradient();
  Teuchos::RCP<CompositeVector> gradient = lifting.gradient();

  if (advection_limiter == TRANSPORT_LIMITER_BARTH_JESPERSEN) {
    LimiterBarthJespersen(current_component_, component_rcp, gradient, limiter_);
    lifting.ApplyLimiter(limiter_);
  } else if (advection_limiter == TRANSPORT_LIMITER_TENSORIAL) {
    LimiterTensorial(current_component_, component_rcp, gradient);
  } else if (advection_limiter == TRANSPORT_LIMITER_KUZMIN) {
    LimiterKuzmin(current_component_, component_rcp, gradient);
  }

  // ADVECTIVE FLUXES
  // We assume that limiters made their job up to round-off errors.
  // Min-max condition will enforce robustness w.r.t. these errors.
  int f, c1, c2;
  double u, u1, u2, umin, umax, upwind_tcc, tcc_flux;

  f_component.PutScalar(0.0);
  for (int f = 0; f < nfaces_wghost; f++) {  // loop over master and slave faces
    c1 = (*upwind_cell_)[f];
    c2 = (*downwind_cell_)[f];

    if (c1 >= 0 && c2 >= 0) {
      u1 = component[c1];
      u2 = component[c2];
      umin = std::min(u1, u2);
      umax = std::max(u1, u2);
    } else if (c1 >= 0) {
      u1 = u2 = umin = umax = component[c1];
    } else if (c2 >= 0) {
      u1 = u2 = umin = umax = component[c2];
    }

    u = fabs((*darcy_flux)[0][f]);
    const AmanziGeometry::Point& xf = mesh_->face_centroid(f);

    if (c1 >= 0 && c1 < ncells_owned && c2 >= 0 && c2 < ncells_owned) {
      upwind_tcc = lifting.getValue(c1, xf);
      upwind_tcc = std::max(upwind_tcc, umin);
      upwind_tcc = std::min(upwind_tcc, umax);

      tcc_flux = u * upwind_tcc;
      f_component[c1] -= tcc_flux;
      f_component[c2] += tcc_flux;
    } else if (c1 >= 0 && c1 < ncells_owned && (c2 >= ncells_owned || c2 < 0)) {
      upwind_tcc = lifting.getValue(c1, xf);
      upwind_tcc = std::max(upwind_tcc, umin);
      upwind_tcc = std::min(upwind_tcc, umax);

      tcc_flux = u * upwind_tcc;
      f_component[c1] -= tcc_flux;
    } else if (c1 >= ncells_owned && c2 >= 0 && c2 < ncells_owned) {
      upwind_tcc = lifting.getValue(c1, xf);
      upwind_tcc = std::max(upwind_tcc, umin);
      upwind_tcc = std::min(upwind_tcc, umax);

      tcc_flux = u * upwind_tcc;
      f_component[c2] += tcc_flux;
    }
  }

  // process external sources
  if (src_sink != NULL) {
    ComputeAddSourceTerms(t, 1.0, src_sink, f_component);
  }

  for (int c = 0; c < ncells_owned; c++) {  // calculate conservative quantatity
    double vol_phi_ws = mesh_->cell_volume(c) * (*phi)[0][c] * (*ws_start)[0][c];
    f_component[c] /= vol_phi_ws;
  }

  // BOUNDARY CONDITIONS for ADVECTION
  for (int n = 0; n < bcs.size(); n++) {
    if (current_component_ == bcs_tcc_index[n]) {
      for (Amanzi::Functions::TransportBoundaryFunction::Iterator bc = bcs[n]->begin(); bc != bcs[n]->end(); ++bc) {
        f = bc->first;
        c2 = (*downwind_cell_)[f];

        if (c2 >= 0 && f < nfaces_owned) {
          u = fabs((*darcy_flux)[0][f]);
          double vol_phi_ws = mesh_->cell_volume(c2) * (*phi)[0][c2] * (*ws_start)[0][c2];
          tcc_flux = u * bc->second;
          f_component[c2] += tcc_flux / vol_phi_ws;
        }
      }
    }
  }
}

}  // namespace AmanziTransport
}  // namespace Amanzi


