/*
  This is the flow component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Neil Carlson (version 1) 
           Konstantin Lipnikov (version 2) (lipnikov@lanl.gov)
*/

#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"

#include "GMVMesh.hh"
#include "Mesh.hh"
#include "mfd3d.hh"
#include "OperatorDefs.hh"
#include "State.hh"

#include "Flow_PK.hh"

namespace Amanzi {
namespace Flow {

/* ******************************************************************
* default constructor that initializes all pointers to NULL
****************************************************************** */
Flow_PK::Flow_PK() :
    bc_pressure(NULL),
    bc_flux(NULL),
    bc_head(NULL),
    bc_seepage(NULL),
    src_sink(NULL),
    vo_(NULL),
    passwd_("flow")
{
}


/* ******************************************************************
* Setup of static fields common for Darcy and Richards.
****************************************************************** */
void Flow_PK::Setup()
{
  if (!S_->HasField("fluid_density")) {
    S_->RequireScalar("fluid_density", passwd_);
  }
  if (!S_->HasField("fluid_viscosity")) {
    S_->RequireScalar("fluid_viscosity", passwd_);
  }
  if (!S_->HasField("gravity")) {
    S_->RequireConstantVector("gravity", passwd_, dim);  // state resets ownership.
  } 

  if (!S_->HasField("permeability")) {
    S_->RequireField("permeability", passwd_)->SetMesh(mesh_)->SetGhosted(true)
      ->SetComponent("cell", AmanziMesh::CELL, dim);
  }
}


/* ******************************************************************
* Initiazition of fundamental flow sturctures.
****************************************************************** */
void Flow_PK::Initialize()
{
  dt_ = 0.0;

  ncells_owned = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED);
  ncells_wghost = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::USED);

  nfaces_owned = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::OWNED);
  nfaces_wghost = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::USED);

  nseepage_prev = 0;
  ti_phase_counter = 0;

  // Fundamental physical quantities
  // -- temporarily these quantities are constant
  double* gravity_data;
  S_->GetConstantVectorData("gravity")->ExtractView(&gravity_data);
  gravity_.set(dim, &(gravity_data[0]));  // do it in complicated way because we
                                          // are not sure if gravity_data is an
                                          // array or vector
  g_ = fabs(gravity_[dim - 1]);

  rho_ = *S_->GetScalarData("fluid_density");
  mu_ = *S_->GetScalarData("fluid_viscosity");

  // -- molar rescaling of some quantatities.
  molar_rho_ = rho_ / CommonDefs::MOLAR_MASS_H2O;
  molar_gravity_.set(gravity_ * CommonDefs::MOLAR_MASS_H2O);
  // molar_rho_ = rho_;
  // molar_gravity_.set(gravity_);
  flux_units_ = 0.0;  // scaling from kg to moles

  // parallel execution data
  MyPID = 0;
#ifdef HAVE_MPI
  MyPID = mesh_->cell_map(false).Comm().MyPID();
#endif

  InitializeFields_();
}


/* ****************************************************************
* This completes initialization of common fields that were not 
* initialized by the state.
**************************************************************** */
void Flow_PK::InitializeFields_()
{
  Teuchos::OSTab tab = vo_->getOSTab();

  // set popular default values for missed fields.
  if (S_->GetField("porosity")->owner() == passwd_) {
    if (!S_->GetField("porosity", passwd_)->initialized()) {
      S_->GetFieldData("porosity", passwd_)->PutScalar(0.2);
      S_->GetField("porosity", passwd_)->set_initialized();

      if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
          *vo_->os() << "initilized porosity to default value 0.2" << std::endl;  
    }
  }

  if (S_->GetField("fluid_density")->owner() == passwd_) {
    if (!S_->GetField("fluid_density", passwd_)->initialized()) {
      *(S_->GetScalarData("fluid_density", passwd_)) = 1000.0;
      S_->GetField("fluid_density", passwd_)->set_initialized();

      if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
          *vo_->os() << "initilized fluid_density to default value 1000.0" << std::endl;  
    }
  }

  if (!S_->GetField("fluid_viscosity", passwd_)->initialized()) {
    *(S_->GetScalarData("fluid_viscosity", passwd_)) = 0.001;
    S_->GetField("fluid_viscosity", passwd_)->set_initialized();

    if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
        *vo_->os() << "initilized fluid_viscosity to default value 0.001" << std::endl;  
  }

  if (!S_->GetField("gravity", "state")->initialized()) {
    Epetra_Vector& gvec = *S_->GetConstantVectorData("gravity", "state");
    gvec.PutScalar(0.0);
    gvec[dim - 1] = -9.80;
    S_->GetField("gravity", "state")->set_initialized();

    if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
        *vo_->os() << "initilized gravity to default value -9.8" << std::endl;  
  }

  if (!S_->GetField("permeability", passwd_)->initialized()) {
    S_->GetFieldData("permeability", passwd_)->PutScalar(1.0);
    S_->GetField("permeability", passwd_)->set_initialized();

    if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
        *vo_->os() << "initilized permeability to default value 1.0" << std::endl;  
  }

  if (S_->HasField("specific_storage")) {
    if (!S_->GetField("specific_storage", passwd_)->initialized()) {
      S_->GetFieldData("specific_storage", passwd_)->PutScalar(0.0);
      S_->GetField("specific_storage", passwd_)->set_initialized();
    }
  }

  if (S_->HasField("specific_yield")) {
    if (!S_->GetField("specific_yield", passwd_)->initialized()) {
      S_->GetFieldData("specific_yield", passwd_)->PutScalar(0.0);
      S_->GetField("specific_yield", passwd_)->set_initialized();

      if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
          *vo_->os() << "initilized specific_yield to default value 1.0" << std::endl;  
    }
  }

  if (!S_->GetField("pressure", passwd_)->initialized()) {
    S_->GetFieldData("pressure", passwd_)->PutScalar(0.0);
    S_->GetField("pressure", passwd_)->set_initialized();

    if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
        *vo_->os() << "initilized pressure to default value 0.0" << std::endl;  
  }

  if (!S_->GetField("hydraulic_head", passwd_)->initialized()) {
    S_->GetFieldData("hydraulic_head", passwd_)->PutScalar(0.0);
    S_->GetField("hydraulic_head", passwd_)->set_initialized();

    if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
        *vo_->os() << "initilized hydraulic_head to default value 0.0" << std::endl;  
  }

  if (!S_->GetField("darcy_flux", passwd_)->initialized()) {
    S_->GetFieldData("darcy_flux", passwd_)->PutScalar(0.0);
    S_->GetField("darcy_flux", passwd_)->set_initialized();

    if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM)
        *vo_->os() << "initilized darcy_flux to default value 0.0" << std::endl;  
  }
}


/* ****************************************************************
* Hydraulic head support for Flow PKs.
**************************************************************** */
void Flow_PK::UpdateLocalFields_() 
{
  Teuchos::OSTab tab = vo_->getOSTab();
  if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM) {
    *vo_->os() << "Secondary fields: hydraulic head, darcy_velocity" << std::endl;  
  }  

  Epetra_MultiVector& hydraulic_head = *(S_->GetFieldData("hydraulic_head", passwd_)->ViewComponent("cell"));
  const Epetra_MultiVector& pressure = *(S_->GetFieldData("pressure")->ViewComponent("cell"));
  double rho = *(S_->GetScalarData("fluid_density"));

  // calculate hydraulic head
  double g = fabs(gravity_[dim - 1]);

  for (int c = 0; c != ncells_owned; ++c) {
    const AmanziGeometry::Point& xc = mesh_->cell_centroid(c); 
    double z = xc[dim - 1]; 
    hydraulic_head[0][c] = z + (pressure[0][c] - atm_pressure_) / (g * rho);
  }

  // calculate full velocity vector
  darcy_flux_eval_->SetFieldAsChanged(S_.ptr());
  S_->GetFieldEvaluator("darcy_velocity")->HasFieldChanged(S_.ptr(), "darcy_velocity");
}


/* ******************************************************************
* Routine processes parameter list. It needs to be called only once
* on each processor.                                                     
****************************************************************** */
void Flow_PK::InitializeBCsSources_(Teuchos::ParameterList& plist)
{
  // Process main one-line options (not sublists)
  atm_pressure_ = plist.get<double>("atmospheric pressure", FLOW_PRESSURE_ATMOSPHERIC);
  coordinate_system = plist.get<std::string>("absolute permeability coordinate system", "cartesian");

  // Create the BC objects.
  bc_model.resize(nfaces_wghost, 0);
  bc_submodel.resize(nfaces_wghost, 0);
  bc_value.resize(nfaces_wghost, 0.0);
  bc_mixed.resize(nfaces_wghost, 0.0);
  rainfall_factor.resize(nfaces_wghost, 1.0);

  Teuchos::RCP<Teuchos::ParameterList>
      bc_list = Teuchos::rcp(new Teuchos::ParameterList(plist.sublist("boundary conditions", true)));
  FlowBCFactory bc_factory(mesh_, bc_list);

  bc_pressure = bc_factory.CreatePressure(bc_submodel);
  bc_head = bc_factory.CreateStaticHead(atm_pressure_, rho_, gravity_, bc_submodel);
  bc_flux = bc_factory.CreateMassFlux(bc_submodel);
  bc_seepage = bc_factory.CreateSeepageFace(atm_pressure_, bc_submodel);

  VV_ValidateBCs();
  ProcessBCs();

  // Create the source object if any
  if (plist.isSublist("source terms")) {
    std::string distribution_method_name = plist.get<std::string>("source and sink distribution method", "none");
    ProcessStringSourceDistribution(distribution_method_name, &src_sink_distribution); 

    Teuchos::RCP<Teuchos::ParameterList> src_list = Teuchos::rcpFromRef(plist.sublist("source terms", true));
    FlowSourceFactory src_factory(mesh_, src_list);
    src_sink = src_factory.createSource();
    src_sink_distribution = src_sink->CollectActionsList();
  }
}


/* ******************************************************************
* Populate data needed by submodels.
****************************************************************** */
void Flow_PK::ProcessBCs()
{
  for (int f = 0; f < nfaces_owned; f++) {
    if (bc_submodel[f] & FLOW_BC_SUBMODEL_RAINFALL) {
      const AmanziGeometry::Point& normal = mesh_->face_normal(f);
      rainfall_factor[f] = fabs(normal[dim - 1]) / norm(normal);
    }
  }
}


/* ******************************************************************
* Add a boundary marker to used faces.
* WARNING: we can skip update of ghost boundary faces, b/c they 
* should be always owned. 
****************************************************************** */
void Flow_PK::ComputeBCs(const CompositeVector& u)
{
  const Epetra_MultiVector& u_cell = *u.ViewComponent("cell");
  
  for (int n = 0; n < bc_model.size(); n++) {
    bc_model[n] = Operators::OPERATOR_BC_NONE;
    bc_value[n] = 0.0;
    bc_mixed[n] = 0.0;
  }

  FlowBoundaryFunction::Iterator bc;
  for (bc = bc_pressure->begin(); bc != bc_pressure->end(); ++bc) {
    int f = bc->first;
    bc_model[f] = Operators::OPERATOR_BC_DIRICHLET;
    bc_value[f] = bc->second;
  }

  for (bc = bc_head->begin(); bc != bc_head->end(); ++bc) {
    int f = bc->first;
    if (bc_submodel[f] & FLOW_BC_SUBMODEL_NOFLOW_ABOVE_WATER_TABLE) {
      if (bc->second < atm_pressure_) {
        bc_model[f] = Operators::OPERATOR_BC_NEUMANN;
        bc_value[f] = 0.0;
        continue;
      }
    }
    bc_model[f] = Operators::OPERATOR_BC_DIRICHLET;
    bc_value[f] = bc->second;
  }

  for (bc = bc_flux->begin(); bc != bc_flux->end(); ++bc) {
    int f = bc->first;
    bc_model[f] = Operators::OPERATOR_BC_NEUMANN;
    bc_value[f] = bc->second * rainfall_factor[f] * flux_units_;
  }

  // Seepage face BC is implemented for p-lambda discretization only.
  int nseepage_add, nseepage = 0;
  double area_add, area_seepage = 0.0;

  bool done = SeepageFacePFloTran(u, &nseepage_add, &area_add);
  nseepage += nseepage_add;
  area_seepage += area_add;

  if (!done) {
    done = SeepageFaceFACT(u, &nseepage_add, &area_add);
    nseepage += nseepage_add;
    area_seepage += area_add;
  }

  // mark missing boundary conditions as zero flux conditions
  AmanziMesh::Entity_ID_List cells;
  missed_bc_faces_ = 0;
  for (int f = 0; f < nfaces_owned; f++) {
    if (bc_model[f] == Operators::OPERATOR_BC_NONE) {
      cells.clear();
      mesh_->face_get_cells(f, AmanziMesh::USED, &cells);
      int ncells = cells.size();

      if (ncells == 1) {
        bc_model[f] = Operators::OPERATOR_BC_NEUMANN;
        bc_value[f] = 0.0;
        missed_bc_faces_++;
      }
    }
  }

  dirichlet_bc_faces_ = 0;
  for (int f = 0; f < nfaces_owned; ++f) {
    if (bc_model[f] == Operators::OPERATOR_BC_DIRICHLET) dirichlet_bc_faces_++;
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

  if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH) {
#ifdef HAVE_MPI
    int nseepage_tmp = nseepage;
    double area_tmp = area_seepage;
    mesh_->get_comm()->SumAll(&area_tmp, &area_seepage, 1);
    mesh_->get_comm()->SumAll(&nseepage_tmp, &nseepage, 1);
#endif
    if (MyPID == 0 && nseepage > 0 && nseepage != nseepage_prev) {
      Teuchos::OSTab tab = vo_->getOSTab();
      *vo_->os() << "seepage face: " << area_seepage << " [m^2], from "
                 << nseepage_prev << " to " << nseepage << " faces" << std::endl;
    }
  }
  nseepage_prev = nseepage;
}


/* ******************************************************************
*  Temporary convertion from double to tensor.                                               
****************************************************************** */
void Flow_PK::SetAbsolutePermeabilityTensor()
{
  const CompositeVector& cv = *S_->GetFieldData("permeability");
  cv.ScatterMasterToGhosted("cell");
  const Epetra_MultiVector& perm = *cv.ViewComponent("cell", true);
 
  // For permeabilities given in local (layer-based) coordinates
  AmanziGeometry::Point n1(dim), n2(dim), normal(dim), tau(dim);
  WhetStone::Tensor N(dim, 2), Ninv(dim, 2), D(dim, 2);

  if (dim == 2) {
    for (int c = 0; c < K.size(); c++) {
      if (perm[0][c] == perm[1][c]) {
	K[c].Init(dim, 1);
	K[c](0, 0) = perm[0][c];
      } else if (coordinate_system == "cartesian") {
	K[c].Init(dim, 2);
	K[c](0, 0) = perm[0][c];
	K[c](1, 1) = perm[1][c];
      } else {
        VerticalNormals(c, n1, n2);
        normal = (n1 - n2) / 2;
        normal /= norm(normal);

        tau[0] = normal[1];
        tau[1] = -normal[0];
        
        N.SetColumn(0, tau); 
        N.SetColumn(1, normal); 

        Ninv = N;
        Ninv.Inverse();

        D(0, 0) = perm[0][c];
        D(1, 1) = perm[1][c];
        K[c] = N * D * Ninv;
      }
    }    
  } else if (dim == 3) {
    for (int c = 0; c < K.size(); c++) {
      if (perm[0][c] == perm[1][c] && perm[0][c] == perm[2][c]) {
	K[c].Init(dim, 1);
	K[c](0, 0) = perm[0][c];
      } else {
	K[c].Init(dim, 2);
	K[c](0, 0) = perm[0][c];
	K[c](1, 1) = perm[1][c];
	K[c](2, 2) = perm[2][c];
      }
    }        
  }
}


/* ******************************************************************
*  Calculate inner product e^T K e in each cell.                                               
****************************************************************** */
void Flow_PK::CalculatePermeabilityFactorInWell()
{
  for (int c = 0; c < ncells_owned; c++) {
    (*Kxy)[c] = 0.0;
    int idim = std::max(1, K[c].size() - 1);
    for (int i = 0; i < idim; i++) (*Kxy)[c] += K[c](i, i);
    (*Kxy)[c] /= idim;
  }

  // parallelization using CV capability
#ifdef HAVE_MPI
  CompositeVectorSpace cvs;
  cvs.SetMesh(mesh_);
  cvs.SetGhosted(true);
  cvs.SetComponent("cell", AmanziMesh::CELL, 1);

  CompositeVector tmp(cvs, true);
  Epetra_MultiVector& data = *tmp.ViewComponent("cell", true);

  data = *Kxy;
  tmp.ScatterMasterToGhosted("cell", true);
  for (int c = ncells_owned; c < ncells_wghost; c++) {
    (*Kxy)[c] = data[0][c];
  }
#endif
}


/* ******************************************************************
* Add source and sink terms.                                   
****************************************************************** */
void Flow_PK::AddSourceTerms(CompositeVector& rhs)
{
  Epetra_MultiVector& rhs_cell = *rhs.ViewComponent("cell");
  FlowDomainFunction::Iterator src;

  for (src = src_sink->begin(); src != src_sink->end(); ++src) {
    int c = src->first;
    rhs_cell[0][c] += mesh_->cell_volume(c) * src->second;
  }
}


/* ******************************************************************
* BDF methods need a good initial guess.
* This method gives a less smoother solution than in Flow 1.0.
* WARNING: Each owned face must have at least one owned cell. 
* Probability that this assumption is violated is close to zero. 
* Even when it happens, the code will not crash.
****************************************************************** */
void Flow_PK::DeriveFaceValuesFromCellValues(
    const Epetra_MultiVector& ucells, Epetra_MultiVector& ufaces)
{
  AmanziMesh::Entity_ID_List cells;
  int nfaces = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::OWNED);

  for (int f = 0; f < nfaces; f++) {
    cells.clear();
    mesh_->face_get_cells(f, AmanziMesh::OWNED, &cells);
    int ncells = cells.size();

    double face_value = 0.0;
    for (int n = 0; n < ncells; n++) face_value += ucells[0][cells[n]];
    ufaces[0][f] = face_value / ncells;
  }
}


/* ******************************************************************
* Calculate change of water volume per second due to boundary flux.                                          
****************************************************************** */
double Flow_PK::WaterVolumeChangePerSecond(const std::vector<int>& bc_model,
                                           const Epetra_MultiVector& darcy_flux) const
{
  AmanziMesh::Entity_ID_List faces;
  std::vector<int> fdirs;

  double volume = 0.0;
  for (int c = 0; c < ncells_owned; c++) {
    mesh_->cell_get_faces_and_dirs(c, &faces, &fdirs);

    for (int i = 0; i < faces.size(); i++) {
      int f = faces[i];
      if (bc_model[f] != Operators::OPERATOR_BC_NONE && f < nfaces_owned) {
        if (fdirs[i] >= 0) {
          volume -= darcy_flux[0][f];
        } else {
          volume += darcy_flux[0][f];
        }
      }
    }
  }
  return volume;
}


/* ******************************************************************
* Returns the first cell attached to a boundary face.   
****************************************************************** */
int Flow_PK::BoundaryFaceGetCell(int f) const
{
  AmanziMesh::Entity_ID_List cells;
  mesh_->face_get_cells(f, AmanziMesh::USED, &cells);
  return cells[0];
}


/* ******************************************************************
* Returns approximation of a solution on a boundary face   
****************************************************************** */
double Flow_PK::BoundaryFaceValue(int f, const CompositeVector& u)
{
  double face_value;
  if (u.HasComponent("face")) {
    const Epetra_MultiVector& u_face = *u.ViewComponent("face");
    face_value = u_face[0][f];
  } else {
    const Epetra_MultiVector& u_cell = *u.ViewComponent("cell");
    int c = BoundaryFaceGetCell(f);
    face_value = u_cell[0][c];
  }
  return face_value;
}


/* ******************************************************************
* Find cell normals that have direction close to gravity (n1) and
* anti-gravity (n2).
****************************************************************** */
void Flow_PK::VerticalNormals(int c, AmanziGeometry::Point& n1, AmanziGeometry::Point& n2)
{
  AmanziMesh::Entity_ID_List faces;
  std::vector<int> dirs;

  mesh_->cell_get_faces_and_dirs(c, &faces, &dirs);
  int nfaces = faces.size();

  int i1, i2;
  double amax(-1e+50), amin(1e+50), a;
  for (int i = 0; i < nfaces; i++) {
    int f = faces[i];
    double area = mesh_->face_area(f);
    const AmanziGeometry::Point normal = mesh_->face_normal(f);

    a = normal[dim - 1] * dirs[i] / area;
    if (a > amax) { 
      i1 = i;
      amax = a;
    } 
    if (a < amin) { 
      i2 = i;
      amin = a;
    } 
  }

  n1 = mesh_->face_normal(faces[i1]) * dirs[i1];
  n2 = mesh_->face_normal(faces[i2]) * dirs[i2];
}


/* ******************************************************************
* Returns position of face f in the list faces.  
****************************************************************** */
int Flow_PK::FindPosition(int f, AmanziMesh::Entity_ID_List faces)
{
  for (int i = 0; i < faces.size(); i++) {
    if (faces[i] == f) return i;
  }
  return -1;
}


/* ****************************************************************
* DEBUG: creating GMV file 
**************************************************************** */
void Flow_PK::WriteGMVfile(Teuchos::RCP<State> FS) const
{
  GMV::open_data_file(*mesh_, (std::string)"flow.gmv");
  GMV::start_data();
  GMV::write_cell_data(*(S_->GetFieldData("pressure")->ViewComponent("cell")), 0, "pressure");
  GMV::write_cell_data(*(S_->GetFieldData("saturation_liquid")->ViewComponent("cell")), 0, "saturation");
  GMV::close_data_file();
}


/* ****************************************************************
* Process string for the linear solver.
**************************************************************** */
void Flow_PK::ProcessStringSourceDistribution(const std::string name, int* method)
{
  if (name != "none") {
    Errors::Message msg;
    msg << "\nFlow_PK: \"source and sink distribution method\" is obsolete.\n"
        << "         see desription of sublist \"source terms\" in the native spec.\n";
    Exceptions::amanzi_throw(msg);
  }
}

}  // namespace Flow
}  // namespace Amanzi

