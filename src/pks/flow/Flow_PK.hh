/*
  This is the flow component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Neil Carlson (version 1) 
           Konstantin Lipnikov (version 2) (lipnikov@lanl.gov)

  This is a virtual base class for two flow models: Darcy and Richards.
*/

#ifndef AMANZI_FLOW_PK_HH_
#define AMANZI_FLOW_PK_HH_

// TPLs
#include "Epetra_Vector.h"
#include "Epetra_IntVector.h"
#include "Epetra_FECrsMatrix.h"
#include "Teuchos_RCP.hpp"

// Amanzi
#include "BDFFnBase.hh"
#include "checkpoint.hh"
#include "CompositeVectorSpace.hh"
#include "FnTimeIntegratorPK.hh"
#include "independent_variable_field_evaluator_fromfunction.hh"
#include "PK.hh"
#include "primary_variable_field_evaluator.hh"
#include "tensor.hh"
#include "VerboseObject.hh"

// Flow
#include "Flow_BC_Factory.hh"
#include "Flow_SourceFactory.hh"
#include "FlowBoundaryFunction.hh"
#include "FlowDefs.hh"
#include "FlowDomainFunction.hh"
#include "FlowTypeDefs.hh"

namespace Amanzi {
namespace Flow {

class Flow_PK : public FnTimeIntegratorPK {
 public:
  Flow_PK();
  virtual ~Flow_PK() {};

  // members required by PK interface
  virtual void Setup();
  virtual void Initialize();

  // other members of this PK.
  // -- initialize simple fields common for both flow models.
  void UpdateLocalFields_();

  // --- management of boundary and source terms
  void ProcessBCs();
  void ComputeBCs(const CompositeVector& pressure);
  bool SeepageFacePFloTran(const CompositeVector& u, int* nseepage, double* area_seepage);
  bool SeepageFaceFACT(const CompositeVector& u, int* nseepage, double* area_seepage);

  void AddSourceTerms(CompositeVector& rhs);

  // -- absolute permeability and derived quantities.
  void SetAbsolutePermeabilityTensor();
  void CalculatePermeabilityFactorInWell();

  void ProcessShiftWaterTableList(const Teuchos::ParameterList& list);
  void CalculateShiftWaterTable(const std::string region);

  // -- miscallenous members
  void DeriveFaceValuesFromCellValues(const Epetra_MultiVector& ucells, Epetra_MultiVector& ufaces);
  int FindPosition(int f, AmanziMesh::Entity_ID_List faces);

  // -- io members
  void ProcessStringSourceDistribution(const std::string name, int* method);
  void OutputTimeHistory(const Teuchos::ParameterList& plist, std::vector<dt_tuple>& dt_history);
  void WriteGMVfile(Teuchos::RCP<State> S) const;

  // -- utilities
  double WaterVolumeChangePerSecond(const std::vector<int>& bc_model,
                                    const Epetra_MultiVector& darcy_flux) const;

  void CalculateDarcyVelocity(std::vector<AmanziGeometry::Point>& xyz, 
                              std::vector<AmanziGeometry::Point>& velocity);
  void CalculatePoreVelocity(std::vector<AmanziGeometry::Point>& xyz, 
                             std::vector<AmanziGeometry::Point>& velocity,
                             std::vector<double>& porosity, std::vector<double>& saturation,
                             std::vector<double>& pressure, std::vector<double>& water_density);
  void WriteWalkabout(const Teuchos::Ptr<Checkpoint>& wlk);

  // -- V&V
  void VV_ValidateBCs() const;
  void VV_ReportWaterBalance(const Teuchos::Ptr<State>& S) const;
  void VV_ReportSeepageOutflow(const Teuchos::Ptr<State>& S) const;
  void VV_PrintHeadExtrema(const CompositeVector& pressure) const;

  // -- extensions 
  int BoundaryFaceGetCell(int f) const;  // of AmanziMesh
  void VerticalNormals(int c, AmanziGeometry::Point& n1, AmanziGeometry::Point& n2);
  virtual double BoundaryFaceValue(int f, const CompositeVector& u);

  void set_intersection(const std::vector<AmanziMesh::Entity_ID>& v1,  // of std
                        const std::vector<AmanziMesh::Entity_ID>& v2, 
                        std::vector<AmanziMesh::Entity_ID>* vv);

  // -- support of unit tests
  double rho() { return rho_; }
  double mu() { return mu_; }
  const AmanziGeometry::Point& gravity() { return gravity_; }
  double seepage_mass() { return seepage_mass_; }

 private:
  void InitializeFields_();

 protected:
  void InitializeBCsSources_(Teuchos::ParameterList& list);

 public:
  int ncells_owned, ncells_wghost;
  int nfaces_owned, nfaces_wghost;

  double dt_, dt_next_;

  int MyPID;  // parallel information: will be moved to private
  int missed_bc_faces_, dirichlet_bc_faces_;
  int ti_phase_counter;

 public:
  Teuchos::RCP<const Teuchos::ParameterList> linear_operator_list_;
  Teuchos::RCP<const Teuchos::ParameterList> preconditioner_list_;
  Teuchos::RCP<Teuchos::ParameterList> ti_list_;

 protected:
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
  int dim;

  Teuchos::RCP<State> S_;
  std::string passwd_;

  // Stationary physical quantatities
  std::vector<WhetStone::Tensor> K; 
  AmanziGeometry::Point gravity_, molar_gravity_;
  double g_, rho_, molar_rho_, mu_, atm_pressure_;
  double flux_units_;  // scaling for flux units from kg to moles.

  Teuchos::RCP<Epetra_Vector> Kxy;
  std::string coordinate_system;

  // boundary conditons
  FlowBoundaryFunction* bc_pressure; 
  FlowBoundaryFunction* bc_head;
  FlowBoundaryFunction* bc_flux;
  FlowBoundaryFunction* bc_seepage;
  int nseepage_prev;

  std::vector<int> bc_model, bc_submodel; 
  std::vector<double> bc_value, bc_mixed;

  std::vector<double> rainfall_factor;
  Teuchos::RCP<Epetra_Vector> shift_water_table_;

  // water balance
  FlowDomainFunction* src_sink;
  int src_sink_distribution; 
  mutable double mass_bc, seepage_mass_;

  // field evaluators (MUST GO AWAY lipnikov@lanl.gov)
  Teuchos::RCP<PrimaryVariableFieldEvaluator> darcy_flux_eval_;
  Teuchos::RCP<PrimaryVariableFieldEvaluator> pressure_eval_;
  Teuchos::RCP<IndependentVariableFieldEvaluatorFromFunction> porosity_eval_;

 protected:
  VerboseObject* vo_;
};

}  // namespace Flow
}  // namespace Amanzi

#endif
