/*
This is the flow component of the Amanzi code. 

Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
Amanzi is released under the three-clause BSD License. 
The terms of use and "as is" disclaimer for this license are 
provided in the top-level COPYRIGHT file.

Authors: Neil Carlson (version 1) 
         Konstantin Lipnikov (version 2) (lipnikov@lanl.gov)
*/

#ifndef __RICHARDS_PK_HH__
#define __RICHARDS_PK_HH__

#include "Epetra_Vector.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"
#include "AztecOO.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "BDF2_TI.hh"
#include "BDF1_TI.hh"

#include "Flow_PK.hh"
#include "Flow_BC_Factory.hh"
#include "Flow_SourceFactory.hh"
#include "Matrix_MFD.hh"
#include "WaterRetentionModel.hh"
#include "RelativePermeability.hh"
#include "TI_Specs.hh"

#include "TimerManager.hh"


namespace Amanzi {
namespace AmanziFlow {

class Flow_State;  // forward declarations

class Richards_PK : public Flow_PK {
 public:
  Richards_PK(Teuchos::ParameterList& global_list, Teuchos::RCP<Flow_State> FS_MPC);
  ~Richards_PK();

  // main methods
  void InitPK();
  void InitSteadyState(double T0, double dT0);
  void InitTransient(double T0, double dT0);
  void InitPicard(double T0);
  void InitNextTI(double T0, double dT0, TI_Specs& ti_specs);

  double CalculateFlowDt();
  int Advance(double dT_MPC); 
  int AdvanceToSteadyState(double T0, double dT0);
  void InitializeAuxiliaryData();
  void InitializeSteadySaturated();

  int AdvanceToSteadyState_Picard(TI_Specs& ti_specs);
  int AdvanceToSteadyState_PicardNewton(TI_Specs& ti_specs);
  int AdvanceToSteadyState_BackwardEuler(TI_Specs& ti_specs);
  int AdvanceToSteadyState_BDF1(TI_Specs& ti_specs);
  int AdvanceToSteadyState_BDF2(TI_Specs& ti_specs);

  void CommitState(Teuchos::RCP<Flow_State> FS);

  // methods for experimental time integration
  int PicardTimeStep(double T, double dT, double& dTnext);
  int AndersonMixingTimeStep(double T, double dT, double& dTnext);
  double ErrorNormRC1(const Epetra_Vector& u, const Epetra_Vector& du);
  double ErrorNormSTOMP(const Epetra_Vector& u, const Epetra_Vector& du);
  double ErrorNormPicardExperimental(const Epetra_Vector& uold, const Epetra_Vector& unew);
 
  // methods required for time integration
  void fun(double T, const Epetra_Vector& u, const Epetra_Vector& udot, Epetra_Vector& rhs, double dT = 0.0);
  void precon(const Epetra_Vector& u, Epetra_Vector& Hu);
  double enorm(const Epetra_Vector& u, const Epetra_Vector& du);
  void update_norm(double rtol, double atol) {};
  void update_precon(double T, const Epetra_Vector& u, double dT, int& ierr);
  bool modify_update_step(double h, Epetra_Vector&u, Epetra_Vector& du );
  bool IsPureNewton() const;

  // other main methods
  void SetAbsolutePermeabilityTensor(std::vector<WhetStone::Tensor>& K);

  void AddTimeDerivative_MFD(Epetra_Vector& pressure_cells, double dTp, Matrix_MFD* matrix_operator);
  void AddTimeDerivative_MFDfake(Epetra_Vector& pressure_cells, double dTp, Matrix_MFD* matrix_operator);
  void AddTimeDerivative_MFDpicard(Epetra_Vector& pressure_cells, 
                                   Epetra_Vector& pressure_cells_dSdP, double dTp, Matrix_MFD* matrix_operator);

  double ComputeUDot(double T, const Epetra_Vector& u, Epetra_Vector& udot);
  void AssembleMatrixMFD(const Epetra_Vector &u, double Tp);
  void AssemblePreconditionerMFD(const Epetra_Vector &u, double Tp, double dTp);

  void UpdateSourceBoundaryData(double Tp, Epetra_Vector& p_cells, Epetra_Vector& p_faces);
  double AdaptiveTimeStepEstimate(double* dTfactor);

  // linear problems and solvers
  void AssembleSteadyStateMatrix_MFD(Matrix_MFD* matrix);
  void AssembleSteadyStatePreconditioner_MFD(Matrix_MFD* preconditioner);
  void SolveFullySaturatedProblem(double T, Epetra_Vector& u);
  void EnforceConstraints_MFD(double Tp, Epetra_Vector& u);

  // io members
  void ProcessParameterList();
  void ProcessStringLinearSolver(const std::string name, LinearSolver_Specs* ls_specs);
  void ProcessStringExperimentalSolver(const std::string name, int* method);
  void ProcessStringErrorOptions(Teuchos::ParameterList& list, int* control);

  std::string FindStringPreconditioner(const Teuchos::ParameterList& list);
  void AnalysisTI_Specs();

  // water retention models
  void DeriveSaturationFromPressure(const Epetra_Vector& p, Epetra_Vector& s);
  void DerivePressureFromSaturation(const Epetra_Vector& s, Epetra_Vector& p);

  // initization members
  void ClipHydrostaticPressure(const double pmin, Epetra_Vector& pressure_cells);
  void ClipHydrostaticPressure(const double pmin, const double s0, Epetra_Vector& pressure_cells);

  double CalculateRelaxationFactor(const Epetra_Vector& uold, const Epetra_Vector& unew);

  // control method
  void ResetErrorControl(int error) { error_control_ = error; }
  void ResetParameterList(const Teuchos::ParameterList& rp_list_new) { rp_list_ = rp_list_new; }
  void PrintStatistics() const;
  void PrintStatisticsCPU();
  
  // access methods
  const Epetra_Map& super_map() { return *super_map_; }
  AmanziGeometry::Point& gravity() { return gravity_; }

  // developement members
  bool SetSymmetryProperty();
  void ImproveAlgebraicConsistency(const Epetra_Vector& flux, 
                                   const Epetra_Vector& ws_prev, Epetra_Vector& ws);
  
  Matrix_MFD* preconditioner() { return preconditioner_; }
  int ApllyPrecInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y);

 private:
  Teuchos::ParameterList rp_list_;
  Teuchos::ParameterList solver_list_;
  Teuchos::ParameterList preconditioner_list_;

  double atm_pressure;

  Epetra_Map* super_map_;

  Teuchos::RCP<Epetra_Import> cell_importer_;  // parallel communicators
  Teuchos::RCP<Epetra_Import> face_importer_;

  AztecOO* solver;  // Linear solver data
  Matrix_MFD* matrix_;
  Matrix_MFD* preconditioner_;

  BDF2::Dae* bdf2_dae;  // Time integrators
  BDF1Dae* bdf1_dae;
  int block_picard;

  int error_control_;
  double functional_max_norm;
  int functional_max_cell;

  TI_Specs ti_specs_igs_;  // Three time integration phases
  TI_Specs ti_specs_sss_;
  TI_Specs ti_specs_trs_;
  TI_Specs* ti_specs;

  int max_itrs_linear;  // Generic parameters (igs, sss or trs)
  double convergence_tol_linear;

  Teuchos::RCP<Epetra_Vector> solution;  // global solution
  Teuchos::RCP<Epetra_Vector> solution_cells;  // cell-based pressures
  Teuchos::RCP<Epetra_Vector> solution_faces;  // face-base pressures
  Teuchos::RCP<Epetra_Vector> rhs;  // It has same size as solution.
  Teuchos::RCP<Epetra_Vector> rhs_faces;

  Teuchos::RCP<Epetra_Vector> pdot_cells_prev;  // time derivative of pressure
  Teuchos::RCP<Epetra_Vector> pdot_cells;

  BoundaryFunction* bc_pressure;  // Pressure BC.
  BoundaryFunction* bc_head;  // Static pressure head BC.
  BoundaryFunction* bc_flux;  // Outward mass flux BC.
  BoundaryFunction* bc_seepage;  // Seepage face BC.
  std::vector<int> bc_model, bc_submodel; 
  std::vector<bc_tuple> bc_values;
  Teuchos::RCP<Epetra_Vector> shift_water_table_;
  std::vector<double> rainfall_factor;

  DomainFunction* src_sink;  // Source and sink terms
  int src_sink_distribution; 

  std::vector<WhetStone::Tensor> K;  // tensor of absolute permeability
  Teuchos::RCP<Epetra_Vector> Kxy;  // absolute permeability in plane xy

  Teuchos::RCP<RelativePermeability> rel_perm;

  int mfd3d_method_, mfd3d_method_preconditioner_;
  bool is_matrix_symmetric;
  int experimental_solver_; 

  double mass_bc, mass_amanzi;

  // CPU statistics
  TimerManager timer;

 private:
  void operator=(const Richards_PK& RPK);
};

}  // namespace AmanziFlow
}  // namespace Amanzi

#endif

