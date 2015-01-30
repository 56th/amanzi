/*
  This is the operator component of the Amanzi code. 

  Copyright 2010-2013 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_RECONSTRUCTION_CELL_HH_
#define AMANZI_RECONSTRUCTION_CELL_HH_

#include <vector>

#include "Epetra_IntVector.h"
#include "Epetra_MultiVector.h"
#include "Teuchos_RCP.hpp"

#include "CompositeVector.hh"
#include "DenseMatrix.hh"
#include "DenseVector.hh"
#include "Mesh.hh"
#include "Point.hh"

#include "Reconstruction.hh"


namespace Amanzi {
namespace Operators {

class ReconstructionCell : public Reconstruction {  
 public:
  ReconstructionCell() {};
  ReconstructionCell(Teuchos::RCP<const Amanzi::AmanziMesh::Mesh> mesh) : Reconstruction(mesh) {};
  ~ReconstructionCell() {};

  // main members for base class
  void Init(Teuchos::RCP<const Epetra_MultiVector> field, Teuchos::ParameterList& plist);
  void Compute();

  // internal and external limiters
  void InitLimiter(Teuchos::RCP<const Epetra_MultiVector> flux);
  void ApplyLimiter(const std::vector<int>& bc_model, const std::vector<double>& bc_value);
  void ApplyLimiter(Teuchos::RCP<Epetra_MultiVector> limiter);

  // estimate value of a reconstructed piece-wise smooth function
  double getValue(int cell, const AmanziGeometry::Point& p);

  // estimate value of a reconstructed linear function with prescribed gradient. 
  double getValue(AmanziGeometry::Point& gradient, int cell, const AmanziGeometry::Point& p);

  // access
  Teuchos::RCP<CompositeVector> gradient() { return gradient_; }
 
 private:
  void PopulateLeastSquareSystem(AmanziGeometry::Point& centroid,
                                 double field_value,
                                 WhetStone::DenseMatrix& matrix,
                                 WhetStone::DenseVector& rhs);

  void PrintLeastSquareSystem(WhetStone::DenseMatrix& matrix,
                              WhetStone::DenseVector& rhs);

  // internal limiters and supporting routines
  void LimiterBarthJespersen_(
      const std::vector<int>& bc_model, const std::vector<double>& bc_value,
      Teuchos::RCP<Epetra_Vector> limiter);

  void LimiterTensorial_(
      const std::vector<int>& bc_model, const std::vector<double>& bc_value);

  void LimiterKuzmin_(
      const std::vector<int>& bc_model, const std::vector<double>& bc_value);

  void CalculateDescentDirection_(std::vector<AmanziGeometry::Point>& normals,
                                  AmanziGeometry::Point& normal_new,
                                  double& L22normal_new, 
                                  AmanziGeometry::Point& direction);

  void ApplyDirectionalLimiter_(AmanziGeometry::Point& normal, 
                                AmanziGeometry::Point& p,
                                AmanziGeometry::Point& direction, 
                                AmanziGeometry::Point& gradient);

  void IdentifyUpwindCells_();

  void LimiterExtensionTransportTensorial_(
      const std::vector<double>& field_local_min, const std::vector<double>& field_local_max);

  void LimiterExtensionTransportKuzmin_(
      const std::vector<double>& field_local_min, const std::vector<double>& field_local_max);

  void LimiterExtensionTransportBarthJespersen_(
      const std::vector<double>& field_local_min, const std::vector<double>& field_local_max,
      Teuchos::RCP<Epetra_Vector> limiter);

 private:
  int dim;
  int ncells_owned, nfaces_owned, nnodes_owned;
  int ncells_wghost, nfaces_wghost, nnodes_wghost;

  Teuchos::RCP<CompositeVector> gradient_;

 private: 
  Teuchos::RCP<const Epetra_MultiVector> flux_;  // for limiters
  std::vector<int> upwind_cell_, downwind_cell_;

  double bc_scaling_;
  int limiter_id_, poly_order_;
  bool limiter_correction_;
};

}  // namespace Operators
}  // namespace Amanzi

#endif
