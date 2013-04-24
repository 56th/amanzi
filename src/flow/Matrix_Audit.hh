/*
This is the Audit component of the Amanzi code. 
License: BSD
Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef __MATRIX_AUDIT_HH__
#define __MATRIX_AUDIT_HH__

#include "Epetra_Operator.h"

#include "Matrix_MFD.hh"


namespace Amanzi {

int const MATRIX_AUDIT_MFD = 1;

class Matrix_Audit {
 public:
  Matrix_Audit(Teuchos::RCP<const AmanziMesh::Mesh> mesh, AmanziFlow::Matrix_MFD* matrix);
  ~Matrix_Audit();

  // main members
  void InitAudit();
  int RunAudit();
  int CheckSpectralBounds();
  int CheckSpectralBoundsExtended();
  int CheckSpectralBoundsSchurComplement();
  int CheckMatrixSymmetry();

 private:
  void OrderByIncrease(int n, double* mem);

  int MyPID;
  int matrix_type;
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
  AmanziFlow::Matrix_MFD* matrix_;

  std::vector<Teuchos::SerialDenseMatrix<int, double> >* A;  // local matrices
  int lda;  // maximum size of elemental matrices

  int lwork1;  // work memory
  double *dmem1, *dmem2;
  double *dwork1;
};

}  // namespace Amanzi

#endif
