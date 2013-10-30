/*
This is the audit component of the Amanzi code. 
License: BSD
Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include "Epetra_SerialDenseVector.h"

#include "Matrix_MFD.hh"
#include "Matrix_Audit.hh"


namespace Amanzi {

/* ******************************************************************
* Constructor.                                      
****************************************************************** */
Matrix_Audit::Matrix_Audit(Teuchos::RCP<const AmanziMesh::Mesh> mesh, AmanziFlow::Matrix_MFD* matrix)
   : mesh_(mesh)
{ 
  matrix_type = MATRIX_AUDIT_MFD; 
  matrix_ = matrix;
}


/* ******************************************************************
* Destructor.                                      
****************************************************************** */
Matrix_Audit::~Matrix_Audit()
{
  delete [] dmem1;
  delete [] dmem2;
  delete [] dwork1;
}


/* ******************************************************************
* Calculates global information about matrices                                       
****************************************************************** */
void Matrix_Audit::InitAudit()
{
  MyPID = mesh_->cell_map(false).Comm().MyPID();

  if (matrix_type == MATRIX_AUDIT_MFD) {
    A = &(matrix_->Aff_cells());
  }

  lda = 1;
  for (int i = 0; i < A->size(); i++) {
    Teuchos::SerialDenseMatrix<int, double>& Ai = (*A)[i];
    lda = std::max(lda, Ai.numRows());
  }

  // allocate memory for Lapack
  dmem1 = new double[lda + 1];
  dmem2 = new double[lda + 1];

  lwork1 = 10 * (lda + 1);
  dwork1 = new double[lwork1];
  if (MyPID == 0) {
    printf("Matrix_Audit: initializing for matrix id =%2d\n", matrix_type);
    printf("Matrix_Audit: maximum matrix size =%3d\n", lda); 
  }
}


/* ******************************************************************
* AAA.                                      
****************************************************************** */
int Matrix_Audit::RunAudit()
{
  int ierr;
  ierr = CheckSpectralBounds();
  ierr |= CheckSpectralBoundsExtended();
  ierr |= CheckSpectralBoundsSchurComplement();
  if (matrix_type == MATRIX_AUDIT_MFD) { 
    CheckMatrixSymmetry();
    CheckMatrixCoercivity();
  }
  return ierr;
}


/* ******************************************************************
* AAA.                                      
****************************************************************** */
int Matrix_Audit::CheckSpectralBounds()
{ 
  Teuchos::LAPACK<int, double> lapack;
  int info;
  double VL, VR;

  double emin = 1e+99, emax = -1e+99;
  double cndmin = 1e+99, cndmax = 1.0, cndavg = 0.0;

  for (int c = 0; c < A->size(); c++) {
    Teuchos::SerialDenseMatrix<int, double> Acell((*A)[c]);
    int n = Acell.numRows();
    
    lapack.GEEV('N', 'N', n, Acell.values(), n, dmem1, dmem2, 
                &VL, 1, &VR, 1, dwork1, lwork1, &info);

    OrderByIncrease(n, dmem1);

    double e, a = dmem1[1], b = dmem1[1];  // skipping the first eigenvalue
    for (int k=2; k<n; k++) {
      e = dmem1[k];
      a = std::min(a, e);
      b = std::max(b, e);
    }

    emin = std::min(emin, a);
    emax = std::max(emax, b);

    double cnd = b / a;
    cndmin = std::min(cndmin, cnd);
    cndmax = std::max(cndmax, cnd);
    cndavg += cnd;
  }
  cndavg /= A->size();

  if (MyPID == 0) {
    printf("Matrix_Audit: lambda matrices\n");
    printf("   eigenvalues (min,max) = %8.3g %8.3g\n", emin, emax); 
    printf("   conditioning (min,max,avg) = %8.2g %8.2g %8.2g\n", cndmin, cndmax, cndavg);
  }
  return 0;
}


/* ******************************************************************
* AAA.                                      
****************************************************************** */
int Matrix_Audit::CheckSpectralBoundsExtended()
{ 
  Errors::Message msg;

  Teuchos::LAPACK<int, double> lapack;
  int info;
  double VL, VR;

  double emin = 1e+99, emax = -1e+99;
  double cndmin = 1e+99, cndmax = 1.0, cndavg = 0.0;

  for (int c = 0; c < A->size(); c++) {
    Teuchos::SerialDenseMatrix<int, double>& Aff = matrix_->Aff_cells()[c];
    Epetra_SerialDenseVector& Acf = matrix_->Acf_cells()[c];
    Epetra_SerialDenseVector& Afc = matrix_->Afc_cells()[c];
    double& Acc = matrix_->Acc_cells()[c];
    int n = Aff.numRows();

    Teuchos::SerialDenseMatrix<int, double> Acell(n+1, n+1);
    for (int i = 0; i < n; i++) {
      Acell(n, n) = Acc;
      Acell(i, n) = Afc[i];
      Acell(n, i) = Acf[i];
      for (int j = 0; j < n; j++) Acell(i, j) = Aff(i, j);
    }
    Teuchos::SerialDenseMatrix<int, double> Acopy(Acell);

    if (Acc <= 0.0) {
      cout << Acell << endl;
      msg << "Matrix Audit: Acc is not positive.";
      Exceptions::amanzi_throw(msg);
    }
    
    n++;
    lapack.GEEV('N', 'N', n, Acell.values(), n, dmem1, dmem2, 
                &VL, 1, &VR, 1, dwork1, lwork1, &info);

    OrderByIncrease(n, dmem1);

    double e, a = dmem1[1], b = dmem1[1];  // skipping the first eigenvalue
    for (int k=2; k<n; k++) {
      e = dmem1[k];
      a = std::min(a, e);
      b = std::max(b, e);
    }

    emin = std::min(emin, a);
    emax = std::max(emax, b);

    double cnd = b / a;
    cndmin = std::min(cndmin, cnd);
    cndmax = std::max(cndmax, cnd);
    cndavg += cnd;
  }
  cndavg /= A->size();

  if (MyPID == 0) {
    printf("Matrix_Audit: p-lambda matrices\n");
    printf("   eigenvalues (min,max) = %8.3g %8.3g\n", emin, emax); 
    printf("   conditioning (min,max,avg) = %8.2g %8.2g %8.2g\n", cndmin, cndmax, cndavg);
  }
  return 0;
}


/* ******************************************************************
* AAA.                                      
****************************************************************** */
int Matrix_Audit::CheckSpectralBoundsSchurComplement()
{ 
  Teuchos::LAPACK<int, double> lapack;
  int info;
  double VL, VR;

  double emin = 1e+99, emax = -1e+99;
  double cndmin = 1e+99, cndmax = 1.0, cndavg = 0.0;

  for (int c = 0; c < A->size(); c++) {
    Teuchos::SerialDenseMatrix<int, double> Acell((*A)[c]);
    Epetra_SerialDenseVector& Acf = matrix_->Acf_cells()[c];
    Epetra_SerialDenseVector& Afc = matrix_->Afc_cells()[c];
    double& Acc = matrix_->Acc_cells()[c];
    int n = Acell.numRows();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        Acell(i, j) -= Afc[i] * Acf[j] / Acc;
      }
    }
    Teuchos::SerialDenseMatrix<int, double> Acopy(Acell);

    lapack.GEEV('N', 'N', n, Acell.values(), n, dmem1, dmem2, 
                &VL, 1, &VR, 1, dwork1, lwork1, &info);

    OrderByIncrease(n, dmem1);

    double e, a = dmem1[1], b = dmem1[1];  // skipping the first eigenvalue
    for (int k = 2; k < n; k++) {
      e = dmem1[k];
      a = std::min(a, e);
      b = std::max(b, e);
    }

    emin = std::min(emin, a);
    emax = std::max(emax, b);

    double cnd = b / a;
    cndmin = std::min(cndmin, cnd);
    cndmax = std::max(cndmax, cnd);
    cndavg += cnd;
  }
  cndavg /= A->size();

  if (MyPID == 0) {
    printf("Matrix_Audit: Schur complement matrices\n");
    printf("   eigenvalues (min,max) = %8.3g %8.3g\n", emin, emax); 
    printf("   conditioning (min,max,avg) = %8.2g %8.2g %8.2g\n", cndmin, cndmax, cndavg);
  }
  return 0;
}


/* ******************************************************************
* Verify symmetry of the matrix.                                      
****************************************************************** */
int Matrix_Audit::CheckMatrixSymmetry()
{
  const Epetra_Map& fmap_owned = mesh_->face_map(false);
  Epetra_Vector x(fmap_owned);
  Epetra_Vector y(fmap_owned);
  Epetra_Vector z(fmap_owned);

  double axy, ayx;
  int nfaces_owned = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::OWNED);

  for (int n = 0; n < 10; n++) {
    for (int f = 0; f < nfaces_owned; f++) {
      x[f] = double(random()) / RAND_MAX;
      y[f] = double(random()) / RAND_MAX;
    }
    matrix_->Aff()->Multiply(false, x, z);
    z.Dot(y, &axy);

    matrix_->Aff()->Multiply(false, y, z);
    z.Dot(x, &ayx);
    double err = fabs(axy - ayx) / (fabs(axy) + fabs(ayx) + 1e-10);
    if (MyPID == 0 && err > 1e-10) {	
      printf("   Summetry violation: (Ax,y)=%12.7g (Ay,x)=%12.7g\n", axy, ayx);
    }
  }
  return 0;
}


/* ******************************************************************
* Verify coercivity of the matrix.                                      
****************************************************************** */
int Matrix_Audit::CheckMatrixCoercivity()
{
  const Epetra_Map& fmap_owned = mesh_->face_map(false);
  Epetra_Vector x(fmap_owned);
  Epetra_Vector y(fmap_owned);

  double axx;
  int nfaces_owned = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::OWNED);

  for (int n = 0; n < 10; n++) {
    for (int f = 0; f < nfaces_owned; f++) {
      x[f] = double(random()) / RAND_MAX;
      y[f] = double(random()) / RAND_MAX;
    }
    matrix_->Aff()->Multiply(false, x, y);
    y.Dot(x, &axx);

    if (MyPID == 0 && axx <= 1e-12) {	
      printf("   Coercivity violation: (Ax,x)=%12.7g\n", axx);
    }
  }
  return 0;
}


/* ******************************************************************
* Bubble algorithm.                                            
****************************************************************** */
void Matrix_Audit::OrderByIncrease(int n, double* mem)
{
  for (int i = 0; i < n; i++) {
    for (int j = 1; j < n-i; j++) {
      if (mem[j-1] > mem[j]) {
         double tmp = mem[j];
         mem[j] = mem[j-1];
         mem[j-1] = tmp;
      }
    }
  }
}

}  // namespace Amanzi


