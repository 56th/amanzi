/*
The transport component of the Amanzi code. 

Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
Amanzi is released under the three-clause BSD License. 
The terms of use and "as is" disclaimer for this license are 
provided Reconstruction.cppin the top-level COPYRIGHT file.

Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <algorithm>
#include <cmath>
#include <vector>

#include "Epetra_Vector.h"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

#include "Point.hh"

#include "Mesh.hh"
#include "Transport_PK.hh"
#include "Reconstruction.hh"

namespace Amanzi {
namespace AmanziTransport {

/* We set up most popular parameters here. */
void Reconstruction::Init()
{
  status = RECONSTRUCTION_NULL;
  const Epetra_Map& cmap = mesh_->cell_map(true);
  const Epetra_Map& fmap = mesh_->face_map(true);

  cmax = cmap.MaxLID();

  number_owned_cells = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED);
  cmax_owned = number_owned_cells - 1;

  fmin = fmap.MinLID();
  fmax = fmap.MaxLID();

  number_owned_faces = mesh_->num_entities(AmanziMesh::FACE, AmanziMesh::OWNED);
  fmax_owned = fmin + number_owned_faces - 1;

  Teuchos::SerialDenseMatrix<int, double> matrix(TRANSPORT_MAX_FACES,
                                                 TRANSPORT_MAX_FACES);

  dim = mesh_->space_dimension();
  gradient_ = CreateCompositeVector(mesh_, AmanziMesh::CELL, dim, true);


  status = RECONSTRUCTION_INIT;
}


/* ******************************************************************
* Implementation is tuned up for gradient (first-order reconstruction).
* It can be extended easily if needed in the future.
****************************************************************** */
void Reconstruction::CalculateCellGradient()
{
  Teuchos::RCP<Epetra_MultiVector> grad = gradient_->ViewComponent("cell", false);

  Epetra_Vector& u = *scalar_field_;  // a few aliases
  Teuchos::LAPACK<int, double> lapack;

  AmanziMesh::Entity_ID_List cells;
  AmanziGeometry::Point xcc(dim);

  double *rhs;
  rhs = new double[dim];

  for (int c = 0; c <= cmax_owned; c++) {
    matrix.shape(dim, dim);  // Teuchos will initilize this matrix by zeros
    for (int i = 0; i < dim; i++) rhs[i] = 0.0;

    mesh_->cell_get_face_adj_cells(c, AmanziMesh::USED, &cells);
    const AmanziGeometry::Point& xc = mesh_->cell_centroid(c);

    for (int n = 0; n < cells.size(); n++) {
      const AmanziGeometry::Point& xc2 = mesh_->cell_centroid(cells[n]);
      for (int i = 0; i < dim; i++) xcc[i] = xc2[i] - xc[i];

      double value = u[cells[n]] - u[c];
      populateLeastSquareSystem(xcc, value, matrix, rhs);
    }

    // improve robustness w.r.t degenerate matrices
    double det = calculateMatrixDeterminant(matrix);
    double norm = calculateMatrixNorm(matrix);

    if (det < pow(norm, 1.0/dim)) {
      norm *= RECONSTRUCTION_MATRIX_CORRECTION;
      for (int i = 0; i < dim; i++) matrix(i, i) += norm;
    }
    // printLeastSquareSystem(matrix, rhs);

    int info;
    lapack.POSV('U', dim, 1, matrix.values(), dim, rhs, dim, &info);
    if (info) {  // reduce reconstruction order
      for (int i = 0; i < dim; i++) rhs[i] = 0.0;
    }

    // rhs[0] = rhs[1] = rhs[2] = 0.0;  // TESTING COMPATABILITY
    for (int i = 0; i < dim; i++) (*grad)[i][c] = rhs[i];
  }

  delete [] rhs;

  gradient_->ScatterMasterToGhosted("cell");
}


/* ******************************************************************
 * The limiter must be between 0 and 1
****************************************************************** */
void Reconstruction::applyLimiter(Teuchos::RCP<Epetra_Vector>& limiter)
{
  Teuchos::RCP<Epetra_MultiVector> grad = gradient_->ViewComponent("cell", false);

  for (int c = 0; c <= cmax; c++) {
    for (int i = 0; i < dim; i++) (*grad)[i][c] *= (*limiter)[c];
  }
}


/* ******************************************************************
 * calculates a value at point p using gradinet and centroid
****************************************************************** */
double Reconstruction::getValue(const int cell, const AmanziGeometry::Point& p)
{
  Teuchos::RCP<Epetra_MultiVector> grad = gradient_->ViewComponent("cell", false);
  const AmanziGeometry::Point& xc = mesh_->cell_centroid(cell);

  double value = (*scalar_field_)[cell];
  for (int i = 0; i < dim; i++) value += (*grad)[i][cell] * (p[i] - xc[i]);

  return value;
}


/* ******************************************************************
 * calculates a value at point p using gradinet and centroid
****************************************************************** */
double Reconstruction::getValue(
    AmanziGeometry::Point& gradient, const int cell, const AmanziGeometry::Point& p)
{
  const AmanziGeometry::Point& xc = mesh_->cell_centroid(cell);

  double value = (*scalar_field_)[cell];
  for (int i = 0; i < dim; i++) value += gradient[i] * (p[i] - xc[i]);

  return value;
}


/* ******************************************************************
 * Assemble a SPD least square matrix
****************************************************************** */
void Reconstruction::populateLeastSquareSystem(AmanziGeometry::Point& centroid,
                                               double field_value,
                                               Teuchos::SerialDenseMatrix<int, double>& matrix,
                                               double* rhs)
{
  for (int i = 0; i < dim; i++) {
    double xyz = centroid[i];

    matrix(i, i) += xyz * xyz;
    for (int j = i+1; j < dim; j++) matrix(j, i) = matrix(i, j) += xyz * centroid[j];

    rhs[i] += xyz * field_value;
  }
}


/* ******************************************************************
 * Optimized linear algebra: norm
****************************************************************** */
double Reconstruction::calculateMatrixNorm(Teuchos::SerialDenseMatrix<int, double>& matrix)
{
  double a = 0.0;
  for (int i = 0; i < dim; i++) {
    for (int j = i; j < dim; j++) a = std::max(a, matrix(i, j));
  }
  return a;
}


/* ******************************************************************
 * Optimized linear algebra: determinant
****************************************************************** */
double Reconstruction::calculateMatrixDeterminant(Teuchos::SerialDenseMatrix<int, double>& matrix)
{
  double a = 0.0;
  if (dim == 2) {
    a = matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(0, 1);
  } else if (dim == 3) {
    a = matrix(0, 0) * matrix(1, 1) * matrix(2, 2)
      + matrix(0, 1) * matrix(1, 2) * matrix(2, 0) * 2
      - matrix(0, 2) * matrix(1, 1) * matrix(2, 0)
      - matrix(0, 1) * matrix(1, 0) * matrix(2, 2)
      - matrix(0, 0) * matrix(1, 2) * matrix(2, 1);
  } else {
    a = matrix(0, 0);
  }
  return a;
}


/* ******************************************************************
 * Search routine.
****************************************************************** */
int Reconstruction::findMinimalDiagonalEntry(Teuchos::SerialDenseMatrix<int, double>& matrix)
{
  double a = matrix(0, 0);  // We assume that matrix is SPD.
  int k = 0;

  for (int i = 1; i < dim; i++) {
    double b = matrix(i, i);
    if (b < a) {
      a = b;
      k = i;
    }
  }
  return k;
}


/* ******************************************************************
 * IO routines
****************************************************************** */
void Reconstruction::printLeastSquareSystem(Teuchos::SerialDenseMatrix<int, double>matrix, double* rhs)
{
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) std::printf("%6.3f ", matrix(i, j));
    std::printf("  f[%1d] =%8.5f\n", i, rhs[i]);
  }
  std::printf("\n");
}


}  // namespace AmanziTransport
}  // namespace Amanzi

