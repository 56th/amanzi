/*
  This is the Operator component of the Amanzi code.

  Copyright 2010-2013 held jointly by LANS/LANL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
  Ethan Coon (ecoon@lanl.gov)
*/

#include "DenseMatrix.hh"
#include "Op_Cell_Cell.hh"
#include "Op_Face_Cell.hh"

#include "SuperMap.hh"
#include "GraphFE.hh"
#include "MatrixFE.hh"
#include "Operator_Cell.hh"

/* ******************************************************************
Operator whose unknowns are CELL

See Operator_Cell.hh for more detail.
****************************************************************** */

namespace Amanzi {
namespace Operators {

/* ******************************************************************
* Apply a source which may or may not have cell volume included already. 
****************************************************************** */
void Operator_Cell::UpdateRHS(const CompositeVector& source,
                              bool volume_included)
{
  if (volume_included) {
    Operator::UpdateRHS(source);
  } else {
    Epetra_MultiVector& rhs_c = *rhs_->ViewComponent("cell", false);
    const Epetra_MultiVector& source_c = *source.ViewComponent("cell", false);
    for (int c = 0; c != ncells_owned; ++c) {
      rhs_c[0][c] += source_c[0][c] * mesh_->cell_volume(c);
    }
  }
}


/* ******************************************************************
* Visit methods for Apply.
* Apply the local matrices directly as schema is a subset of 
* assembled schema.
****************************************************************** */
int Operator_Cell::ApplyMatrixFreeOp(const Op_Cell_Cell& op,
                                     const CompositeVector& X, CompositeVector& Y) const
{
  ASSERT(op.vals.size() == ncells_owned);
  const Epetra_MultiVector& Xc = *X.ViewComponent("cell");
  Epetra_MultiVector& Yc = *Y.ViewComponent("cell");

  for (int c = 0; c != ncells_owned; ++c) {
    Yc[0][c] += Xc[0][c] * op.vals[c];
  }
  return 0;
}


/* ******************************************************************
* Apply the local matrices directly as schema is a subset of
* assembled schema
****************************************************************** */
int Operator_Cell::ApplyMatrixFreeOp(const Op_Face_Cell& op,
                                     const CompositeVector& X, CompositeVector& Y) const
{
  ASSERT(op.matrices.size() == nfaces_owned);
  
  X.ScatterMasterToGhosted();
  const Epetra_MultiVector& Xc = *X.ViewComponent("cell", true);

  Y.PutScalarGhosted(0.);
  Epetra_MultiVector& Yc = *Y.ViewComponent("cell", true);

  AmanziMesh::Entity_ID_List cells;
  for (int f=0; f!=nfaces_owned; ++f) {
    mesh_->face_get_cells(f, AmanziMesh::USED, &cells);
    int ncells = cells.size();

    WhetStone::DenseVector v(ncells), av(ncells);
    for (int n=0; n!=ncells; ++n) {
      v(n) = Xc[0][cells[n]];
    }

    const WhetStone::DenseMatrix& Aface = op.matrices[f];
    Aface.Multiply(v, av, false);

    for (int n=0; n!=ncells; ++n) {
      Yc[0][cells[n]] += av(n);
    }
  }

  Y.GatherGhostedToMaster("cell",Add);
  return 0;
}


/* ******************************************************************
* Visit methods for symbolic assemble.
* Insert the diagonal on cells
****************************************************************** */
void Operator_Cell::SymbolicAssembleMatrixOp(const Op_Cell_Cell& op,
                                             const SuperMap& map, GraphFE& graph,
                                             int my_block_row, int my_block_col) const
{
  const std::vector<int>& cell_row_inds = map.GhostIndices("cell", my_block_row);
  const std::vector<int>& cell_col_inds = map.GhostIndices("cell", my_block_col);

  int ierr(0);
  for (int c=0; c!=ncells_owned; ++c) {
    int row = cell_row_inds[c];
    int col = cell_col_inds[c];

    ierr |= graph.InsertMyIndices(row, 1, &col);
  }
  ASSERT(!ierr);
}


/* ******************************************************************
* Insert each cells neighboring cells.
****************************************************************** */
void Operator_Cell::SymbolicAssembleMatrixOp(const Op_Face_Cell& op,
                                             const SuperMap& map, GraphFE& graph,
                                             int my_block_row, int my_block_col) const
{
  // ELEMENT: face, DOF: cell
  int lid_r[2];
  int lid_c[2];
  const std::vector<int>& cell_row_inds = map.GhostIndices("cell", my_block_row);
  const std::vector<int>& cell_col_inds = map.GhostIndices("cell", my_block_col);

  int ierr(0);
  AmanziMesh::Entity_ID_List cells;
  for (int f=0; f!=nfaces_owned; ++f) {
    mesh_->face_get_cells(f, AmanziMesh::USED, &cells);
    
    int ncells = cells.size();
    for (int n=0; n!=ncells; ++n) {
      lid_r[n] = cell_row_inds[cells[n]];
      lid_c[n] = cell_col_inds[cells[n]];
    }

    ierr |= graph.InsertMyIndices(ncells, lid_r, ncells, lid_c);
  }
  ASSERT(!ierr);
}


/* ******************************************************************
* Visit methods for assemble
* Insert each cells neighboring cells.
****************************************************************** */
void Operator_Cell::AssembleMatrixOp(const Op_Cell_Cell& op,
                                     const SuperMap& map, MatrixFE& mat,
                                     int my_block_row, int my_block_col) const
{
  ASSERT(op.vals.size() == ncells_owned);

  const std::vector<int>& cell_row_inds = map.GhostIndices("cell", my_block_row);
  const std::vector<int>& cell_col_inds = map.GhostIndices("cell", my_block_col);

  int ierr(0);
  for (int c=0; c!=ncells_owned; ++c) {
    int row = cell_row_inds[c];
    int col = cell_col_inds[c];

    ierr |= mat.SumIntoMyValues(row, 1, &op.vals[c], &col);
  }
  ASSERT(!ierr);
}


void Operator_Cell::AssembleMatrixOp(const Op_Face_Cell& op,
                                     const SuperMap& map, MatrixFE& mat,
                                     int my_block_row, int my_block_col) const
{
  ASSERT(op.matrices.size() == nfaces_owned);
  
  // ELEMENT: face, DOF: cell
  int lid_r[2];
  int lid_c[2];
  const std::vector<int>& cell_row_inds = map.GhostIndices("cell", my_block_row);
  const std::vector<int>& cell_col_inds = map.GhostIndices("cell", my_block_col);

  int ierr(0);
  AmanziMesh::Entity_ID_List cells;
  for (int f=0; f!=nfaces_owned; ++f) {
    mesh_->face_get_cells(f, AmanziMesh::USED, &cells);
    
    int ncells = cells.size();
    for (int n=0; n!=ncells; ++n) {
      lid_r[n] = cell_row_inds[cells[n]];
      lid_c[n] = cell_col_inds[cells[n]];
    }

    ierr |= mat.SumIntoMyValues(lid_r, lid_c, op.matrices[f]);
    ASSERT(!ierr);
  }
  ASSERT(!ierr);
}

}  // namespace Operators
}  // namespace Amanzi

