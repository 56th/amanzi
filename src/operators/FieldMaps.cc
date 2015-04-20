/*
  This is the operators component of the Amanzi code. 

  Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)

  Collection of non-member functions f2 = Map(f1, f2) where 
  Map() connects fields living on different geometric objects.
*/

#ifndef AMANZI_OPERATORS_FIELD_MAPS_HH_
#define AMANZI_OPERATORS_FIELD_MAPS_HH_

#include "Teuchos_RCP.hpp"

#include "CompositeVector.hh"

#include "FieldMaps.hh"

namespace Amanzi {
namespace Operators {

/* ******************************************************************
* f2 = f2 * Map(f1)
****************************************************************** */
int CellToFace_Scale(Teuchos::RCP<CompositeVector>& f1,
                     Teuchos::RCP<CompositeVector>& f2)
{
}


/* ******************************************************************
* f2 = Map(f1, f2):
*   cell comp:  f2_cell = f2_cell / f1_cell
*   face comp:  f2_face = f2_face / FaceAverage(f1_cell)
****************************************************************** */
int CellToFace_ScaleInverse(Teuchos::RCP<const CompositeVector> f1,
                            Teuchos::RCP<CompositeVector>& f2)
{
  ASSERT(f1->HasComponent("cell"));
  ASSERT(f2->HasComponent("cell") && f2->HasComponent("face"));

  f1->ScatterMasterToGhosted("cell");

  const Epetra_MultiVector& f1c = *f1->ViewComponent("cell", true);
  Epetra_MultiVector& f2c = *f2->ViewComponent("cell", true);
  Epetra_MultiVector& f2f = *f2->ViewComponent("face", true);

  AmanziMesh::Entity_ID_List cells;
  Teuchos::RCP<const AmanziMesh::Mesh> mesh = f1->Map().Mesh();

  // cell-part of the map
  int ncells_wghost = mesh->num_entities(AmanziMesh::CELL, AmanziMesh::USED);
  for (int c = 0; c < ncells_wghost; ++c) {
    f2c[0][c] /= f1c[0][c]; 
  }

  // face-part of the map
  int nfaces_wghost = mesh->num_entities(AmanziMesh::FACE, AmanziMesh::USED);
  for (int f = 0; f < nfaces_wghost; ++f) {
    mesh->face_get_cells(f, AmanziMesh::USED, &cells);
    int ncells = cells.size();

    double tmp(0.0);
    for (int n = 0; n < ncells; ++n) tmp += f1c[0][cells[n]];
    f2f[0][f] /= (tmp / ncells); 
  }
}

}  // namespace Operators
}  // namespace Amanzi


#endif
