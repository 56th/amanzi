#include <UnitTest++.h>
#include <iostream>

#include "../Mesh_MSTK.hh"


#include "Epetra_Map.h"
#include "Epetra_MpiComm.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_Array.hpp"

#include "mpi.h"


// Extract some surfaces as-is from 3D mesh

TEST(Extract_Column_MSTK)
{

  Teuchos::RCP<Epetra_MpiComm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));

  Teuchos::ParameterList reg_spec; // no regions declared here
  
  Amanzi::AmanziGeometry::GeometricModelPtr gm = new Amanzi::AmanziGeometry::GeometricModel(3, reg_spec, comm.get());

  // Generate a mesh consisting of 3x3x3 elements 

  Amanzi::AmanziMesh::Mesh_MSTK mesh(0,0,0,1,1,1,3,3,3,comm.get(),gm);

  CHECK_EQUAL(9,mesh.num_columns());

  int cell0 = 0;
  int colid = mesh.column_ID(cell0);
  Amanzi::AmanziMesh::Entity_ID_List const& cell_list = mesh.cells_of_column(colid);

  CHECK_EQUAL(3,cell_list.size());

  Amanzi::AmanziMesh::Mesh_MSTK column_mesh(mesh,cell_list,
                                            Amanzi::AmanziMesh::CELL,
                                            false,false);


  
  // Number of cells in column mesh

  int ncells_col = column_mesh.num_entities(Amanzi::AmanziMesh::CELL,
                                            Amanzi::AmanziMesh::OWNED);
  CHECK_EQUAL(3,ncells_col);


  // Check that their parents are as expected

  for (int i = 0; i < ncells_col; ++i) {
    int parent_cell = column_mesh.entity_get_parent(Amanzi::AmanziMesh::CELL,i);
    CHECK_EQUAL(cell_list[i], parent_cell);
  }


  // Once we can make RegionFactory work with reference counted pointers 
  // we can get rid of this code

  for (int i = 0; i < gm->Num_Regions(); i++)
    delete (gm->Region_i(i));
  delete gm;
  
}

