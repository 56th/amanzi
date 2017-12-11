#include <UnitTest++.h>

#include <iostream>

#include "../Mesh_MOAB.hh"


#include "Epetra_Map.h"
#include "Epetra_MpiComm.h"

#include "mpi.h"


TEST(MOAB_HEX_3x3x3_4P)
{
  int i, j, k, err, nc, nf, nv;
  unsigned int faces[6], nodes[8];
  int facedirs[6];
  double ccoords[24], fcoords[12];

  int NVowned[4] = {16,16,16,16};
  int NFowned[4] = {16,26,26,40};
  int NCowned[4] = {3,6,6,12};
  int NVused[4] = {36,48,48,64};
  int NFused[4] = {52,75,75,108};
  int NCused[4] = {12,18,18,27};
  int NVghost[4] = {20,32,32,48};
  int NFghost[4] = {36,49,49,68};
  int NCghost[4] = {9,12,12,15};

			      
  std::shared_ptr<Epetra_MpiComm> comm_(new Epetra_MpiComm(MPI_COMM_WORLD));			      

  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  CHECK_EQUAL(4,size);

  if (rank == 0) {
    int DebugWait = 0;
    while (DebugWait);
  }

  // Load a single hex from the hex1.exo file

  Amanzi::AmanziMesh::Mesh_MOAB mesh("test/hex_3x3x3_ss_4P.h5m",comm_.get());


  nv = mesh.num_entities(Amanzi::AmanziMesh::NODE,Amanzi::AmanziMesh::OWNED);  
  CHECK_EQUAL(NVowned[rank],nv);
  
  nf = mesh.num_entities(Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::OWNED);  
  CHECK_EQUAL(NFowned[rank],nf);
  
  nc = mesh.num_entities(Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::OWNED);
  CHECK_EQUAL(NCowned[rank],nc);

  nv = mesh.num_entities(Amanzi::AmanziMesh::NODE,Amanzi::AmanziMesh::USED);  
  CHECK_EQUAL(NVused[rank],nv);
  
  nf = mesh.num_entities(Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::USED);  
  CHECK_EQUAL(NFused[rank],nf);
  
  nc = mesh.num_entities(Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::USED);
  CHECK_EQUAL(NCused[rank],nc);

  nv = mesh.num_entities(Amanzi::AmanziMesh::NODE,Amanzi::AmanziMesh::GHOST);  
  CHECK_EQUAL(NVghost[rank],nv);
  
  nf = mesh.num_entities(Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::GHOST);  
  CHECK_EQUAL(NFghost[rank],nf);
  
  nc = mesh.num_entities(Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::GHOST);
  CHECK_EQUAL(NCghost[rank],nc);


  Amanzi::AmanziMesh::Entity_ID_List  c2f;
  std::vector<int> c2fdirs;
  Epetra_Map cell_map(mesh.cell_map(false));
  Epetra_Map face_map(mesh.face_map(true));

  for (int c=cell_map.MinLID(); c<=cell_map.MaxLID(); c++)
    {
      CHECK_EQUAL(cell_map.GID(c),mesh.GID(c,Amanzi::AmanziMesh::CELL));
      mesh.cell_get_faces_and_dirs( c, &c2f, &c2fdirs, true );

      for (int j=0; j<6; j++)
	{
	  int f = face_map.LID(mesh.GID(c2f[j],Amanzi::AmanziMesh::FACE));
	  CHECK_EQUAL( f,c2f[j] );
	  CHECK_EQUAL(1,abs(c2fdirs[j]));
	}

    }
  

  // Verify cell sets

  //  int ns;
  //  ns = mesh.num_sets(Amanzi::AmanziMesh::CELL);
  //  CHECK_EQUAL(3,ns);

  //  std::vector<unsigned int> csetids(3);
  //  unsigned int expcsetids[3] = {10000,20000,30000};

  //  mesh.get_set_ids(Amanzi::AmanziMesh::CELL,&csetids);

  //  CHECK_ARRAY_EQUAL(expcsetids,csetids,3);

}

