#include <UnitTest++.h>

#include <iostream>

#include "../Mesh_MSTK.hh"

#include "Epetra_Map.h"
#include "Epetra_MpiComm.h"

#include "MeshAudit.hh"


TEST(MSTK_QUAD_GEN_3x3_4P)
{

  int i, j, k, err, nc, nf, nv;
  std::vector<Amanzi::AmanziMesh::Entity_ID> faces(6), nodes(8);
  std::vector<int> facedirs(4);
  std::vector<Amanzi::AmanziGeometry::Point> ccoords(8), fcoords(4);

  Teuchos::RCP<Epetra_MpiComm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
			      
  int rank, size;

  int initialized;
  MPI_Initialized(&initialized);
  
  if (!initialized)
    MPI_Init(NULL,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  CHECK_EQUAL(4,size);

  if (size != 4) {
    std::cerr << "Test must be run with 4 processors" << std::endl;
  }


  // Load a single hex from the hex1.exo file

  Teuchos::RCP<Amanzi::AmanziMesh::Mesh> mesh(new Amanzi::AmanziMesh::Mesh_MSTK(0.0,0.0,1.0,1.0,3,3,comm.get()));



  std::vector<Amanzi::AmanziMesh::Entity_ID>  c2f(4);
  std::vector<int> c2fdirs(4);
  Epetra_Map cell_map(mesh->cell_map(false));
  Epetra_Map face_map(mesh->face_map(true));

  for (int c=cell_map.MinLID(); c<=cell_map.MaxLID(); c++)
    {
      CHECK_EQUAL(cell_map.GID(c),mesh->GID(c,Amanzi::AmanziMesh::CELL));
      mesh->cell_get_faces_and_dirs(c, &c2f, &c2fdirs, true);

      for (int j=0; j<4; j++)
	{
	  int f = face_map.LID(mesh->GID(c2f[j],Amanzi::AmanziMesh::FACE));
	  CHECK_EQUAL( f,c2f[j] );
	  if (f != c2f[j]) {
	    std::cout << std::endl;
	    std::cout << "Processor ID " << rank << std::endl;
	    std::cout << "Cell ID " << cell_map.GID(c) << std::endl;
	    std::cout << "Problem face c2f[j] = " << c2f[j] << " GID = " << mesh->GID(c2f[j],Amanzi::AmanziMesh::FACE) << " f = " << f << std::endl;
	    std::cout << std::endl;
	  }
	}

    }

  std::stringstream fname;
  fname << "test/mstk_quad_gen_3x3_4P." << rank << ".out";
  std::ofstream fout(fname.str().c_str());
  Amanzi::MeshAudit auditor(mesh,fout);
  auditor.Verify();


}

