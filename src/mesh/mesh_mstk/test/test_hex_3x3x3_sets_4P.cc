#include <UnitTest++.h>

#include <iostream>

#include "../Mesh_MSTK.hh"


#include "Epetra_Map.h"
#include "Epetra_MpiComm.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"


TEST(MSTK_HEX_3x3x3_SETS_4P)
{
  int rank, size;

  std::string expcsetnames[8] = {"Bottom LS", "Middle LS", "Top LS", 
                                 "Bottom+Middle Box", "Top Box",
                                 "Bottom ColFunc", "Middle ColFunc", "Top ColFunc"};

  int csetsize;
  
  int expcsetcells[4][8][9];


  std::string expfsetnames[4] = {"Face 101",  
				  "Face 30004",
                                  "ZLO FACE Plane", 
				 "YLO FACE Box"};

  int fsetsize;

  Teuchos::RCP<Epetra_MpiComm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));

  
  int initialized;
  MPI_Initialized(&initialized);
  
  if (!initialized)
    MPI_Init(NULL,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  if (size != 4) {
    std::cerr << "Test must be run with 4 processors" << std::endl;
  }
  CHECK_EQUAL(4,size);


  std::string infilename = "test/hex_3x3x3_4P.xml";
  Teuchos::ParameterXMLFileReader xmlreader(infilename);

  Teuchos::ParameterList reg_spec(xmlreader.getParameters());

  Amanzi::AmanziGeometry::GeometricModelPtr gm = new Amanzi::AmanziGeometry::GeometricModel(3, reg_spec, comm.get());

  // Load a mesh consisting of 3x3x3 elements

  Teuchos::RCP<Amanzi::AmanziMesh::Mesh> mesh(new Amanzi::AmanziMesh::Mesh_MSTK("test/hex_3x3x3_sets.exo",comm.get(),3,gm));

  Teuchos::ParameterList::ConstIterator i;
  for (i = reg_spec.begin(); i != reg_spec.end(); i++) {
        const std::string reg_name = reg_spec.name(i);     

    Teuchos::ParameterList reg_params = reg_spec.sublist(reg_name);

    // See if the geometric model has a region by this name
  
    Amanzi::AmanziGeometry::RegionPtr reg = gm->FindRegion(reg_name);

    CHECK(reg != NULL);

    // Do their names match ?

    CHECK_EQUAL(reg->name(),reg_name);


    // Get the region info directly from the XML and compare
  
    Teuchos::ParameterList::ConstIterator j = reg_params.begin(); 

    std::string shape = reg_params.name(j);

    if (shape == "Region: Plane") {

      // Do we have a valid sideset by this name

      CHECK(mesh->valid_set_name(reg_name,Amanzi::AmanziMesh::FACE));

      int j;
      for (j = 0; j < 4; j++) {
        if (expfsetnames[j] == reg_name) break;
      }

      CHECK(j < 4);


      // Verify that we can get the number of entities in the set

      int set_size = mesh->get_set_size(reg_name,Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::OWNED);


      // Verify that we can get the set entities
     
      Amanzi::AmanziMesh::Entity_ID_List setents;
      mesh->get_set_entities(reg_name,Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::OWNED,&setents);

    }
    else if (shape == "Region: Box") {

      Teuchos::ParameterList box_params = reg_params.sublist(shape);
      Teuchos::Array<double> pmin = box_params.get< Teuchos::Array<double> >("Low Coordinate");
      Teuchos::Array<double> pmax = box_params.get< Teuchos::Array<double> >("High Coordinate");

      if (pmin[0] == pmax[0] || pmin[1] == pmax[1] || pmin[2] == pmax[2])
	{
	  // This is a reduced dimensionality box - request a faceset

	  // Do we have a valid sideset by this name

	  CHECK(mesh->valid_set_name(reg_name,Amanzi::AmanziMesh::FACE));
	  
	  int j;
	  for (j = 0; j < 4; j++) {
	    if (expfsetnames[j] == reg_name) break;
	  }
	  
	  CHECK(j < 4);
	  
	  
	  // Verify that we can get the number of entities in the set
	  
	  int set_size = mesh->get_set_size(reg_name,Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::OWNED);

	  
	  // Verify that we can get the correct set entities
	  
	  Amanzi::AmanziMesh::Entity_ID_List setents;
	  mesh->get_set_entities(reg_name,Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::OWNED,&setents);
	  
	}
      else 
	{
	  // Do we have a valid cellset by this name
	  
	  CHECK(mesh->valid_set_name(reg_name,Amanzi::AmanziMesh::CELL));
	  
	  // Find the expected cell set info corresponding to this name 
	  
	  int j;
	  for (j = 0; j < 8; j++)
	    if (reg_name == expcsetnames[j]) break;
	  
	  CHECK(j < 8);
	  
	  // Verify that we can get the number of entities in the set
	  
	  int set_size = mesh->get_set_size(reg_name,Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::OWNED);
	  
	  
	  // Verify that we can get the set entities
	  
	  Amanzi::AmanziMesh::Entity_ID_List setents;
	  mesh->get_set_entities(reg_name,Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::OWNED,&setents);
	  
	}
    }
    else if (shape == "Region: Labeled Set") {

      Teuchos::ParameterList lsparams = reg_params.sublist(shape);

      // Find the entity type in this parameter list

      std::string entity_type = lsparams.get<std::string>("Entity");

      if (entity_type == "Face") {

	// Do we have a valid sideset by this name

	CHECK(mesh->valid_set_name(reg_name,Amanzi::AmanziMesh::FACE));

        // Find the expected face set info corresponding to this name

        int j;
        for (j = 0; j < 4; j++)
          if (reg_name == expfsetnames[j]) break;

	if (j >= 4) 
	  std::cerr << "Cannot find regname " << reg_name << "on processor " << rank << std::endl;
        CHECK(j < 4);
	
	// Verify that we can get the number of entities in the set
	
	int set_size = mesh->get_set_size(reg_name,Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::OWNED);

	// Verify that we can get the correct set entities
	
        Amanzi::AmanziMesh::Entity_ID_List setents;
	mesh->get_set_entities(reg_name,Amanzi::AmanziMesh::FACE,Amanzi::AmanziMesh::OWNED,&setents);

      }
      else if (entity_type == "Cell") {

	// Do we have a valid sideset by this name

	CHECK(mesh->valid_set_name(reg_name,Amanzi::AmanziMesh::CELL));
	
        // Find the expected face set info corresponding to this name

        int j;
        for (j = 0; j < 8; j++)
          if (reg_name == expcsetnames[j]) break;

        CHECK(j < 8);
	
	// Verify that we can get the number of entities in the set
	
	int set_size = mesh->get_set_size(reg_name,Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::OWNED);

	
	// Verify that we can get the set entities
	
        Amanzi::AmanziMesh::Entity_ID_List setents;
	mesh->get_set_entities(reg_name,Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::OWNED,&setents);

      }

    }
    else if (shape == "Region: Color Function") {

      // Do we have a valid cellset by this name

      CHECK(mesh->valid_set_name(reg_name,Amanzi::AmanziMesh::CELL));
	
      // Find the expected cell set info corresponding to this name

      int j;
      for (j = 0; j < 8; j++)
        if (reg_name == expcsetnames[j]) break;

      CHECK(j < 8);
	
      // Verify that we can get the number of entities in the set
	
      int set_size = mesh->get_set_size(reg_name,Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::OWNED);

	
      // Verify that we can get the set entities
	
      Amanzi::AmanziMesh::Entity_ID_List setents;
      mesh->get_set_entities(reg_name,Amanzi::AmanziMesh::CELL,Amanzi::AmanziMesh::OWNED,&setents);

    }
  }


  // Once we can make RegionFactory work with reference counted pointers 
  // we can get rid of this code

  for (int i = 0; i < gm->Num_Regions(); i++)
    delete (gm->Region_i(i));
  delete gm;

}

