//#include <Teuchos_RCP.hpp>

#include "dbc.hh"
#include "errors.hh"
#include "Mesh_MSTK.hh"

using namespace std;


namespace Amanzi
{

namespace AmanziMesh
{

  char kind_to_string[4][256] = {"NODE","EDGE","FACE","CELL"};

//--------------------------------------
// Constructor - load up mesh from file
//--------------------------------------

Mesh_MSTK::Mesh_MSTK (const char *filename, const Epetra_MpiComm *incomm,
		      const AmanziGeometry::GeometricModelPtr& gm,
                      const VerboseObject * verbobj) :
  Mesh(verbobj), mpicomm(incomm->GetMpiComm()), meshxyz(NULL), 
  target_cell_volumes(NULL), min_cell_volumes(NULL)
{  

  int numprocs = incomm->NumProc();

  // Assume three dimensional problem if constructor called without 
  // the space_dimension parameter

  int ok;

#ifdef DEBUG
  if (verbobj && verbobj->os_OK(Teuchos::VERB_MEDIUM)) {
      *(verbobj->os()) << "Testing Verbosity !!!! - Construct mesh from file" << std::endl;
  }
#endif

  // Pre-processing (init, MPI queries etc)

  int space_dim = 3;
  pre_create_steps_(space_dim, incomm, gm);



  if (myprocid == 0) {
    int DebugWait=0;
    while (DebugWait);
  }


  mesh = MESH_New(F1);

  int len = strlen(filename);
  if (len > 4 && strncmp(&(filename[len-4]),".exo",4) == 0) { // Exodus file 

    if (numprocs == 1) {
      ok = MESH_ImportFromExodusII(mesh,filename,NULL,mpicomm);
    }
    else {
      int opts[5] = {0,0,0,0,0};

      opts[0] = 1;   // Partition the input mesh
      opts[1] = 0;   // Use the default method for distributing the mesh
      opts[2] = 1;   // Number of ghost layers
      opts[3] = 1;   // Use Zoltan for partitioning if available

      ok = MESH_ImportFromExodusII(mesh,filename,opts,mpicomm);
    }

  }
  else if (len > 4 && strncmp(&(filename[len-4]),".par",4) == 0) { // Nemesis file 
    int opts[5] = {0,0,0,0,0};

    opts[0] = 1;     // Parallel weave distributed meshes
    opts[1] = 1;     // Number of ghost layers

    ok = MESH_ImportFromNemesisI(mesh,filename,opts,mpicomm);

  }
  else {
    std::stringstream mesg_stream;
    mesg_stream << "Cannot identify file type from extension of input file " << filename << " on processor " << myprocid << std::endl;
    Errors::Message mesg(mesg_stream.str());
    amanzi_throw(mesg);
  }

  if (!ok) {
    std::stringstream mesg_stream;
    mesg_stream << "Failed to load " << filename << " on processor " << myprocid << std::endl;
    Errors::Message mesg(mesg_stream.str());
    amanzi_throw(mesg);
  }

  int cell_dim = MESH_Num_Regions(mesh) ? 3 : 2;
  
  int max;
  incomm->MaxAll(&cell_dim,&max,1);

  if (max != cell_dim) {
    Errors::Message mesg("cell dimension on this processor is different from max cell dimension across all processors");
    amanzi_throw(mesg);
  }

  set_cell_dimension(cell_dim);

  if (cell_dim == 2 && space_dim == 3) {
      
    // Check if this is a completely planar mesh 
    // in which case one can label the space dimension as 2
    
    MVertex_ptr mv, mv0 = MESH_Vertex(mesh,0);
    double vxyz[3], z0;
    
    MV_Coords(mv0,vxyz);
    z0 = vxyz[2];
    
    bool planar = true;
    int idx = 0;
    while ((mv = MESH_Next_Vertex(mesh,&idx))) {
      MV_Coords(mv,vxyz);
      if (z0 != vxyz[2]) {
        planar = false;
        break;
      }
    }
    
    if (planar)
      space_dim = 2;

    incomm->MaxAll(&space_dim,&max,1);

    space_dim = max;
    set_space_dimension(space_dim);      
  }

  // Do all the processing required for setting up the mesh for Amanzi 
  
  post_create_steps_();

}


//--------------------------------------
// Constructor - load up mesh from file
//--------------------------------------

Mesh_MSTK::Mesh_MSTK (const char *filename, const Epetra_MpiComm *incomm, 
		      int space_dimension,
		      const AmanziGeometry::GeometricModelPtr& gm,
                      const VerboseObject * verbobj) :
  Mesh(verbobj), mpicomm(incomm->GetMpiComm()), meshxyz(NULL), 
  target_cell_volumes(NULL), min_cell_volumes(NULL)
{

  // Assume three dimensional problem if constructor called without 
  // the space_dimension parameter

  int ok;


  // Pre-processing (init, MPI queries etc)

  int space_dim = 3;
  pre_create_steps_(space_dim, incomm, gm);



  if (myprocid == 0) {
    int DebugWait=0;
    while (DebugWait);
  }

  mesh = MESH_New(F1);

  int len = strlen(filename);
  if (len > 4 && strncmp(&(filename[len-4]),".exo",4) == 0) { // Exodus file

    if (numprocs == 1) {
      ok = MESH_ImportFromExodusII(mesh,filename,NULL,mpicomm);
    }
    else {
      int opts[5] = {0,0,0,0,0};

      opts[0] = 1;   // Partition the input mesh
      opts[1] = 0;   // Use the default method for distributing the mesh
      opts[2] = 1;   // Number of ghost layers
      opts[3] = 1;   // Use Zoltan for partitioning if available

      ok = MESH_ImportFromExodusII(mesh,filename,opts,mpicomm);
    }

  }
  else if (len > 4 && strncmp(&(filename[len-4]),".par",4) == 0) { // Nemesis file 
    int opts[5] = {0,0,0,0,0};

    opts[0] = 1;     // Parallel weave distributed meshes
    opts[1] = 1;     // Number of ghost layers

    ok = MESH_ImportFromNemesisI(mesh,filename,opts,mpicomm);

  }
  else {
    std::stringstream mesg_stream;
    mesg_stream << "Cannot identify file type from extension of input file " << filename << " on processor " << myprocid << std::endl;
    Errors::Message mesg(mesg_stream.str());
    amanzi_throw(mesg);
  }

  if (!ok) {
    std::stringstream mesg_stream;
    mesg_stream << "Failed to load " << filename << " on processor " << myprocid << std::endl;
    Errors::Message mesg(mesg_stream.str());
    amanzi_throw(mesg);
  }

  int cell_dim = MESH_Num_Regions(mesh) ? 3 : 2;
  
  int max;
  incomm->MaxAll(&cell_dim,&max,1);

  if (max != cell_dim) {
    Errors::Message mesg("cell dimension on this processor is different from max cell dimension across all processors");
    amanzi_throw(mesg);
  }

  set_cell_dimension(cell_dim);

  // Do all the processing required for setting up the mesh for Amanzi 
  
  post_create_steps_();

}


//--------------------------------------
// Construct a 3D regular hexahedral mesh internally
//--------------------------------------


Mesh_MSTK::Mesh_MSTK(const double x0, const double y0, const double z0,
		     const double x1, const double y1, const double z1,
		     const unsigned int nx, const unsigned int ny, 
		     const unsigned int nz, 
		     const Epetra_MpiComm *incomm,
		     const AmanziGeometry::GeometricModelPtr& gm,
                     const VerboseObject * verbobj) :
  Mesh(verbobj), mpicomm(incomm->GetMpiComm()), meshxyz(NULL), 
  target_cell_volumes(NULL), min_cell_volumes(NULL)
{
  int ok;

  int space_dimension = 3;
  pre_create_steps_(space_dimension, incomm, gm);


  Mesh::set_mesh_type(RECTANGULAR);   // Discretizations can use this info if they want


  if (serial_run) {

    // Load serial mesh

    mesh = MESH_New(F1);
    ok = generate_regular_mesh(mesh,x0,y0,z0,x1,y1,z1,nx,ny,nz);

    set_cell_dimension(3);

    myprocid = 0;
  }
  else {
    Mesh_ptr globalmesh;
    int topo_dim=3; // What is the topological dimension of the mesh
    int ring = 1; // One layer of ghost cells in parallel meshes
    int with_attr = 1;  // update of attributes in parallel meshes
    int del_inmesh = 1; // delete input mesh as soon as possible
    int method = 1; /* Partition with ZOLTAN */

    
    if (myprocid == 0) {
      globalmesh = MESH_New(F1);

      ok = generate_regular_mesh(globalmesh,x0,y0,z0,x1,y1,z1,nx,ny,nz);
      
      topo_dim = (MESH_Num_Regions(globalmesh) == 0) ? 2 : 3;
    }
    else {
      globalmesh = NULL;
      ok = 1;
    }

#ifdef MSTK_2_21rc1_OR_NEWER
    ok = ok & MSTK_Mesh_Distribute(globalmesh,&mesh,&topo_dim,ring,with_attr,
                                   method,del_inmesh,mpicomm);
#else
    ok = ok & MSTK_Mesh_Distribute(globalmesh,&mesh,&topo_dim,ring,with_attr,
                                   method,mpicomm);
    if (myprocid == 0)
      MESH_Delete(globalmesh);
#endif

    set_cell_dimension(topo_dim);
  }

  if (!ok) {
    std::stringstream mesg_stream;
    mesg_stream << "Failed to generate mesh on processor " << myprocid;
    Errors::Message mesg(mesg_stream.str());
    amanzi_throw(mesg);
    assert(ok);
  }




  // Do all the processing required for setting up the mesh for Amanzi 
  
  post_create_steps_();

}



//--------------------------------------
// Construct a 2D regular quadrilateral mesh internally
//--------------------------------------


Mesh_MSTK::Mesh_MSTK(const double x0, const double y0,
		     const double x1, const double y1,
		     const int nx, const int ny, 
		     const Epetra_MpiComm *incomm,
		     const AmanziGeometry::GeometricModelPtr& gm,
                     const VerboseObject *verbobj) :
  Mesh(verbobj), mpicomm(incomm->GetMpiComm()), meshxyz(NULL), 
  target_cell_volumes(NULL), min_cell_volumes(NULL)
{
  int ok;
  int space_dim = 2;
  pre_create_steps_(space_dim, incomm, gm);




  if (myprocid == 0) {
    int DebugWait=0;
    while (DebugWait);
    }

#ifdef DEBUG
  if (Mesh::verbosity_obj()) {
    if (verbosity_obj()->os_OK(Teuchos::VERB_MEDIUM)) {
      verbosity_obj()->os() << "Testing Verbosity !!!! - Construct mesh from low/hi coords - 2D" << std::endl;
    }
  }
#endif


  Mesh::set_mesh_type(RECTANGULAR);   // Discretizations can use this info if they want


  int topo_dim=space_dim; // What is the topological dimension of the mesh
  set_cell_dimension(topo_dim);

  if (serial_run) {

    // Load serial mesh

    mesh = MESH_New(F1);
    ok = generate_regular_mesh(mesh,x0,y0,x1,y1,nx,ny);

    myprocid = 0;
  }
  else {
    Mesh_ptr globalmesh;
    int ring = 1; // One layer of ghost cells in parallel meshes
    int with_attr = 1;  // update of attributes in parallel meshes
    int del_inmesh = 1; // delete input mesh at the earliest
    int method = 1; /* Partition with ZOLTAN */

    if (myprocid == 0) {
      globalmesh = MESH_New(F1);

      ok = generate_regular_mesh(globalmesh,x0,y0,x1,y1,nx,ny);
      
      topo_dim = (MESH_Num_Regions(globalmesh) == 0) ? 2 : 3;
    }
    else {
      globalmesh = NULL;
      ok = 1;
    }

#ifdef MSTK_2_21rc1_OR_NEWER
    ok = ok & MSTK_Mesh_Distribute(globalmesh,&mesh,&topo_dim,ring,with_attr,
                                   method,del_inmesh,mpicomm);
#else
    ok = ok & MSTK_Mesh_Distribute(globalmesh,&mesh,&topo_dim,ring,with_attr,
                                   method,mpicomm);
    if (myprocid == 0)
      MESH_Delete(globalmesh);
#endif

  }

  if (!ok) {
    std::stringstream mesg_stream;
    mesg_stream << "Failed to generate mesh on processor " << myprocid;
    Errors::Message mesg(mesg_stream.str());
    amanzi_throw(mesg);
    assert(ok);
  }




  // Do all the processing required for setting up the mesh for Amanzi 
  
  post_create_steps_();

}



//-------------------------------------- 
// Construct a 2D or 3D regular mesh using input from the
// GenerationSpec class 
//--------------------------------------

Mesh_MSTK::Mesh_MSTK(const GenerationSpec& gspec,
		     const Epetra_MpiComm *incomm,
		     const AmanziGeometry::GeometricModelPtr& gm,
                     const VerboseObject *verbobj) :
  Mesh(verbobj), mpicomm(incomm->GetMpiComm()), meshxyz(NULL), 
  target_cell_volumes(NULL), min_cell_volumes(NULL)
{
  int ok;


  // Get info about the domain from the generation specification class

  AmanziGeometry::Point p0(gspec.domain().point0());
  AmanziGeometry::Point p1(gspec.domain().point1());

  int space_dim = p0.dim();
  pre_create_steps_(space_dim, incomm, gm);


  Mesh::set_mesh_type(RECTANGULAR);   // Discretizations can use this info if they want


  if (myprocid == 0) {
    int DebugWait=0;
    while (DebugWait);
  }

  int topo_dim=space_dim;
  set_cell_dimension(topo_dim);

  if (serial_run) {

    // Load serial mesh

    mesh = MESH_New(F1);

    if (topo_dim == 2) {
      ok = generate_regular_mesh(mesh,p0.x(),p0.y(),p1.x(),p1.y(),
				 gspec.xcells(),gspec.ycells());
    }
    else if (topo_dim == 3) {
      ok = generate_regular_mesh(mesh,p0.x(),p0.y(),p0.z(),p1.x(),p1.y(),p1.z(),
				 gspec.xcells(),gspec.ycells(),gspec.zcells());
    }


    myprocid = 0;
  }
  else {
    Mesh_ptr globalmesh;
    int ring = 1; // One layer of ghost cells in parallel meshes
    int with_attr = 1;  // update of attributes in parallel meshes
    int del_inmesh = 1; // delete global mesh
    int method = 1; // Partition with ZOLTAN
    
    if (myprocid == 0) {
      globalmesh = MESH_New(F1);

      if (topo_dim == 2) {
	ok = generate_regular_mesh(globalmesh,p0.x(),p0.y(),p1.x(),p1.y(),
				   gspec.xcells(),gspec.ycells());
      }
      else if (topo_dim == 3) {
	ok = generate_regular_mesh(globalmesh,
				   p0.x(),p0.y(),p0.z(),p1.x(),p1.y(),p1.z(),
				   gspec.xcells(),gspec.ycells(),gspec.zcells());
      }
    }
    else {
      globalmesh = NULL;
      ok = 1;
    }
      
#ifdef MSTK_2_21rc1_OR_NEWER
    ok = ok & MSTK_Mesh_Distribute(globalmesh,&mesh,&topo_dim,ring,with_attr,
                                   method,del_inmesh,mpicomm);
#else
    ok = ok & MSTK_Mesh_Distribute(globalmesh,&mesh,&topo_dim,ring,with_attr,
                                   method,mpicomm);
    if (myprocid == 0)
      MESH_Delete(globalmesh);
#endif
  }

  if (!ok) {
    std::stringstream mesg_stream;
    mesg_stream << "Failed to generate mesh on processor " << myprocid;
    Errors::Message mesg(mesg_stream.str());
    amanzi_throw(mesg);
  }




  // Do all the processing required for setting up the mesh for Amanzi 
  
  post_create_steps_();

}


std::string 
Mesh_MSTK::internal_name_of_set(const AmanziGeometry::RegionPtr r,
                                const Entity_kind entity_kind) const {

  std::string internal_name;
  
  if (r->type() == AmanziGeometry::LABELEDSET) {
    
    AmanziGeometry::LabeledSetRegionPtr lsrgn = 
      dynamic_cast<AmanziGeometry::LabeledSetRegionPtr> (r);
    std::string label = lsrgn->label();

    if (entity_kind == CELL)
      internal_name = "matset_" + label;
    else if (entity_kind == FACE)
      internal_name = "sideset_" + label;
    else if (entity_kind == NODE)
      internal_name = "nodeset_" + label;      
  }
  else {
    if (entity_kind == CELL)
      internal_name = "CELLSET_" + r->name();
    else if (entity_kind == FACE)
      internal_name = "FACESET_" + r->name();
    else if (entity_kind == NODE)
      internal_name = "NODESET_" + r->name();
  }

  return internal_name;
}



//--------------------------------------
// Constructor - Construct a new mesh from a subset of an existing mesh
//--------------------------------------

void Mesh_MSTK::extract_mstk_mesh(const Mesh_MSTK& inmesh,
                                  const std::vector<std::string>& setnames, 
                                  const Entity_kind setkind,
                                  const bool flatten,
                                  const bool extrude)
{
                         
  // Assume three dimensional problem if constructor called without 
  // the space_dimension parameter

  int ok, ival, idx;
  double rval, xyz[3];
  void *pval;

  Mesh_ptr inmesh_mstk = inmesh.mesh;

  if (extrude) {
    Errors::Message mesg("Extrude option not implemented yet");
    amanzi_throw(mesg);
  }


  if (flatten || extrude) {
    if ((setkind == CELL  && inmesh.cell_dimension() == 3) || setkind == NODE) {
      Errors::Message mesg("Flattening or extruding allowed only for sets of FACEs in volume mesh or CELLs in surface meshes");
      amanzi_throw(mesg);
    }
  }


  if ((inmesh.cell_dimension() == 2) && (setkind == FACE) && flatten) {
    Errors::Message mesg("Requested mesh constructor produces 1D mesh which is not supported by Amanzi");
    amanzi_throw(mesg);
  }


  // CODE_LABEL_NUMBER 1 - referred to by a comment later 
  //
  // Access all the requested sets to make sure they get
  // created within MSTK (they get created the first time
  // this set is accessed)

  for (int i = 0; i < setnames.size(); ++i)
    inmesh.get_set_size(setnames[i],setkind,OWNED);
    

  // Pre-processing (init, MPI queries etc)

  if (flatten)
    pre_create_steps_(inmesh.space_dimension()-1, inmesh.get_comm(), 
                      inmesh.geometric_model());
  else
    pre_create_steps_(inmesh.space_dimension(), inmesh.get_comm(), 
                      inmesh.geometric_model());

  if (myprocid == 0) {
    int DebugWait=0;
    while (DebugWait);
  }


  // Set parent mesh
  
  parent_mesh = &inmesh;


  // Create new mesh in MSTK

  mesh = MESH_New(MESH_RepType(inmesh_mstk));

  MType entdim;
  switch (setkind) {
  case CELL:
    if (inmesh.space_dimension() == 3)
      entdim = inmesh.cell_dimension() == 3 ? MREGION : MFACE;
    else if (inmesh.space_dimension() == 2)
      entdim = MFACE;
    set_cell_dimension(inmesh.cell_dimension()); 
    break;
  case FACE:
    if (inmesh.space_dimension() == 3)
      entdim = inmesh.cell_dimension() == 3 ? MFACE : MEDGE;
    else if (inmesh.space_dimension() == 2)
      entdim = MEDGE; // We are not supporting 1D meshes 
    if (extrude)
      set_cell_dimension(inmesh.cell_dimension());
    else
      set_cell_dimension(inmesh.cell_dimension()-1); 
    break;
  case NODE:
    entdim = MVERTEX;
    set_cell_dimension(0);
    break;
  default:
    Errors::Message mesg("Unrecognized Entity_kind");
    amanzi_throw(mesg);
  }



  int mkid = MSTK_GetMarker();
  MSet_ptr src_ents = MSet_New(inmesh_mstk,"src_ents",entdim);
  for (int i = 0; i < setnames.size(); ++i) {
    MSet_ptr mset;
    
    AmanziGeometry::GeometricModelPtr gm = inmesh.geometric_model();
    AmanziGeometry::RegionPtr rgn = gm->FindRegion(setnames[i]);

    std::string internal_name = internal_name_of_set(rgn,setkind);

    mset = MESH_MSetByName(inmesh_mstk,internal_name.c_str());

    idx = 0;
    MEntity_ptr ment;
    while ((ment = (MEntity_ptr) MSet_Next_Entry(mset,&idx))) {
      if (!MEnt_IsMarked(ment,mkid) && MEnt_PType(ment) != PGHOST) {
        MSet_Add(src_ents,ment);
        MEnt_Mark(ment,mkid);
      }
    }
  }

  MSet_Unmark(src_ents,mkid);
  MSTK_FreeMarker(mkid);  


  // Have to do some additional work for extruding an extracted mesh
  // Extrusion applicable only in the case of entdim = MFACE/MEDGE

  MAttrib_ptr copyatt = MAttrib_New(inmesh_mstk,"copyatt",POINTER,MALLTYPE);
  vparentatt = MAttrib_New(mesh,"vparentatt",POINTER,MVERTEX);
  eparentatt = MAttrib_New(mesh,"eparentatt",POINTER,MEDGE);
  fparentatt = MAttrib_New(mesh,"fparentatt",POINTER,MFACE);
  rparentatt = MAttrib_New(mesh,"rparentatt",POINTER,MREGION);

  switch (entdim) {
  case MREGION:
      
    idx = 0; 
    MRegion_ptr mr;
    while ((mr = (MRegion_ptr) MSet_Next_Entry(src_ents,&idx))) {

      List_ptr rfaces = MR_Faces(mr);                                  
      int nrf = List_Num_Entries(rfaces);
      MFace_ptr rfaces_new[MAXPF3];
      int rfdirs_new[MAXPF3];
      for (int i = 0; i < nrf; ++i) {
        MFace_ptr mf = List_Entry(rfaces,i);

        MEnt_Get_AttVal(mf,copyatt,&ival,&rval,&pval);
        if (pval) {
          rfaces_new[i] = pval;
          rfdirs_new[i] = MR_FaceDir_i(mr,i);
        }
        else {

          List_ptr fverts = MF_Vertices(mf,1,0);
          int nfv = List_Num_Entries(fverts);
          MVertex_ptr fverts_new[MAXPV2];
          for (int j = 0; j < nfv; ++j) {
            MVertex_ptr mv = List_Entry(fverts,j);
            MEnt_Get_AttVal(mv,copyatt,&ival,&rval,&pval);
            if (pval)
              fverts_new[j] = pval;
            else {
              fverts_new[j] = MV_New(mesh);
              MV_Coords(mv,xyz);
              MV_Set_Coords(fverts_new[j],xyz);
              MV_Set_GEntDim(fverts_new[j],MV_GEntDim(mv));
              MV_Set_GEntID(fverts_new[j],MV_GEntID(mv));
              MEnt_Set_AttVal(mv,copyatt,ival,rval,fverts_new[j]);
              MEnt_Set_AttVal(fverts_new[j],vparentatt,0,0.0,mv);
            }
          }
          List_Delete(fverts);
            
          rfaces_new[i] = MF_New(mesh);
          MF_Set_Vertices(rfaces_new[i],nfv,fverts_new);
          MF_Set_GEntDim(rfaces_new[i],MF_GEntDim(mf));
          MF_Set_GEntID(rfaces_new[i],MF_GEntID(mf));
          rfdirs_new[i] = MR_FaceDir_i(mr,i);

          MEnt_Set_AttVal(mf,copyatt,ival,rval,rfaces_new[i]);
          MEnt_Set_AttVal(rfaces_new[i],fparentatt,0,0.0,mf);
        }
      }
      List_Delete(rfaces);

      MRegion_ptr mr_new = MR_New(mesh);
      MR_Set_Faces(mr_new,nrf,rfaces_new,rfdirs_new);
      MR_Set_GEntID(mr_new,MR_GEntID(mr));

      MEnt_Set_AttVal(mr,copyatt,ival,rval,mr_new);
      MEnt_Set_AttVal(mr_new,rparentatt,0,0.0,mr);
    }

    break;

  case MFACE:

    idx = 0; 
    MFace_ptr mf;
    while ((mf = (MFace_ptr) MSet_Next_Entry(src_ents,&idx))) {

      List_ptr fedges = MF_Edges(mf,1,0);
      int nfe = List_Num_Entries(fedges);
      int fedirs[MAXPV2];
      MEdge_ptr fedges_new[MAXPV2];
      for (int j = 0; j < nfe; ++j) {
        MEdge_ptr me = List_Entry(fedges,j);
        MEnt_Get_AttVal(me,copyatt,&ival,&rval,&pval);
        if (pval)
          fedges_new[j] = pval;
        else {
          fedges_new[j] = ME_New(mesh);

          for (int k = 0; k < 2; ++k) {
            MVertex_ptr mv = ME_Vertex(me,k);
            MVertex_ptr mv_new;
            MEnt_Get_AttVal(mv,copyatt,&ival,&rval,&pval);
            if (pval)
              mv_new = pval;
            else {              
              MV_Coords(mv,xyz);
              if (flatten) xyz[2] = 0.0;
              mv_new = MV_New(mesh);
              MV_Set_Coords(mv_new,xyz);
              MV_Set_GEntDim(mv_new,MV_GEntDim(mv));
              MV_Set_GEntID(mv_new,MV_GEntID(mv));
              MEnt_Set_AttVal(mv,copyatt,ival,rval,mv_new);
              MEnt_Set_AttVal(mv_new,vparentatt,0,0.0,mv);
            }

            ME_Set_Vertex(fedges_new[j],k,mv_new);
            ME_Set_GEntDim(fedges_new[j],ME_GEntDim(me));
            ME_Set_GEntID(fedges_new[j],ME_GEntID(me));
            MEnt_Set_AttVal(me,copyatt,ival,rval,fedges_new[j]);
            MEnt_Set_AttVal(fedges_new[j],eparentatt,0,0.0,me);
          }
        }
        fedirs[j] = MF_EdgeDir_i(mf,j);
      }
      List_Delete(fedges);
            
      MFace_ptr mf_new = MF_New(mesh);
      MF_Set_Edges(mf_new,nfe,fedges_new,fedirs);
      MF_Set_GEntDim(mf_new,MF_GEntDim(mf));
      MF_Set_GEntID(mf_new,MF_GEntID(mf));

      MEnt_Set_AttVal(mf,copyatt,ival,rval,mf_new);
      MEnt_Set_AttVal(mf_new,fparentatt,0,0.0,mf);
    }

    break;

  case MEDGE:

    idx = 0;
    MEdge_ptr me;
    while ((me = (MEdge_ptr) MSet_Next_Entry(src_ents,&idx))) {

      MEdge_ptr me_new = ME_New(mesh);

      for (int j = 0; j < 2; ++j)  {
        MVertex_ptr mv = ME_Vertex(me,j);

        MVertex_ptr mv_new;

        MEnt_Get_AttVal(mv,copyatt,&ival,&rval,&pval);
        if (pval)
          mv_new = pval;
        else {
          MV_Coords(mv,xyz);
          if (flatten) {
            xyz[1] = 0.0;
            xyz[2] = 0.0;
          }
          mv_new = MV_New(mesh);
          MV_Set_Coords(mv_new,xyz);
          MV_Set_GEntDim(mv_new,MV_GEntDim(mv));
          MV_Set_GEntID(mv_new,MV_GEntID(mv));

          MEnt_Set_AttVal(mv,copyatt,ival,rval,mv_new);
          MEnt_Set_AttVal(mv_new,vparentatt,0,0.0,mv);
        }

        ME_Set_Vertex(me_new,j,mv_new);
      }

      MEnt_Set_AttVal(me,copyatt,ival,rval,me_new);
      MEnt_Set_AttVal(me,eparentatt,0,0.0,me);
    }

    break;

  case MVERTEX:

    idx = 0;
    MVertex_ptr mv;
    while ((mv = (MVertex_ptr) MSet_Next_Entry(src_ents,&idx))) {

      MVertex_ptr mv_new = MV_New(mesh);
      MV_Set_Coords(mv_new,xyz);
      if (flatten) xyz[2] = 0.0;
      MV_Set_GEntDim(mv_new,MV_GEntDim(mv));
      MV_Set_GEntID(mv_new,MV_GEntID(mv));
        
      MEnt_Set_AttVal(mv,copyatt,ival,rval,mv_new);
      MEnt_Set_AttVal(mv_new,vparentatt,0,0.0,mv);
    }

    break;

  default:
    Errors::Message mesg("Unknown entity type");
    amanzi_throw(mesg);
  }


  if (!serial_run) {
    // Have to assign global IDs and build ghost entities 

    int num_ghost_layers = 1; 
    int input_type = 0; /* No parallel info is given */
    int status = MSTK_Weave_DistributedMeshes(mesh, cell_dimension(),
                                              num_ghost_layers, input_type, mpicomm);

    // Now we have to build parent information for global entities

    MAttrib_ptr vparentgid_att = MAttrib_New(mesh,"vparent_gid",INT,MVERTEX);
    MAttrib_ptr eparentgid_att = MAttrib_New(mesh,"eparent_gid",INT,MEDGE);
    MAttrib_ptr fparentgid_att = MAttrib_New(mesh,"fparent_gid",INT,MFACE);
    MAttrib_ptr rparentgid_att = MAttrib_New(mesh,"rparent_gid",INT,MREGION);

    // Attach parent global ID info to entities used by other processors

    idx = 0;
    MVertex_ptr mv;
    while ((mv = (MVertex_ptr) MESH_Next_Vertex(mesh,&idx)))
      if (MV_PType(mv) == POVERLAP) {
        MEnt_Get_AttVal(mv,vparentatt,&ival,&rval,&pval);
        MEnt_Set_AttVal(mv,vparentgid_att,MV_GlobalID((MVertex_ptr)pval),0.0,
                        NULL);
      }
    idx = 0;
    MEdge_ptr me;
    while ((me = (MEdge_ptr) MESH_Next_Edge(mesh,&idx)))
      if (ME_PType(me) == POVERLAP) {
        MEnt_Get_AttVal(me,eparentatt,&ival,&rval,&pval);
        MEnt_Set_AttVal(me,eparentgid_att,ME_GlobalID((MEdge_ptr)pval),0.0,
                        NULL);
      }
    idx = 0;
    MFace_ptr mf;
    while ((mf = (MFace_ptr) MESH_Next_Face(mesh,&idx)))
      if (MF_PType(mf) == POVERLAP) {
        MEnt_Get_AttVal(mf,fparentatt,&ival,&rval,&pval);
        MEnt_Set_AttVal(mf,fparentgid_att,MF_GlobalID((MFace_ptr)pval),0.0,
                        NULL);
      }
    idx = 0;
    MRegion_ptr mr;
    while ((mr = (MRegion_ptr) MESH_Next_Region(mesh,&idx)))
      if (MR_PType(mr) == POVERLAP) {
        MEnt_Get_AttVal(mr,rparentatt,&ival,&rval,&pval);
        MEnt_Set_AttVal(mr,rparentgid_att,MR_GlobalID((MRegion_ptr)pval),0.0,
                        NULL);
      }
      
    // Update attributes on ghost entities - this will ensure that
    // ghost entities have their parent global ID information

    status &= MSTK_UpdateAttr(mesh,mpicomm);

    // Now reverse engineer the parents of ghost entities from the global IDs

    idx = 0;
    while ((mv = (MVertex_ptr) MESH_Next_GhostVertex(mesh,&idx))) {
      MEnt_Get_AttVal(mv,vparentgid_att,&ival,&rval,&pval);
      MVertex_ptr mv_parent = MESH_VertexFromGlobalID(inmesh_mstk,ival);
      if (!mv_parent) {
        Errors::Message 
          mesg("Cannot find ghost vertex with given global ID");
        amanzi_throw(mesg);
      }
      MEnt_Set_AttVal(mv,vparentatt,0,0.0,mv_parent);
    }
    idx = 0;
    while ((me = (MEdge_ptr) MESH_Next_GhostEdge(mesh,&idx))) {
      MEnt_Get_AttVal(me,eparentgid_att,&ival,&rval,&pval);
      MEdge_ptr me_parent = MESH_EdgeFromGlobalID(inmesh_mstk,ival);
      if (!me_parent) {
        Errors::Message 
          mesg("Cannot find ghost edge with given global ID");
        amanzi_throw(mesg);
      }
      MEnt_Set_AttVal(me,eparentatt,0,0.0,me_parent);
    }
    idx = 0;
    while ((mf = (MFace_ptr) MESH_Next_GhostFace(mesh,&idx))) {
      MEnt_Get_AttVal(mf,fparentgid_att,&ival,&rval,&pval);
      MFace_ptr mf_parent = MESH_FaceFromGlobalID(inmesh_mstk,ival);
      if (!mf_parent) {
        Errors::Message 
          mesg("Cannot find ghost face with given global ID");
        amanzi_throw(mesg);
      }
      MEnt_Set_AttVal(mf,fparentatt,0,0.0,mf_parent);
    }
    idx = 0;
    while ((mr = (MRegion_ptr) MESH_Next_GhostRegion(mesh,&idx))) {
      MEnt_Get_AttVal(mr,rparentgid_att,&ival,&rval,&pval);
      MRegion_ptr mr_parent = MESH_RegionFromGlobalID(inmesh_mstk,ival);
      if (!mr_parent) {
        Errors::Message 
          mesg("Cannot find ghost region with given global ID");
        amanzi_throw(mesg);
      }
      MEnt_Set_AttVal(mr,rparentatt,0,0.0,mr_parent);
    }

    MAttrib_Delete(vparentgid_att);
    MAttrib_Delete(eparentgid_att);
    MAttrib_Delete(fparentgid_att);
    MAttrib_Delete(rparentgid_att);
  }



  // For this constructor, we have to do an extra step to build new
  // labeled sets based on the base mesh

  inherit_labeled_sets(copyatt);


  // Do all the processing required for setting up the mesh for Amanzi 
  
  post_create_steps_();


  // Clean up

  switch (entdim) {
  case MREGION:
    MRegion_ptr mr;
    idx = 0; 
    while ((mr = (MRegion_ptr) MSet_Next_Entry(src_ents,&idx))) {

      List_ptr rfaces = MR_Faces(mr);                                  
      int nrf = List_Num_Entries(rfaces);

      for (int i = 0; i < nrf; ++i) {
        MFace_ptr mf = List_Entry(rfaces,i);
        MEnt_Rem_AttVal(mf,copyatt);

        List_ptr fverts = MF_Vertices(mf,1,0);
        int nfv = List_Num_Entries(fverts);

        for (int j = 0; j < nfv; ++j) {
          MVertex_ptr mv = List_Entry(fverts,j);
          MEnt_Rem_AttVal(mv,copyatt);
        }
        List_Delete(fverts);
            
        MEnt_Rem_AttVal(mf,copyatt);
      }
      List_Delete(rfaces);

      MEnt_Rem_AttVal(mr,copyatt);
    }

    break;

  case MFACE:
    MFace_ptr mf;
    idx = 0; 
    while ((mf = (MFace_ptr) MSet_Next_Entry(src_ents,&idx))) {

      List_ptr fedges = MF_Edges(mf,1,0);
      int nfe = List_Num_Entries(fedges);
      for (int j = 0; j < nfe; ++j) {
        MEdge_ptr me = List_Entry(fedges,j);
        MEnt_Rem_AttVal(me,copyatt);
        MVertex_ptr mv = ME_Vertex(me,MF_EdgeDir_i(mf,j));
        MEnt_Rem_AttVal(mv,copyatt);
      }
      List_Delete(fedges);
            
      MEnt_Rem_AttVal(mf,copyatt);
    }

    break;

  case MEDGE:
    MEdge_ptr me;
    idx = 0;
    while ((me = (MEdge_ptr) MSet_Next_Entry(src_ents,&idx))) {
      for (int j = 0; j < 2; ++j)  {
        MVertex_ptr mv = ME_Vertex(me,j);
        MEnt_Rem_AttVal(mv,copyatt);
      }
      MEnt_Rem_AttVal(me,copyatt);
    }

    break;

  case MVERTEX:
    MVertex_ptr mv;
    idx = 0;
    while ((mv = (MVertex_ptr) MSet_Next_Entry(src_ents,&idx)))
      MEnt_Rem_AttVal(mv,copyatt);

    break;

  default:
    Errors::Message mesg("Unknown entity type");
    amanzi_throw(mesg);
  }
  
  MAttrib_Delete(copyatt);
  MSet_Delete(src_ents);

}

Mesh_MSTK::Mesh_MSTK (const Mesh *inmesh, 
                      const std::vector<std::string>& setnames, 
                      const Entity_kind setkind,
                      const bool flatten,
                      const bool extrude) :
  mpicomm(inmesh->get_comm()->GetMpiComm()),
  Mesh(inmesh->verbosity_obj())
{  

  extract_mstk_mesh(*((Mesh_MSTK *) inmesh),setnames,setkind,flatten,extrude);

}

Mesh_MSTK::Mesh_MSTK (const Mesh_MSTK& inmesh, 
                      const std::vector<std::string>& setnames, 
                      const Entity_kind setkind,
                      const bool flatten,
                      const bool extrude) :
  mpicomm(inmesh.get_comm()->GetMpiComm()),
  Mesh(inmesh.verbosity_obj())
{  

  extract_mstk_mesh(inmesh,setnames,setkind,flatten,extrude);

}



// Destructor with cleanup

Mesh_MSTK::~Mesh_MSTK() {
  delete cell_map_wo_ghosts_;
  delete cell_map_w_ghosts_;
  delete face_map_wo_ghosts_;
  delete face_map_w_ghosts_;
  delete node_map_wo_ghosts_;
  delete node_map_w_ghosts_;
  delete extface_map_wo_ghosts_;
  delete owned_to_extface_importer_;
  delete [] faceflip;

  if (OwnedVerts) MSet_Delete(OwnedVerts);
  if (NotOwnedVerts) MSet_Delete(NotOwnedVerts);
  if (OwnedFaces) MSet_Delete(OwnedFaces);
  if (NotOwnedFaces) MSet_Delete(NotOwnedFaces);
  if (OwnedCells) MSet_Delete(OwnedCells);
  if (GhostCells) MSet_Delete(GhostCells);

  if (entities_deleted) {
    if (deleted_vertices) List_Delete(deleted_vertices);
    if (deleted_edges) List_Delete(deleted_edges);
    if (deleted_faces) List_Delete(deleted_faces);
    if (deleted_regions) List_Delete(deleted_regions);
  }

  MAttrib_Delete(celltype_att);
  if (vparentatt) MAttrib_Delete(vparentatt);
  if (eparentatt) MAttrib_Delete(eparentatt);
  if (fparentatt) MAttrib_Delete(fparentatt);
  if (rparentatt) MAttrib_Delete(rparentatt);

  MESH_Delete(mesh);
}


// Number of OWNED, GHOST or USED entities of different types

    
// Number of entities of any kind (cell, face, node) and in a
// particular category (OWNED, GHOST, USED)
    
unsigned int Mesh_MSTK::num_entities (const Entity_kind kind, 
				      const Parallel_type ptype) const
{
  const int rank = (int) kind;
  const int index = ((int) ptype) - 1;

  switch (kind) {
  case NODE:

    switch (ptype) {
    case OWNED:
      return MSet_Num_Entries(OwnedVerts);
      break;
    case GHOST:
      return !serial_run ? MSet_Num_Entries(NotOwnedVerts) : 0;
      break;
    case USED:
      return MESH_Num_Vertices(mesh);
      break;
    default:
      return 0;
    }
    break;


  case FACE:
    switch (ptype) {
    case OWNED:
      return MSet_Num_Entries(OwnedFaces);
      break;
    case GHOST:
      return !serial_run ? MSet_Num_Entries(NotOwnedFaces) : 0;
      break;
    case USED:
      return (cell_dimension() == 2 ? MESH_Num_Edges(mesh) : MESH_Num_Faces(mesh));
      break;
    default:
      return 0;
    }
    break;


  case CELL:

    switch (ptype) {
    case OWNED:
      return MSet_Num_Entries(OwnedCells);
      break;
    case GHOST:
      return !serial_run ? MSet_Num_Entries(GhostCells) : 0;
      break;
    case USED:
      return ((cell_dimension() == 2) ? MESH_Num_Faces(mesh) : MESH_Num_Regions(mesh));
      break;
    default:
      return 0;
    }

    break;
  default:
    std::cerr << "Count requested for unknown entity type" << std::endl;
  }
  return 0;
}


// Get cell type

Cell_type Mesh_MSTK::cell_get_type(const Entity_ID cellid) const {
  MEntity_ptr cell;
  int ival;
  Cell_type celltype;
  
  cell = cell_id_to_handle[cellid];
  
  MEnt_Get_AttVal(cell,celltype_att,&ival,NULL,NULL);
  celltype = (Cell_type) ival;

  return celltype;

} // Mesh_MSTK::cell_get_type


  // Get faces of a cell and directions in which the cell uses the face 

  // The Amanzi coding guidelines regarding function arguments is purposely
  // violated here to allow for a default input argument

  // On a distributed mesh, this will return all the faces of the
  // cell, OWNED or GHOST. If ordered = true, the faces will be
  // returned in a standard order according to Exodus II convention
  // for standard cells; in all other situations (ordered = false or
  // non-standard cells), the list of faces will be in arbitrary order

  // In 3D, direction is 1 if face normal points out of cell
  // and -1 if face normal points into cell
  // In 2D, direction is 1 if face/edge is defined in the same
  // direction as the cell polygon, and -1 otherwise

void Mesh_MSTK::cell_get_faces_and_dirs_ordered (const Entity_ID cellid,
                                                 Entity_ID_List *faceids,
                                                 std::vector<int> *face_dirs) const 
{

  MEntity_ptr cell;

  if (cell_dimension() == 3) {

    int celltype = cell_get_type(cellid);
      
    if (celltype >= TET && celltype <= HEX) {
      int lid, nf;
      
      cell = cell_id_to_handle[cellid];

      List_ptr rfaces = MR_Faces((MRegion_ptr)cell);   
      nf = List_Num_Entries(rfaces);
      
      faceids->resize(nf);
      if (face_dirs)
        face_dirs->resize(nf);
      
      /* base face */
      
      MFace_ptr face0;
      int fdir0;
      
      if (celltype == TET || celltype == HEX) {
        face0 = List_Entry(rfaces,0);
        fdir0 = MR_FaceDir_i((MRegion_ptr)cell,0);
      }
      else if (celltype == PRISM) { /* Find the first triangular face */
        for (int i = 0; i < 5; ++i) {
          MFace_ptr face = List_Entry(rfaces,i);
          if (MF_Num_Edges(face) == 3) {
            face0 = face;
            fdir0 = MR_FaceDir_i((MRegion_ptr)cell,i);
            break;
          }
        }
      }
      else if (celltype == PYRAMID) { /* Find the quad face */
        for (int i = 0; i < 5; ++i) {
          MFace_ptr face = List_Entry(rfaces,i);
          if (MF_Num_Edges(face) == 4) {
            face0 = face;
            fdir0 = MR_FaceDir_i((MRegion_ptr)cell,i);
            break;
          }
        }
      }
      
      /* Markers for faces to avoid searching */
      
      int mkid = MSTK_GetMarker();
      MEnt_Mark(face0,mkid);
      
      /* Add all lateral faces first (faces adjacent to the base face) */
      
      List_ptr fedges0 = MF_Edges(face0,!fdir0,0);
      int idx = 0;
      MEdge_ptr fe;
      nf = 0;
      while ((fe = List_Next_Entry(fedges0,&idx))) {
        
        /* Is there an unprocessed face in this region that is
           adjacent to this edge */
        
        int idx2 = 0;
        MFace_ptr fadj; 
        int i = 0;
        while ((fadj = List_Next_Entry(rfaces,&idx2))) {
          if (fadj != face0 && !MEnt_IsMarked(fadj,mkid)) {
            
            if (MF_UsesEntity(fadj,fe,MEDGE)) {
              
              lid = MEnt_ID(fadj);              
              (*faceids)[nf] = lid-1;
              
              if (face_dirs) {
                int fdir = (MR_FaceDir_i((MRegion_ptr)cell,i) == 1) ? 1 : -1;
                if (faceflip[lid-1]) fdir *= -1;
                (*face_dirs)[nf] = fdir;
              }
              
              MEnt_Mark(fadj,mkid);
              nf++;
            }
          }
          ++i;
        }
      }
      List_Delete(fedges0);
      
      /* Add the base face */
      
      lid = MEnt_ID(face0);
      (*faceids)[nf] = lid-1;
      
      if (face_dirs) {
        fdir0 = fdir0 ? 1 : -1;
        if (faceflip[lid-1]) fdir0 *= -1;
        (*face_dirs)[nf] = fdir0;
      }
      nf++;
      
      /* If there is a last remaining face, it is the top face */
      
      MFace_ptr fopp;
      idx = 0;
      int i = 0;
      while ((fopp = List_Next_Entry(rfaces,&idx))) {
        if (fopp != face0 && !MEnt_IsMarked(fopp,mkid)) {
          
          lid = MEnt_ID(fopp);
          (*faceids)[nf] = lid-1;
          
          if (face_dirs) {
            int fdir = (MR_FaceDir_i((MRegion_ptr)cell,i) == 1) ? 1 : -1;
            if (faceflip[lid-1]) fdir *= -1;
            (*face_dirs)[nf] = fdir;
          }
          nf++;
          break;
          
        }
        ++i;
      }
      
      List_Unmark(rfaces,mkid);
      MSTK_FreeMarker(mkid);
      
      List_Delete(rfaces);
    }
    else 
      cell_get_faces_and_dirs_unordered(cellid,faceids,face_dirs);
  }
  else
    cell_get_faces_and_dirs_unordered(cellid,faceids,face_dirs);

}


void Mesh_MSTK::cell_get_faces_and_dirs_unordered (const Entity_ID cellid,
                                                   Entity_ID_List *faceids,
                                                   std::vector<int> *face_dirs)
  const

{

  MEntity_ptr cell;

  ASSERT(faceids != NULL);

  cell = cell_id_to_handle[cellid];

  if (cell_dimension() == 3) {
    int nrf;
    List_ptr rfaces;

    rfaces = MR_Faces((MRegion_ptr)cell);
    nrf = List_Num_Entries(rfaces);
    faceids->resize(nrf);

    Entity_ID_List::iterator itf = faceids->begin();
    for (int i = 0; i < nrf; ++i) {
      MFace_ptr face = List_Entry(rfaces,i);
      int lid = MEnt_ID(face);
      *itf = lid-1;  // assign to next spot by dereferencing iterator
      ++itf;
    }

    List_Delete(rfaces);
    
    /* Reserved for next major MSTK release 
    int rfaceids[MAXPF3];

    MR_FaceIDs((MRegion_ptr)cell,&nrf,rfaceids);
    faceids->resize(nrf);
    Entity_ID_List::iterator it = faceids->begin();
    for (int i = 0; i < nrf; ++i) { 
      *it = rfaceids[i]-1;
      ++it;
    }
    */
    
    if (face_dirs) {
      face_dirs->resize(nrf);

      std::vector<int>::iterator itd = face_dirs->begin();
      for (int i = 0; i < nrf; ++i) {
        int lid = (*faceids)[i];
        int fdir = 2*MR_FaceDir_i((MRegion_ptr)cell,i) - 1;
        fdir = faceflip[lid] ? -fdir : fdir;
        *itd = fdir;  // assign to next spot by dereferencing iterator
	++itd;
      }
    }
    
  }
  else {  // cell_dimension() = 2; surface or 2D mesh
    int nfe;

    List_ptr fedges;
    fedges = MF_Edges((MFace_ptr)cell,1,0);
    nfe = List_Num_Entries(fedges);

    faceids->resize(nfe);

    Entity_ID_List::iterator itf = faceids->begin();
    for (int i = 0; i < nfe; ++i) {
      MEdge_ptr edge = List_Entry(fedges,i);
      int lid = MEnt_ID(edge);
      *itf = lid-1;  // assign to next spot by dereferencing iterator
      ++itf;
    }

    List_Delete(fedges);

    /* Reserved for next major MSTK release 

    int fedgeids[MAXPV2];
    MF_EdgeIDs((MFace_ptr)cell,1,0,&nfe,fedgeids);
    
    faceids->resize(nfe);
    Entity_ID_List::iterator itf = faceids->begin();
    for (int i = 0; i < nfe; ++i) {
      *itf = fedgeids[i]-1;
      ++itf;
    }
    */

    if (face_dirs) {
      face_dirs->resize(nfe);
      std::vector<int>::iterator itd = face_dirs->begin();
      for (int i = 0; i < nfe; ++i) { 
        int lid = (*faceids)[i];
        int fdir = 2*MF_EdgeDir_i((MFace_ptr)cell,i) - 1;
        fdir = faceflip[lid] ? -fdir : fdir;
        *itd = fdir;  // assign to next spot by dereferencing iterator
	itd++;
      }
    }

  }

}



void Mesh_MSTK::cell_get_faces_and_dirs_internal (const Entity_ID cellid,
                                                  Entity_ID_List *faceids,
                                                  std::vector<int> *face_dirs,
                                                  const bool ordered) const 
{
  if (ordered)
    cell_get_faces_and_dirs_ordered(cellid, faceids, face_dirs);
  else
    cell_get_faces_and_dirs_unordered(cellid, faceids, face_dirs);
}


 
// Get nodes of cell 
// On a distributed mesh, all nodes (OWNED or GHOST) of the cell 
// are returned
// Nodes are returned in a standard order (Exodus II convention)
// STANDARD CONVENTION WORKS ONLY FOR STANDARD CELL TYPES in 3D
// For a general polyhedron this will return the nodes in
// arbitrary order
// In 2D, the nodes of the polygon will be returned in ccw order 
// consistent with the face normal

void Mesh_MSTK::cell_get_nodes (const Entity_ID cellid, 
				std::vector<Entity_ID> *nodeids) const
{
  MEntity_ptr cell;
  int nn, lid;

  ASSERT(nodeids != NULL);

  cell = cell_id_to_handle[cellid];
      
  /* Reserved for next major MSTK release
  int vertids[MAXPV3];

  if (cell_dimension() == 3)            // Volume mesh
    MR_VertexIDs(cell,&nn,vertids);
  else                                  // Surface mesh
    MF_VertexIDs(cell,1,0,&nn,vertids);

  nodeids->resize(nn);
  Entity_ID_List::iterator it = nodeids->begin();
  for (int i = 0; i < nn; ++i) {
    *it = vertids[i]-1;
    ++it;
  }
  */
    
  if (cell_dimension() == 3) {                    // Volume mesh
    List_ptr rverts = MR_Vertices(cell);
 
    nn = List_Num_Entries(rverts);
    nodeids->resize(nn);
    Entity_ID_List::iterator it = nodeids->begin();
    for (int i = 0; i < nn; ++i) {
      lid = MEnt_ID(List_Entry(rverts,i));
      *it = lid-1;  // assign to next spot by dereferencing iterator
      ++it;
    }
    
    List_Delete(rverts);
  }
  else {                                 // Surface mesh
    List_ptr fverts = MF_Vertices(cell,1,0);

    nn = List_Num_Entries(fverts);
    nodeids->resize(nn);
    Entity_ID_List::iterator it = nodeids->begin();
    for (int i = 0; i < nn; ++i) {
      lid = MEnt_ID(List_Entry(fverts,i));
      *it = lid-1;  // assign to next spot by dereferencing iterator
      it++;
    }
    
    List_Delete(fverts);
  }

}  // Mesh_MSTK::cell_get_nodes




    
// Get nodes of face 
// On a distributed mesh, all nodes (OWNED or GHOST) of the face 
// are returned
// In 3D, the nodes of the face are returned in ccw order consistent
// with the face normal
// In 2D, nfnodes is 2

void Mesh_MSTK::face_get_nodes (const Entity_ID faceid, 
				std::vector<Entity_ID> *nodeids) const
{
  MEntity_ptr genface;
  int nn, lid;

  ASSERT(nodeids != NULL);

  genface = face_id_to_handle[faceid];

  if (cell_dimension() == 3) {  // Volume mesh
    int dir = !faceflip[faceid];

    /* Reserved for next major MSTK release
    int vertids[MAXPV2];

    MF_VertexIDs((MFace_ptr) genface,dir,0,&nn,vertids);

    nodeids->resize(nn);
    for (int i = 0; i < nn; ++i) 
      (*nodeids)[i] = vertids[i]-1;
    */

    List_ptr fverts = MF_Vertices(genface,dir,0);
    assert(fverts != NULL);

    nn = List_Num_Entries(fverts);
    nodeids->resize(nn);
    Entity_ID_List::iterator it = nodeids->begin();

    for (int i = 0; i < nn; ++i) {
      lid = MEnt_ID(List_Entry(fverts,i));
      *it = lid-1;  // assign to next spot by dereferencing iterator
      ++it;
    }

    List_Delete(fverts);
    
  }
  else {                // Surface mesh or 2D mesh
    nodeids->resize(2);

    if (faceflip[faceid]) {
      (*nodeids)[0] = MEnt_ID(ME_Vertex(genface,1))-1;
      (*nodeids)[1] = MEnt_ID(ME_Vertex(genface,0))-1;
    }
    else {
      (*nodeids)[0] = MEnt_ID(ME_Vertex(genface,0))-1;
      (*nodeids)[1] = MEnt_ID(ME_Vertex(genface,1))-1;
    }
  }

} // Mesh_MSTK::face_get_nodes
  




// Cells of type 'ptype' connected to a node. This routine uses
// push_back on or near the partition boundary since we cannot tell at
// the outset how many entries will be put into the list
    
void Mesh_MSTK::node_get_cells (const Entity_ID nodeid, 
				const Parallel_type ptype,
				std::vector<Entity_ID> *cellids) const
{
  int idx, lid, nc;
  List_ptr cell_list;
  MEntity_ptr ment;

  assert (cellids != NULL);

  MVertex_ptr mv = (MVertex_ptr) vtx_id_to_handle[nodeid];
  
  /* Reserved for next major release of MSTK
  if (MV_PType(mv) == PINTERIOR && ptype != GHOST) { 

    if (cell_dimension() == 3) {
      int nvr, regionids[200];
      MV_RegionIDs(mv,&nvr,regionids);
      ASSERT(nvr < 200);
      cellids->resize(nvr);
      Entity_ID_List::iterator it = cellids->begin();
      for (int i = 0; i < nvr; ++i) {
        *it = regionids[i]-1;  // assign to next spot by dereferencing iterator
	++it;
      }
    }
    else {      
      int nvf, faceids[200];      
      MV_FaceIDs(mv,&nvf,faceids);
      ASSERT(nvf < 200);
      cellids->resize(nvf);
      Entity_ID_List::iterator it = cellids->begin();
      for (int i = 0; i < nvf; ++i) {
        *it = faceids[i]-1;  // assign to next spot by dereferencing iterator
	++it;
      }
    }
    
  }
  else {
  */
    // mesh vertex on a processor boundary may be connected to owned
    // and ghost cells. So depending on the requested cell type, we
    // may have to omit some entries

    if (cell_dimension() == 3)
      cell_list = MV_Regions(mv);
    else
      cell_list = MV_Faces(mv);

    nc = List_Num_Entries(cell_list);
    cellids->resize(nc); // resize to maximum size possible
    Entity_ID_List::iterator it = cellids->begin();

    int n = 0;
    idx = 0; 
    while ((ment = List_Next_Entry(cell_list,&idx))) {
      if (MEnt_PType(ment) == PGHOST) {
        if (ptype == GHOST || ptype == USED) {
          lid = MEnt_ID(ment);
          *it = lid-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++n;
        }
      }
      else {
        if (ptype == OWNED || ptype == USED) {
          lid = MEnt_ID(ment);
	  *it = lid-1;  // assign to next spot by dereferencing iterator
          ++it;
	  ++n;
        }
      }
    }
    cellids->resize(n); // resize to the actual number of cells being returned

    List_Delete(cell_list);

    /*
  }
    */

} // Mesh_MSTK::node_get_cells


    
// Faces of type 'ptype' connected to a node. This routine uses
// push_back on or near the partition boundary since we cannot tell at
// the outset how many entries will be put into the list
    
void Mesh_MSTK::node_get_faces (const Entity_ID nodeid, 
				const Parallel_type ptype,
				std::vector<Entity_ID> *faceids) const
{
  int idx, lid, n;
  List_ptr face_list;
  MEntity_ptr ment;

  ASSERT(faceids != NULL);

  MVertex_ptr mv = (MVertex_ptr) vtx_id_to_handle[nodeid];

  /* Reserved for next major release of MSTK 
  if (MV_PType(mv) == PINTERIOR && ptype != GHOST) {
    if (cell_dimension() == 3) {
      int nvf, vfaceids[200];

      MV_FaceIDs(mv,&nvf,vfaceids);
      ASSERT(nvf < 200);

      faceids->resize(nvf);
      Entity_ID_List::iterator it = faceids->begin();
      for (int i = 0; i < nvf; ++i) {
        *it = vfaceids[i]-1;  // assign to next spot by dereferencing iterator
	++it;
      }
    }
    else if (cell_dimension() == 2) {
      int nve, vedgeids[200];

      MV_EdgeIDs(mv,&nve,vedgeids);
      ASSERT(nve < 200);

      faceids->resize(nve);
      Entity_ID_List::iterator it = faceids->begin();
      for (int i = 0; i < nve; ++i) {      
        *it = vedgeids[i]-1;  // assign to next spot by dereferencing iterator
        ++it;
      }
    }
  }
  else {
  */
    
    if (cell_dimension() == 3)
      face_list = MV_Faces(mv);
    else
      face_list = MV_Edges(mv);
    
    int nf = List_Num_Entries(face_list);
    faceids->resize(nf); // resize to the maximum
    Entity_ID_List::iterator it = faceids->begin();
    idx = 0; n = 0;
    while ((ment = List_Next_Entry(face_list,&idx))) {
      if (MEnt_PType(ment) == PGHOST) {
        if (ptype == GHOST || ptype == USED) {
          lid = MEnt_ID(ment);
          *it = lid-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++n;
        }
      }
      else {
        if (ptype == OWNED || ptype == USED) {
          lid = MEnt_ID(ment);
          *it = lid-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++n;
        }
      }      
    }
    faceids->resize(n); // resize to the actual number of faces being returned
    
    List_Delete(face_list);

    /*
  }
    */

} // Mesh_MSTK::node_get_faces



    
// Get faces of ptype of a particular cell that are connected to the
// given node. This routine uses push_back since we cannot tell at the
// outset how many entries will be put into the list
    
void Mesh_MSTK::node_get_cell_faces (const Entity_ID nodeid, 
				     const Entity_ID cellid,
				     const Parallel_type ptype,
				     std::vector<Entity_ID> *faceids) const
{
  int idx, lid, n;
  List_ptr cell_list;
  MEntity_ptr ment;
  MRegion_ptr mr;
  MFace_ptr mf;
  MEdge_ptr me;

  ASSERT(faceids != NULL);

  MVertex_ptr mv = (MVertex_ptr) vtx_id_to_handle[nodeid];

  if (cell_dimension() == 3) {
    mr = (MRegion_ptr) cell_id_to_handle[cellid];
    List_ptr rfaces = MR_Faces(mr);

    faceids->resize(List_Num_Entries(rfaces)); // resize to maximum size
    Entity_ID_List::iterator it = faceids->begin();
    
    idx = 0; n = 0;
    while ((mf = List_Next_Entry(rfaces,&idx))) {
      if (!MF_UsesEntity(mf,mv,MVERTEX)) continue;

      if (MEnt_PType(mf) == PGHOST) {
	if (ptype == GHOST || ptype == USED) {
	  lid = MEnt_ID(mf);
	  *it = lid-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++n;
	}
      }
      else {
	if (ptype == OWNED || ptype == USED) {
	  lid = MEnt_ID(mf);
	  *it = lid-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++n;
	}
      }            
    }
    faceids->resize(n); // resize to the actual number of faces being returned
    List_Delete(rfaces);
  }
  else {
    mf = (MFace_ptr) cell_id_to_handle[cellid];
    List_ptr fedges = MF_Edges(mf,1,0);

    faceids->resize(List_Num_Entries(fedges));
    Entity_ID_List::iterator it = faceids->begin();

    idx = 0; n = 0;
    while ((me = List_Next_Entry(fedges,&idx))) {
      if (!ME_UsesEntity(me,mv,MVERTEX)) continue;

      if (MEnt_PType(me) == PGHOST) {
	if (ptype == GHOST || ptype == USED) {
	  lid = MEnt_ID(me);
	  *it = lid-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++n;
	}
      }
      else {
	if (ptype == OWNED || ptype == USED) {
	  lid = MEnt_ID(me);
	  *it = lid-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++n;
	}
      }            
    }
    faceids->resize(n); // resize to the actual number of faces being returned
    List_Delete(fedges);
  }

} // Mesh_MSTK::node_get_cell_faces


    
// Cells connected to a face
    
void Mesh_MSTK::face_get_cells_internal (const Entity_ID faceid, 
                                         const Parallel_type ptype,
                                         std::vector<Entity_ID> *cellids) const
{
  int lid, n;

  ASSERT(cellids != NULL);
  cellids->resize(2); // maximum number
  Entity_ID_List::iterator it = cellids->begin();
  n = 0;

  if (cell_dimension() == 3) {
    MFace_ptr mf = (MFace_ptr) face_id_to_handle[faceid];
   
    List_ptr fregs = MF_Regions(mf);
    MRegion_ptr mr;
    if (ptype == USED) {      
      int idx = 0;
      while ((mr = List_Next_Entry(fregs,&idx))) {
        *it = MR_ID(mr)-1;  // assign to next spot by dereferencing iterator
	++it;
	++n;
      }
    }
    else {
      int idx = 0;
      while ((mr = List_Next_Entry(fregs,&idx))) {
        if (MEnt_PType(mr) == PGHOST) {
          if (ptype == GHOST) {
            *it = MR_ID(mr)-1;  // assign to next spot by dereferencing iterator
	    ++it;
	    ++n;
          }
        }
        else {
	  *it = MR_ID(mr)-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++n;
	}
      }
    }
    List_Delete(fregs);

  }
  else {
    MEdge_ptr me = (MEdge_ptr) face_id_to_handle[faceid];

    List_ptr efaces = ME_Faces(me);
    MFace_ptr mf;
    if (ptype == USED) {
      int idx = 0;
      while ((mf = List_Next_Entry(efaces,&idx))) {
        *it = MF_ID(mf)-1;  // assign to next spot by dereferencing iterator
	++it;
	++n;
      }
    }
    else {
      int idx = 0;
      while ((mf = List_Next_Entry(efaces,&idx))) {
        if (MEnt_PType(mf) == PGHOST) {
          if (ptype == GHOST) {
            *it = MF_ID(mf)-1;  // assign to next spot by dereferencing iterator
	    ++it;
	    ++n;
          }
        }
        else {
          if (ptype == OWNED) {
            *it = MF_ID(mf)-1;  // assign to next spot by dereferencing iterator
            ++it;
	    ++n;
          }
        }
      }
    }
    List_Delete(efaces);

  }
  cellids->resize(n); // resize to the actual number of cells being returned

} // Mesh_MSTK::face_get_cells
    


// Same level adjacencies
//-----------------------

// Face connected neighboring cells of given cell. This routine uses
// push_back since we cannot tell at the outset how many entries will
// be put into the list

void Mesh_MSTK::cell_get_face_adj_cells(const Entity_ID cellid,
					const Parallel_type ptype,
					std::vector<Entity_ID> *fadj_cellids) const
{
  int lid;

  assert(fadj_cellids != NULL);

  fadj_cellids->clear();

  if (cell_dimension() == 3) {

    MRegion_ptr mr = (MRegion_ptr) cell_id_to_handle[cellid];

    List_ptr rfaces = MR_Faces(mr);
    int idx = 0;
    MFace_ptr mf;
    while ((mf = List_Next_Entry(rfaces,&idx))) {
      List_ptr fregs = MF_Regions(mf);
      int idx2 = 0;
      MRegion_ptr mr2;
      while ((mr2 = List_Next_Entry(fregs,&idx2))) {
	if (mr2 != mr) {
	  if (MEnt_PType(mr2) == PGHOST) {
	    if (ptype == GHOST || ptype == USED) {
	      lid = MEnt_ID(mr2);
	      fadj_cellids->push_back(lid-1);
	    }
	  }
	  else {
	    if (ptype == GHOST || ptype == USED) {
	      lid = MEnt_ID(mr2);
	      fadj_cellids->push_back(lid-1);
	    }
	  }
	}
      }
      List_Delete(fregs);
    }

    List_Delete(rfaces);

  }
  else if (cell_dimension() == 2) {

    MFace_ptr mf = (MFace_ptr) cell_id_to_handle[cellid];

    List_ptr fedges = MF_Edges(mf,1,0);
    int idx = 0;
    MEdge_ptr me;
    while ((me = List_Next_Entry(fedges,&idx))) {
      List_ptr efaces = ME_Faces(me);
      int idx2 = 0;
      MFace_ptr mf2;
      while ((mf2 = List_Next_Entry(efaces,&idx2))) {
	if (mf2 != mf) {
	  if (MEnt_PType(mf2) == PGHOST) {
	    if (ptype == GHOST || ptype == USED) {
	      lid = MEnt_ID(mf2);
	      fadj_cellids->push_back(lid-1);
	    }
	  }
	  else {
	    if (ptype == GHOST || ptype == USED) {
	      lid = MEnt_ID(mf2);
	      fadj_cellids->push_back(lid-1);
	    }
	  }
	}
      }
      List_Delete(efaces);
    }

    List_Delete(fedges);

  }

} // Mesh_MSTK::cell_get_face_adj_cells




// Node connected neighboring cells of given cell.  This routine uses
// push_back since we cannot tell at the outset how many entries will
// be put into the list

void Mesh_MSTK::cell_get_node_adj_cells(const Entity_ID cellid,
					const Parallel_type ptype,
					std::vector<Entity_ID> *nadj_cellids) const
{
  int lid, mkid;
  List_ptr cell_list;

  assert(nadj_cellids != NULL);

  nadj_cellids->clear();

  mkid = MSTK_GetMarker();

  cell_list = List_New(0);
  if (cell_dimension() == 3) {

    MRegion_ptr mr = (MRegion_ptr) cell_id_to_handle[cellid];

    List_ptr rvertices = MR_Vertices(mr);
    int idx = 0;
    MVertex_ptr mv;
    while ((mv = List_Next_Entry(rvertices,&idx))) {
      List_ptr vregs = MV_Regions(mv);
      int idx2 = 0;
      MRegion_ptr mr2;
      while ((mr2 = List_Next_Entry(vregs,&idx2))) {
	if (mr2 != mr && !MEnt_IsMarked(mr2,mkid)) {
	  MEnt_Mark(mr2,mkid);
	  List_Add(cell_list,mr2);
	  if (MEnt_PType(mr2) == PGHOST) {
	    if (ptype == GHOST || ptype == USED) {
	      lid = MEnt_ID(mr2);
	      nadj_cellids->push_back(lid-1);
	    }
	  }
	  else {
	    if (ptype == GHOST || ptype == USED) {
	      lid = MEnt_ID(mr2);
	      nadj_cellids->push_back(lid-1);
	    }
	  }
	}
      }
      List_Delete(vregs);
    }

    List_Delete(rvertices);
    
  }
  else if (cell_dimension() == 2) {

    MFace_ptr mf = (MFace_ptr) cell_id_to_handle[cellid];

    List_ptr fverts = MF_Vertices(mf,1,0);
    int idx = 0;
    MVertex_ptr mv;
    while ((mv = List_Next_Entry(fverts,&idx))) {
      List_ptr vfaces = MV_Faces(mv);
      int idx2 = 0;
      MFace_ptr mf2;
      while ((mf2 = List_Next_Entry(vfaces,&idx2))) {
	if (mf2 != mf && !MEnt_IsMarked(mf2,mkid)) {
	  MEnt_Mark(mf2,mkid);
	  List_Add(cell_list,mf2);
	  if (MEnt_PType(mf2) == PGHOST) {
	    if (ptype == GHOST || ptype == USED) {
	      lid = MEnt_ID(mf2);
	      nadj_cellids->push_back(lid-1);
	    }
	  }
	  else {
	    if (ptype == GHOST || ptype == USED) {
	      lid = MEnt_ID(mf2);
	      nadj_cellids->push_back(lid-1);
	    }
	  }
	}
      }
      List_Delete(vfaces);
    }

    List_Delete(fverts);

  }

  List_Unmark(cell_list,mkid);
  List_Delete(cell_list);
  MSTK_FreeMarker(mkid);

} // Mesh_MSTK::cell_get_node_adj_cells


    
// Node coordinates - 3 in 3D and 2 in 2D
    
void Mesh_MSTK::node_get_coordinates (const Entity_ID nodeid, AmanziGeometry::Point *ncoords) const
{    
  MEntity_ptr vtx;
  double coords[3];
  int spdim = space_dimension();
  
  ASSERT(ncoords != NULL);

  vtx = vtx_id_to_handle[nodeid];

  MV_Coords(vtx,coords);
  ncoords->set(spdim,coords);

} // Mesh_MSTK::node_get_coordinates


// Coordinates of cells in standard order (Exodus II convention)
// STANDARD CONVENTION WORKS ONLY FOR STANDARD CELL TYPES IN 3D
// For a general polyhedron this will return the node coordinates in
// arbitrary order
// Number of nodes is vector size divided by number of spatial dimensions

void Mesh_MSTK::cell_get_coordinates (const Entity_ID cellid, std::vector<AmanziGeometry::Point> *ccoords) const
{    
  MEntity_ptr cell;
  double coords[3];
  int nn, result;
  int spdim = space_dimension(), celldim = cell_dimension();

  ASSERT(ccoords != NULL);

  cell = cell_id_to_handle[cellid];
      
  if (celldim == 3) {
    List_ptr rverts = MR_Vertices(cell);
    
    nn = List_Num_Entries(rverts);

    ccoords->resize(nn);
    std::vector<AmanziGeometry::Point>::iterator it = ccoords->begin();

    for (int i = 0; i < nn; ++i) {
      MV_Coords(List_Entry(rverts,i),coords);
      it->set(spdim,coords); // same as (*it).set()
      ++it; 
    }    

    List_Delete(rverts);
  }
  else if (celldim == 2) {
    List_ptr fverts = MF_Vertices(cell,1,0);

    nn = List_Num_Entries(fverts);

    ccoords->resize(nn);
    std::vector<AmanziGeometry::Point>::iterator it = ccoords->begin();

    for (int i = 0; i < nn; ++i) {
      MV_Coords(List_Entry(fverts,i),coords);
      it->set(spdim,coords); // same as (*it).set()
      ++it;
    }

    List_Delete(fverts);
  }

} // Mesh_MSTK::cell_get_coordinates




// Face coordinates - conventions same as face_to_nodes call 
// Number of nodes is the vector size divided by number of spatial dimensions
    
void Mesh_MSTK::face_get_coordinates (const Entity_ID faceid, std::vector<AmanziGeometry::Point> *fcoords) const
{
  MEntity_ptr genface;
  double coords[3];
  int spdim = space_dimension(), celldim = cell_dimension();

  ASSERT(fcoords != NULL);

  genface = face_id_to_handle[faceid];

  if (celldim == 3) {
    int dir = !faceflip[faceid];

    List_ptr fverts = MF_Vertices((MFace_ptr) genface,dir,0);

    int nn = List_Num_Entries(fverts);
    fcoords->resize(nn);
    std::vector<AmanziGeometry::Point>::iterator it = fcoords->begin();
	
    for (int i = 0; i < nn; ++i) {
      MV_Coords(List_Entry(fverts,i),coords);
      it->set(spdim,coords); // same as (*it).set()
      ++it;
    }

    List_Delete(fverts);
  }
  else { // Planar mesh or Surface mesh embedded in 3D
    MVertex_ptr ev[2];
    if (!faceflip[faceid]) {
      ev[0] = ME_Vertex((MEdge_ptr)genface,0);
      ev[1] = ME_Vertex((MEdge_ptr)genface,1);
    }
    else {
      ev[1] = ME_Vertex((MEdge_ptr)genface,0);
      ev[0] = ME_Vertex((MEdge_ptr)genface,1);
    }

    fcoords->resize(2);

    MV_Coords(ev[0],coords);
    ((*fcoords)[0]).set(spdim,coords);
	
    MV_Coords(ev[1],coords);
    ((*fcoords)[1]).set(spdim,coords);
  }

} // Mesh_MSTK::face_get_coordinates
  


// Modify a node's coordinates

void Mesh_MSTK::node_set_coordinates(const AmanziMesh::Entity_ID nodeid, 
                                      const double *coords) {
  MVertex_ptr v = vtx_id_to_handle[nodeid];
  MV_Set_Coords(v,(double *)coords);
}

void Mesh_MSTK::node_set_coordinates(const AmanziMesh::Entity_ID nodeid, 
                                     const AmanziGeometry::Point coords) {
  MVertex_ptr v = vtx_id_to_handle[nodeid];

  double coordarray[3] = {0.0,0.0,0.0};
  for (int i = 0; i < Mesh::space_dimension(); i++)
    coordarray[i] = coords[i];

  MV_Set_Coords(v,(double *)coordarray);
}


MSet_ptr Mesh_MSTK::build_set(const AmanziGeometry::RegionPtr region,
                              const Entity_kind kind) const {

  int celldim = Mesh::cell_dimension();
  int spacedim = Mesh::space_dimension();
  AmanziGeometry::GeometricModelPtr gm = Mesh::geometric_model();

  // Modify region/set name by prefixing it with the type of entity requested

  std::string internal_name = internal_name_of_set(region,kind);

  // Create entity set based on the region defintion  

  MSet_ptr mset;
  MType enttype;
  switch (kind) {      
  case CELL:    // cellsets      

    enttype = (celldim == 3) ? MREGION : MFACE;
    mset = MSet_New(mesh,internal_name.c_str(),enttype);
      
    if (region->type() == AmanziGeometry::BOX ||
        region->type() == AmanziGeometry::COLORFUNCTION) {

      int ncell = num_entities(CELL, USED);              

      for (int icell = 0; icell < ncell; icell++)
        if (region->inside(cell_centroid(icell)))
          MSet_Add(mset,cell_id_to_handle[icell]);

    }
    else if (region->type() == AmanziGeometry::POINT) {
      AmanziGeometry::Point vpnt(spacedim);
      AmanziGeometry::Point rgnpnt(spacedim);

      rgnpnt = ((AmanziGeometry::PointRegionPtr)region)->point();
        
      int nnode = num_entities(NODE, USED);
      double mindist2 = 1.e+16;
      int minnode = -1;
        
      int inode;
      for (inode = 0; inode < nnode; inode++) {
                  
        node_get_coordinates(inode, &vpnt);                  
        double dist2 = (vpnt-rgnpnt)*(vpnt-rgnpnt);
 
        if (dist2 < mindist2) {
          mindist2 = dist2;
          minnode = inode;
          if (mindist2 <= 1.0e-32)
            break;
        }
      }

      Entity_ID_List cells, cells1;

      node_get_cells(minnode,USED,&cells);
      
      int ncells = cells.size();
      for (int ic = 0; ic < ncells; ic++) {
        Entity_ID icell = cells[ic];
        
        // Check if point is contained in cell            
        if (point_in_cell(rgnpnt,icell))
          MSet_Add(mset,cell_id_to_handle[icell]);
      }

    }
    else if (region->type() == AmanziGeometry::PLANE) {

      if (celldim == 2) {

        int ncells = num_entities(CELL, USED);              
        for (int ic = 0; ic < ncells; ic++) {

          std::vector<AmanziGeometry::Point> ccoords(spacedim);

          cell_get_coordinates(ic, &ccoords);

          bool on_plane = true;
          for (int j = 0; j < ccoords.size(); ++j) {
            if (!region->inside(ccoords[j])) {
              on_plane = false;
              break;
            }
          }
                  
          if (on_plane)
            MSet_Add(mset,cell_id_to_handle[ic]);
        }
      }

    }
    else if (region->type() == AmanziGeometry::LOGICAL) {
      // will process later in this subroutine
    }
    else if (region->type() == AmanziGeometry::LABELEDSET) {
      // Just retrieve and return the set

      AmanziGeometry::LabeledSetRegionPtr lsrgn = 
        dynamic_cast<AmanziGeometry::LabeledSetRegionPtr> (region);
      std::string label = lsrgn->label();
      std::string entity_type = lsrgn->entity_str();

      if (entity_type != "CELL") {
        Errors::Message mesg("Entity type of labeled set region and build_set request do not match");
        amanzi_throw(mesg);
      }

      mset = MESH_MSetByName(mesh,internal_name.c_str());
    }
    else {
      Errors::Message mesg("Region type not applicable/supported for cell sets");
      amanzi_throw(mesg);
    }
      
    break;
      
  case FACE:  // sidesets
    //
    // Commented out so that we can ask for a face set in a 3D box
    //
    //          if (region->dimension() != celldim-1) 
    //            {
    //              std::cerr << "No region of dimension " << celldim-1 << " defined in geometric model" << std::endl;
    //              std::cerr << "Cannot construct cell set from region " << setname << std::endl;
    //            }

    enttype = (celldim == 3) ? MFACE : MEDGE;
    mset = MSet_New(mesh,internal_name.c_str(),enttype);

    if (region->type() == AmanziGeometry::BOX)  {

      int nface = num_entities(FACE, USED);
        
      for (int iface = 0; iface < nface; iface++) {
        if (region->inside(face_centroid(iface)))
          MSet_Add(mset,face_id_to_handle[iface]);

      }
    }
    else if (region->type() == AmanziGeometry::PLANE ||
             region->type() == AmanziGeometry::POLYGON) {

      int nface = num_entities(FACE, USED);
              
      for (int iface = 0; iface < nface; iface++) {
        std::vector<AmanziGeometry::Point> fcoords(spacedim);
            
        face_get_coordinates(iface, &fcoords);
            
        bool on_plane = true;
        for (int j = 0; j < fcoords.size(); ++j) {
          if (!region->inside(fcoords[j])) {
            on_plane = false;
            break;
          }
        }
                  
        if (on_plane)
          MSet_Add(mset,face_id_to_handle[iface]);
      }

    }
    else if (region->type() == AmanziGeometry::LABELEDSET) {
      // Just retrieve and return the set

      AmanziGeometry::LabeledSetRegionPtr lsrgn = 
        dynamic_cast<AmanziGeometry::LabeledSetRegionPtr> (region);
      std::string label = lsrgn->label();
      std::string entity_type = lsrgn->entity_str();

      if (entity_type != "FACE") {
        Errors::Message mesg("Entity type of labeled set region and build_set request do not match");
        amanzi_throw(mesg);
      }

      mset = MESH_MSetByName(mesh,internal_name.c_str());
    }
    else if (region->type() == AmanziGeometry::LOGICAL) {
      // Will handle it later in the routine
    }
    else {
      Errors::Message mesg("Region type not applicable/supported for face sets");
      amanzi_throw(mesg);
    }
    break;

  case NODE: // Nodesets

    enttype = MVERTEX;
    mset = MSet_New(mesh,internal_name.c_str(),enttype);

    if (region->type() == AmanziGeometry::BOX ||
        region->type() == AmanziGeometry::PLANE ||
        region->type() == AmanziGeometry::POLYGON ||
        region->type() == AmanziGeometry::POINT) {

      int nnode = num_entities(NODE, USED);

      for (int inode = 0; inode < nnode; inode++) {

        AmanziGeometry::Point vpnt(spacedim);
        node_get_coordinates(inode, &vpnt);
                  
        if (region->inside(vpnt)) {
          MSet_Add(mset,vtx_id_to_handle[inode]);

          // Only one node per point region
          if (region->type() == AmanziGeometry::POINT)
            break;      
        }
      }
    }
    else if (region->type() == AmanziGeometry::LABELEDSET) {
      // Just retrieve and return the set

      AmanziGeometry::LabeledSetRegionPtr lsrgn = 
        dynamic_cast<AmanziGeometry::LabeledSetRegionPtr> (region);
      std::string label = lsrgn->label();
      std::string entity_type = lsrgn->entity_str();

      if (entity_type != "FACE") {
        Errors::Message mesg("Entity type of labeled set region and build_set request do not match");
        amanzi_throw(mesg);
      }

      mset = MESH_MSetByName(mesh,internal_name.c_str());
    }
    else if (region->type() == AmanziGeometry::LOGICAL) {
      // We will handle it later in the routine
    }
    else {
      Errors::Message mesg("Region type not applicable/supported for node sets");
      amanzi_throw(mesg);
    }
      
    break;
  }


  if (region->type() == AmanziGeometry::LOGICAL) {
    AmanziGeometry::LogicalRegionPtr boolregion = (AmanziGeometry::LogicalRegionPtr) region;
    std::vector<std::string> region_names = boolregion->component_regions();
    int nreg = region_names.size();
    
    std::vector<MSet_ptr> msets;
    std::vector<AmanziGeometry::RegionPtr> regions;
    
    for (int r = 0; r < nreg; r++) {
      AmanziGeometry::RegionPtr rgn1 = gm->FindRegion(region_names[r]);
      regions.push_back(rgn1);

      // Did not find the region
      if (rgn1 == NULL) {
        std::stringstream mesg_stream;
        mesg_stream << "Geometric model has no region named " << 
          region_names[r];
        Errors::Message mesg(mesg_stream.str());
        amanzi_throw(mesg);
      }
        
      internal_name = internal_name_of_set(rgn1,kind);
      MSet_ptr mset1 = MESH_MSetByName(mesh,internal_name.c_str());
      if (!mset1)        
        mset1 = build_set(rgn1,kind);  // Recursive call

      msets.push_back(mset1);
    }

    // Check the entity types of the sets are consistent with the
    // entity type of the requested set

    for (int ms = 0; ms < msets.size(); ms++)
      if (MSet_EntDim(msets[ms]) != enttype) {
        Errors::Message 
          mesg("Amanzi cannot operate on sets of different entity types");
        amanzi_throw(mesg);               
      }
    
    int mkid = MSTK_GetMarker();
      
    if (boolregion->operation() == AmanziGeometry::COMPLEMENT) {
      
      for (int ms = 0; ms < msets.size(); ms++)
        MSet_Mark(msets[ms],mkid);
      
      int idx = 0;
      switch (enttype) {
      case MREGION:
        MRegion_ptr mr;
        while ((mr = MESH_Next_Region(mesh,&idx))) 
          if (!MEnt_IsMarked(mr,mkid))
            MSet_Add(mset,mr);
        break;        
      case MFACE: 
        MFace_ptr mf;
        while ((mf = MESH_Next_Face(mesh,&idx))) 
          if (!MEnt_IsMarked(mf,mkid))
            MSet_Add(mset,mf);
        break;
      case MEDGE:
        MEdge_ptr me;
        while ((me = MESH_Next_Edge(mesh,&idx))) 
          if (!MEnt_IsMarked(me,mkid))
            MSet_Add(mset,me);
      case MVERTEX:
        MVertex_ptr mv;
        while ((mv = MESH_Next_Vertex(mesh,&idx))) 
          if (!MEnt_IsMarked(mv,mkid))
            MSet_Add(mset,mv);
        break;
      }
      
      for (int ms = 0; ms < msets.size(); ms++)
        MSet_Unmark(msets[ms],mkid);

    }
    else if (boolregion->operation() == AmanziGeometry::UNION) {

      for (int ms = 0; ms < msets.size(); ms++) {
        MEntity_ptr ment;
        int idx = 0;
        while ((ment = MSet_Next_Entry(msets[ms],&idx))) 
          if (!MEnt_IsMarked(ment,mkid)) {
            MSet_Add(mset,ment);
            MEnt_Mark(ment,mkid);
          }
      }
      MSet_Unmark(mset,mkid);
      
    }
    else if (boolregion->operation() == AmanziGeometry::SUBTRACT) {

      /* Mark entities in all sets except the first */
      
      for (int ms = 1; ms < msets.size(); ms++)
        MSet_Mark(msets[ms],mkid);
      
      /* Look for entities in the first set but not in 
         any of the other sets */
      MEntity_ptr ment;
      int idx = 0;
      while ((ment = MSet_Next_Entry(msets[0],&idx))) 
        if (!MEnt_IsMarked(ment,mkid)) {
          MSet_Add(mset,ment);
          MEnt_Mark(ment,mkid);
        }
      
      for (int ms = 1; ms < msets.size(); ms++)
        MSet_Unmark(msets[ms],mkid);

    }
    else if (boolregion->operation() == AmanziGeometry::INTERSECT) {

      /* Can't do this using markers alone - need attributes */
      
      MAttrib_ptr matt = MAttrib_New(mesh,"XSECTATT",INT,MALLTYPE);
      
      for (int ms = 0; ms < msets.size(); ms++) {
        MEntity_ptr ment;
        int idx = 0;
        while ((ment = MSet_Next_Entry(msets[ms],&idx))) {
          int ival;
          double rval;
          void *pval;
          MEnt_Get_AttVal(ment,matt,&ival,&rval,&pval);
          ival++;
          MEnt_Set_AttVal(ment,matt,ival,rval,pval);
        }
      }
      
      for (int ms = 0; ms < msets.size(); ms++) {
        MEntity_ptr ment;
        int idx = 0;
        while ((ment = MSet_Next_Entry(msets[ms],&idx))) {
          int ival;
          double rval;
          void *pval;
          MEnt_Get_AttVal(ment,matt,&ival,&rval,&pval);
          if ((ival == msets.size()) && !MEnt_IsMarked(ment,mkid)) {
            /* entity is in all sets */
            MSet_Add(mset,ment);
            MEnt_Mark(ment,mkid);
          }
        }
      }
      
      MSet_Unmark(mset,mkid);
      
      for (int ms = 0; ms < msets.size(); ms++) {
        MEntity_ptr ment;
        int idx = 0;
        while ((ment = MSet_Next_Entry(msets[ms],&idx)))
          MEnt_Rem_AttVal(ment,matt);
      }
      MAttrib_Delete(matt);
    }

    MSTK_FreeMarker(mkid);

    for (int ms = 0; ms < msets.size(); ms++) {
      MSet_Unmark(msets[ms],mkid);
      if (regions[ms]->lifecycle() == AmanziGeometry::TEMPORARY)
        MSet_Delete(msets[ms]);
    }

  }

  return mset;
}


// Get list of entities of type 'category' in set specified by setname

void Mesh_MSTK::get_set_entities (const std::string setname, 
				  const Entity_kind kind, 
				  const Parallel_type ptype, 
				  std::vector<Entity_ID> *setents) const 
{
  int idx, i, lid;
  MSet_ptr mset, mset1;
  MEntity_ptr ment;
  bool found(false);
  int celldim = Mesh::cell_dimension();
  int spacedim = Mesh::space_dimension();
  const Epetra_Comm *epcomm = get_comm();

  assert(setents != NULL);
  
  setents->clear();

  AmanziGeometry::GeometricModelPtr gm = Mesh::geometric_model();

  // Is there an appropriate region by this name?

  AmanziGeometry::RegionPtr rgn = gm->FindRegion(setname);

  // Did not find the region
  
  if (rgn == NULL) {
    std::stringstream mesg_stream;
    mesg_stream << "Geometric model has no region named " << setname;
    Errors::Message mesg(mesg_stream.str());
    amanzi_throw(mesg);
  }


  std::string internal_name = internal_name_of_set(rgn,kind);

  // If region is of type labeled set and a mesh set should have been
  // initialized from the input file
  
  if (rgn->type() == AmanziGeometry::LABELEDSET)
    {
      AmanziGeometry::LabeledSetRegionPtr lsrgn = dynamic_cast<AmanziGeometry::LabeledSetRegionPtr> (rgn);
      std::string label = lsrgn->label();
      std::string entity_type = lsrgn->entity_str();

      if ((kind == CELL && entity_type != "CELL") ||
          (kind == FACE && entity_type != "FACE") ||
          (kind == NODE && entity_type != "NODE"))
        {
          std::stringstream mesg_stream;
          mesg_stream << "Found labeled set region named " << setname << " but it contains entities of type " << entity_type << ", not the requested type";
          Errors::Message mesg(mesg_stream.str());
          amanzi_throw(mesg);
        } 

      mset1 = MESH_MSetByName(mesh,internal_name.c_str());
      
      if (!mset1)
        {
          std::stringstream mesg_stream;
          mesg_stream << "Mesh set " << setname << " should have been read in as set " << label << " from the mesh file.";
          Errors::Message mesg(mesg_stream.str());
          amanzi_throw(mesg);
        }
    }
  else
    {
      // Modify region/set name by prefixing it with the type of
      // entity requested

      mset1 = MESH_MSetByName(mesh,internal_name.c_str());

      // Make sure we retrieved a mesh set with the right kind of entities

      MType entdim;

      switch (kind)
        {
        case CELL:
          if (celldim == 3)
            entdim = MREGION;
          else if (celldim == 2)
            entdim = MFACE;
          break;
        case FACE:
          if (celldim == 3)
            entdim = MFACE;
          else if (celldim == 2)
            entdim = MEDGE;
          break;
        case NODE:
          entdim = MVERTEX;
        }

      // If not, can we find a mesh set with the right name and right
      // kind of entities

      char setname1[256];

      if (mset1 && MSet_EntDim(mset1) != entdim) 
        {
          idx = 0;
          while ((mset1 = MESH_Next_MSet(mesh,&idx))) 
            {
              MSet_Name(mset1,setname1);
              
              if (MSet_EntDim(mset1) == entdim &&
                  strcmp(setname1,internal_name.c_str()) == 0)
                break;
            }
        }
    }

  // All attempts to find the set failed so it must not exist - build it

  if (mset1 == NULL) 
    mset1 = build_set(rgn, kind);


  /* Check if no processor got any mesh entities */

  int nent_loc = MSet_Num_Entries(mset1);

#ifdef DEBUG
  int nent_glob;

  epcomm->SumAll(&nent_loc,&nent_glob,1);
  if (nent_glob == 0) {
    std::stringstream mesg_stream;
    mesg_stream << "Could not retrieve any mesh entities for set " << setname << std::endl;
    Errors::Message mesg(mesg_stream.str());
    Exceptions::amanzi_throw(mesg);
  }
#endif
  
  setents->resize(nent_loc);
  Entity_ID_List::iterator it = setents->begin();
  if (nent_loc) {

    nent_loc = 0; // reset and count to get the real number

    switch (ptype) {
    case OWNED:
      idx = 0;
      while ((ment = MSet_Next_Entry(mset1,&idx))) {
        if (MEnt_PType(ment) != PGHOST) {
          *it = MEnt_ID(ment)-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++nent_loc;
        }
      }
      break;
    case GHOST:
      idx = 0;
      while ((ment = MSet_Next_Entry(mset1,&idx))) {
        if (MEnt_PType(ment) == PGHOST) {
          *it = MEnt_ID(ment)-1;  // assign to next spot by dereferencing iterator
	  ++it;
	  ++nent_loc;
        }
      }
      break;
    case USED:
      idx = 0;
      while ((ment = MSet_Next_Entry(mset1,&idx))) {
        *it = MEnt_ID(ment)-1;  // assign to next spot by dereferencing iterator
	++it;
	++nent_loc;
      }
      break;
    }
    
    setents->resize(nent_loc);
  }
      
    /* Check if there were no entities left on any processor after
       extracting the appropriate category of entities */
    
#ifdef DEBUG
  epcomm->SumAll(&nent_loc,&nent_glob,1);
  
  if (nent_glob == 0) {
    std::stringstream mesg_stream;
    mesg_stream << "Could not retrieve any mesh entities of type " << setkind << " for set " << setname << std::endl;
    Errors::Message mesg(mesg_stream.str());
    Exceptions::amanzi_throw(mesg);
  }
#endif
    
} // Mesh_MSTK::get_set_entities (by set name) 


void Mesh_MSTK::get_set_entities (const char *setname,
				  const Entity_kind kind, 
				  const Parallel_type ptype, 
				  std::vector<Entity_ID> *setents) const 
{
  std::string setname1(setname);

  get_set_entities(setname1,kind,ptype,setents);
  
} // Mesh_MSTK::get_set_entities (by set name) 


void Mesh_MSTK::get_set_entities (const Set_ID setid, 
				  const Entity_kind kind, 
				  const Parallel_type ptype, 
				  std::vector<Entity_ID> *setents) const 
{
  AmanziGeometry::GeometricModelPtr gm = Mesh::geometric_model();
  AmanziGeometry::RegionPtr rgn = gm->FindRegion(setid);

  std::cerr << "DEPRECATED METHOD!" << std::endl;
  std::cerr << "Call get_set_entities with setname instead of setid" << std::endl;

  if (!rgn)
    {
      std::cerr << "No region with id" << setid << std::endl;
    }

  get_set_entities(rgn->name(),kind,ptype,setents);
  
} // Mesh_MSTK::get_set_entities (by set id) 



// Get number of entities of type 'ptype' in set

unsigned int Mesh_MSTK::get_set_size (const char *setname,
                                      const Entity_kind kind, 
                                      const Parallel_type ptype) const 
{
  Entity_ID_List setents;
  std::string setname1(setname);

  get_set_entities(setname1,kind,ptype,&setents);
  
  return setents.size();
  
} // Mesh_MSTK::get_set_size (by set name)

// Get number of entities of type 'ptype' in set

unsigned int Mesh_MSTK::get_set_size (const std::string setname, 
                                      const Entity_kind kind, 
                                      const Parallel_type ptype) const 
{
  Entity_ID_List setents;

  get_set_entities(setname,kind,ptype,&setents);
  
  return setents.size();
  
} // Mesh_MSTK::get_set_size (by set name)



// Get number of entities of type 'ptype' in set

unsigned int Mesh_MSTK::get_set_size (const Set_ID setid, 
                                      const Entity_kind kind, 
                                      const Parallel_type ptype) const 
{
  
  Entity_ID_List setents;

  get_set_entities(setid,kind,ptype,&setents);
  
  return setents.size();
  
} // Mesh_MSTK::get_set_size (by set id)



// Parent entity in the source mesh if mesh was derived from another mesh

Entity_ID Mesh_MSTK::entity_get_parent (const Entity_kind kind, const Entity_ID entid) const
{
  int ival;
  double rval;
  void *pval;
  MEntity_ptr ment;
  MAttrib_ptr att;

  switch(kind) {
  case CELL:
    att = (cell_dimension() == 3) ? rparentatt : fparentatt;
    ment = (MEntity_ptr) cell_id_to_handle[entid];
    break;
  case FACE:
    att = (cell_dimension() == 3) ? fparentatt : eparentatt;
    ment = (MEntity_ptr) face_id_to_handle[entid];
    break;
  case NODE:
    if (!vparentatt) return 0;
    att = vparentatt;
    ment = (MEntity_ptr) vtx_id_to_handle[entid];
    break;
  }
  
  if (!att) return 0;

  MEnt_Get_AttVal(ment,att,&ival,&rval,&pval);
  if (pval) 
    return MEnt_ID((MEntity_ptr)pval)-1;
  else
    return 0;
}




// Epetra map for cells - basically a structure specifying the
// global IDs of cells owned or used by this processor

// Amanzi/Epetra want global IDs to start at 0

void Mesh_MSTK::init_cell_map ()
{
  int *cell_gids;
  int ncell, idx, i;
  MEntity_ptr ment;
  const Epetra_Comm *epcomm = get_comm();

  if (!serial_run) {

    // For parallel runs create map without and with ghost cells included
    // Also, put in owned cells before the ghost cells

    int nowned = MSet_Num_Entries(OwnedCells);
    int nnotowned = MSet_Num_Entries(GhostCells);

    cell_gids = new int[nowned+nnotowned];
    
    ncell = nowned;
   
    idx = 0; i = 0;
    while ((ment = MSet_Next_Entry(OwnedCells,&idx)))
      cell_gids[i++] = MEnt_GlobalID(ment)-1;

    cell_map_wo_ghosts_ = new Epetra_Map(-1,ncell,cell_gids,0,*epcomm);
    


    ncell += nnotowned;

    idx = 0; 
    while ((ment = MSet_Next_Entry(GhostCells,&idx)))
      cell_gids[i++] = MEnt_GlobalID(ment)-1;
    
    cell_map_w_ghosts_ = new Epetra_Map(-1,ncell,cell_gids,0,*epcomm);

  }
  else {    
    ncell = MSet_Num_Entries(OwnedCells);
    cell_gids = new int[ncell];

    idx = 0; i = 0;
    while ((ment = MSet_Next_Entry(OwnedCells,&idx)))      
      cell_gids[i++] = MEnt_ID(ment)-1;

    cell_map_wo_ghosts_ = new Epetra_Map(-1,ncell,cell_gids,0,*epcomm);
  }

  delete [] cell_gids;

} // Mesh_MSTK::init_cell_map




// Epetra map for faces - basically a structure specifying the
// global IDs of cells owned or used by this processor

void Mesh_MSTK::init_face_map ()
{
  int *face_gids, *extface_gids;
  int nface, n_extface, idx, i, j;
  MEntity_ptr ment;
  const Epetra_Comm *epcomm = get_comm();

  if (!serial_run) {

    // For parallel runs create map without and with ghost cells included
    // Also, put in owned cells before the ghost cells
    // Additionally, create a map of exterior faces only

    int nowned = MSet_Num_Entries(OwnedFaces);
    int nnotowned = MSet_Num_Entries(NotOwnedFaces);

    face_gids = new int[nowned+nnotowned];
    extface_gids = new int[nowned]; // Exterior faces
    
    idx = 0; i = 0; j = 0;
    while ((ment = MSet_Next_Entry(OwnedFaces,&idx))) {
      int gid = MEnt_GlobalID(ment);

      face_gids[i++] = gid-1;
     
      if (cell_dimension() == 3) {
        List_ptr fregs = MF_Regions((MFace_ptr) ment);
        if (List_Num_Entries(fregs) == 1)
          extface_gids[j++] = gid-1;
        if (fregs)
          List_Delete(fregs);
      }
      else if (cell_dimension() == 2) {
        List_ptr efaces = ME_Faces((MEdge_ptr) ment);
        if (List_Num_Entries(efaces) == 1)
          extface_gids[j++] = gid-1;
        if (efaces)
          List_Delete(efaces);
      }
    }
    n_extface = j;
    nface = nowned;
    
    face_map_wo_ghosts_ = new Epetra_Map(-1,nface,face_gids,0,*epcomm);

    extface_map_wo_ghosts_ = new Epetra_Map(-1,n_extface,extface_gids,0,*epcomm);

    idx = 0;
    while ((ment = MSet_Next_Entry(NotOwnedFaces,&idx))) 
      face_gids[i++] = MEnt_GlobalID(ment)-1;

    nface += nnotowned;

    face_map_w_ghosts_ = new Epetra_Map(-1,nface,face_gids,0,*epcomm);

  }
  else {

    if (cell_dimension() == 3) {

      nface = MESH_Num_Faces(mesh);
      face_gids = new int[nface];
      extface_gids = new int[nface];
      
      idx = 0; i = 0; j = 0;
      while ((ment = MESH_Next_Face(mesh,&idx))) {
        int gid = MEnt_ID(ment);
        face_gids[i++] = gid-1;
        
        List_ptr fregs = MF_Regions((MFace_ptr) ment);
        if (List_Num_Entries(fregs) == 1)
          extface_gids[j++] = gid-1;
        if (fregs)
          List_Delete(fregs);
      }
      
    }
    else if (cell_dimension() == 2) {
      
      nface = MESH_Num_Edges(mesh);
      face_gids = new int[nface];
      extface_gids = new int[nface];
      
      idx = 0; i = 0; j = 0;
      while ((ment = MESH_Next_Edge(mesh,&idx))) {
        int gid = MEnt_ID(ment);
        face_gids[i++] = gid-1;
        
        List_ptr efaces = ME_Faces((MEdge_ptr) ment);
        if (List_Num_Entries(efaces) == 1)
          extface_gids[j++] = gid-1;
        if (efaces)
          List_Delete(efaces);
      }
      
    }
    n_extface = j;

    face_map_wo_ghosts_ = new Epetra_Map(-1,nface,face_gids,0,*epcomm);

    extface_map_wo_ghosts_ = new Epetra_Map(-1,n_extface,extface_gids,0,*epcomm);
  }

  owned_to_extface_importer_ = new Epetra_Import(*extface_map_wo_ghosts_,*face_map_wo_ghosts_);

  delete [] face_gids;
  delete [] extface_gids;

} // Mesh_MSTK::init_face_map




// Epetra map for nodes - basically a structure specifying the
// global IDs of cells owned or used by this processor

void Mesh_MSTK::init_node_map ()
{
  int *vert_gids;
  int nvert, idx, i;
  MEntity_ptr ment;
  const Epetra_Comm *epcomm = get_comm();

  if (!serial_run) {

    // For parallel runs create map without and with ghost verts included
    // Also, put in owned cells before the ghost verts

    int nowned = MSet_Num_Entries(OwnedVerts);
    int nnotowned = MSet_Num_Entries(NotOwnedVerts);

    vert_gids = new int[nowned+nnotowned];
    
    idx = 0; i = 0;
    while ((ment = MSet_Next_Entry(OwnedVerts,&idx)))
      vert_gids[i++] = MEnt_GlobalID(ment)-1;

    nvert = nowned;
    
    node_map_wo_ghosts_ = new Epetra_Map(-1,nvert,vert_gids,0,*epcomm);
    


    idx = 0;
    while ((ment = MSet_Next_Entry(NotOwnedVerts,&idx)))
      vert_gids[i++] = MEnt_GlobalID(ment)-1;

    nvert += nnotowned;

    node_map_w_ghosts_ = new Epetra_Map(-1,nvert,vert_gids,0,*epcomm);

  }
  else {
    nvert = MSet_Num_Entries(OwnedVerts);

    vert_gids = new int[nvert];

    idx = 0; i = 0;
    while ((ment = MSet_Next_Entry(OwnedVerts,&idx)))
      vert_gids[i++] = MEnt_ID(ment)-1;

    node_map_wo_ghosts_ = new Epetra_Map(-1,nvert,vert_gids,0,*epcomm);
  }

  delete [] vert_gids;

} // Mesh_MSTK::init_node_map


// Global ID of any entity

Entity_ID Mesh_MSTK::GID(const Entity_ID lid, const Entity_kind kind) const
{
  MEntity_ptr ent;
  unsigned int gid;

  switch (kind) {
  case NODE:
    ent = vtx_id_to_handle[lid];
    break;

  case FACE:
    ent = face_id_to_handle[lid];
    break;

  case CELL:
    ent = cell_id_to_handle[lid];
    break;
  default:
    std::cerr << "Global ID requested for unknown entity type" << std::endl;
  }

  if (serial_run)
    return MEnt_ID(ent)-1;
  else
    return MEnt_GlobalID(ent)-1;

} // Mesh_MSTK::GID



// Procedure to perform all the post-mesh creation steps in a constructor

void Mesh_MSTK::post_create_steps_()
{


  // Pre-process the mesh to remove degenerate edges
  
  collapse_degen_edges();

  label_celltype();




  { // Keep together and in this order 

    init_pvert_lists();
    init_pcell_lists(); // cells MUST be initialized before faces
    init_pface_lists();

    
    // Create maps from local IDs to MSTK entity handles (must be after
    // the various init_p*_list calls)
    
    init_id_handle_maps();


  }
    
    
  init_global_ids();


  init_pface_dirs();


  // Create Epetra_maps
  
  init_cell_map();
  init_face_map();
  init_node_map();
  
  if (Mesh::geometric_model() != NULL)
    init_set_info();

}


// Some initializations

void Mesh_MSTK::clear_internals_ () 
{ 

  faceflip = NULL;

  cell_map_w_ghosts_ = cell_map_wo_ghosts_ = NULL;
  face_map_w_ghosts_ = face_map_wo_ghosts_ = NULL;
  node_map_w_ghosts_ = node_map_wo_ghosts_ = NULL;

  mesh = NULL;

  OwnedVerts = NotOwnedVerts = NULL;
  OwnedFaces = NotOwnedFaces = NULL;
  OwnedCells = GhostCells = NULL;

  celltype_att = NULL;
  rparentatt = fparentatt = eparentatt = vparentatt = NULL;
  
} // Mesh_MSTK::clear_internals




void Mesh_MSTK::init_id_handle_maps() {
  int i, lid, nv, nf, nc, idx;
  MVertex_ptr vtx;
  MEntity_ptr genface;  // Mesh face in 3D, edge in 2D
  MEntity_ptr gencell;  // Mesh region in 3D, face in 2D

  // If the mesh is dynamic, then this code has to be revisited
  
  // Amanzi has IDs starting from 0, MSTK has IDs starting from 1

  nv = MESH_Num_Vertices(mesh);

  vtx_id_to_handle.reserve(nv);

  idx = 0; lid = 1;
  while ((vtx = MSet_Next_Entry(OwnedVerts,&idx))) {
    MEnt_Set_ID(vtx,lid);
    vtx_id_to_handle[lid-1] = vtx;
    lid++;
  }
    
  idx = 0;
  while ((vtx = MSet_Next_Entry(NotOwnedVerts,&idx))) {
    MEnt_Set_ID(vtx,lid);
    vtx_id_to_handle[lid-1] = vtx;
    lid++;
  }
    

  nf = (cell_dimension() == 2) ? MESH_Num_Edges(mesh) : MESH_Num_Faces(mesh);

  face_id_to_handle.reserve(nf);

  idx = 0; lid = 1;
  while ((genface = MSet_Next_Entry(OwnedFaces,&idx))) {
    MEnt_Set_ID(genface,lid);
    face_id_to_handle[lid-1] = genface;
    lid++;
  }
  
  idx = 0;
  while ((genface = MSet_Next_Entry(NotOwnedFaces,&idx))) {
    MEnt_Set_ID(genface,lid);
    face_id_to_handle[lid-1] = genface;
    lid++;
  }
    


  nc = (cell_dimension() == 2) ? MESH_Num_Faces(mesh) : MESH_Num_Regions(mesh);

  cell_id_to_handle.reserve(nc);

  idx = 0; lid = 1;
  while ((gencell = MSet_Next_Entry(OwnedCells,&idx))) {
    MEnt_Set_ID(gencell,lid);
    cell_id_to_handle[lid-1] = gencell;
    lid++;
  }
    
  idx = 0;
  while ((gencell = MSet_Next_Entry(GhostCells,&idx))) {
    MEnt_Set_ID(gencell,lid);
    cell_id_to_handle[lid-1] = gencell;
    lid++;
  }
    
} // Mesh_MSTK::init_id_handle_maps



void Mesh_MSTK::init_global_ids() {
    
} // Mesh_MSTK::init_global_ids


void Mesh_MSTK::init_pvert_lists() {
  int idx = 0;
  MVertex_ptr vtx;

  // Get all vertices on this processor 

  NotOwnedVerts = MSet_New(mesh,"NotOwnedVerts",MVERTEX);
  OwnedVerts = MSet_New(mesh,"OwnedVerts",MVERTEX);

  idx = 0;
  while ((vtx = MESH_Next_Vertex(mesh,&idx))) {
    if (MV_PType(vtx) == PGHOST)
      MSet_Add(NotOwnedVerts,vtx);
    else
      MSet_Add(OwnedVerts,vtx);
  }

} // Mesh_MSTK::init_pvert_lists


void Mesh_MSTK::init_pface_lists() {
  int idx = 0;

  // Get all faces on this processor 

  if (cell_dimension() == 3) {

    MFace_ptr face;

    NotOwnedFaces = MSet_New(mesh,"NotOwnedFaces",MFACE);
    OwnedFaces = MSet_New(mesh,"OwnedFaces",MFACE);

    idx = 0;
    while ((face = MESH_Next_Face(mesh,&idx))) {
      if (MF_PType(face) == PGHOST)
	MSet_Add(NotOwnedFaces,face);
      else
	MSet_Add(OwnedFaces,face);
    }
  }
  else if (cell_dimension() == 2) {

    MEdge_ptr edge;

    NotOwnedFaces = MSet_New(mesh,"NotOwnedFaces",MFACE);
    OwnedFaces = MSet_New(mesh,"OwnedFaces",MFACE);

    idx = 0;
    while ((edge = MESH_Next_Edge(mesh,&idx))) {
      if (ME_PType(edge) == PGHOST)
	MSet_Add(NotOwnedFaces,edge);
      else
	MSet_Add(OwnedFaces,edge);
    }
  }
  else {
    std::cerr << "Not implemented for face dimension" << std::endl;
  }

  return;
} // Mesh_MSTK::init_pface_lists


void Mesh_MSTK::init_pface_dirs() {
  MRegion_ptr region0, region1;
  MFace_ptr face, face0, face1;
  MEdge_ptr edge;
  MAttrib_ptr attfc0, attfc1;
  int idx;
  int local_regid0, local_regid1;
  int remote_regid0, remote_regid1;
  int local_faceid0, local_faceid1;
  int remote_faceid0, remote_faceid1;

  int nf = (cell_dimension() == 2) ? MESH_Num_Edges(mesh) : MESH_Num_Faces(mesh);

  if (serial_run) {
    faceflip = new bool[nf];
    for (int i = 0; i < nf; ++i) faceflip[i] = false;
  }
  else {
    // Do some additional processing to see if ghost faces and their masters
    // are oriented the same way; if not, turn on flag to flip the directions
    // when returning to the application code

    if (cell_dimension() == 3) {
      attfc0 = MAttrib_New(mesh,"TMP_FC0_ATT",INT,MFACE);
      attfc1 = MAttrib_New(mesh,"TMP_FC1_ATT",INT,MFACE);
    }
    else if (cell_dimension() == 2) {
      attfc0 = MAttrib_New(mesh,"TMP_FC0_ATT",INT,MEDGE);
      attfc1 = MAttrib_New(mesh,"TMP_FC1_ATT",INT,MEDGE);
    }

    if (cell_dimension() == 3) {
    
      idx = 0;
      while ((face = MESH_Next_Face(mesh,&idx))) {
	if (MF_PType(face) != PINTERIOR) {
	  region0 = MF_Region(face,0);
	  if (region0)
	    MEnt_Set_AttVal(face,attfc0,MEnt_GlobalID(region0),0.0,NULL);
	    
	  region1 = MF_Region(face,1);
	  if (region1)
	    MEnt_Set_AttVal(face,attfc1,MEnt_GlobalID(region1),0.0,NULL);
	}
      }    
	
    }
    else if (cell_dimension() == 2) {

      idx = 0;
      while ((edge = MESH_Next_Edge(mesh,&idx))) {
	if (ME_PType(edge) != PINTERIOR) {
	  List_ptr efaces = ME_Faces(edge);
	    
	  face0 = List_Entry(efaces,0);
	  if (MF_EdgeDir(face0,edge) != 1) {
	    face1 = face0;
	    MEnt_Set_AttVal(edge,attfc1,MEnt_GlobalID(face1),0.0,NULL);

	    face0 = List_Entry(efaces,1);
	    if (face0) {
	      if (MF_EdgeDir(face0,edge) == 1)     // Sanity check
		MEnt_Set_AttVal(edge,attfc0,MEnt_GlobalID(face0),0.0,NULL);
	      else
		std::cerr << "Two faces using edge in same direction in 2D mesh" << std::endl;
	    }
	  }
	  else {
	    MEnt_Set_AttVal(edge,attfc0,MEnt_GlobalID(face0),0.0,NULL);
            face1 = List_Entry(efaces,1);
            if (face1)
              MEnt_Set_AttVal(edge,attfc1,MEnt_GlobalID(face1),0.0,NULL);
          }
          List_Delete(efaces);
	}
      }

    }  // else if (cell_dimension() == 2)



    MSTK_UpdateAttr(mesh,mpicomm);



    faceflip = new bool[nf];
    for (int i = 0; i < nf; ++i) faceflip[i] = false;
    
    if (cell_dimension() == 3) {
      double rval;
      void *pval;

      idx = 0;
      while ((face = MSet_Next_Entry(NotOwnedFaces,&idx))) {
      
	MEnt_Get_AttVal(face,attfc0,&remote_regid0,&rval,&pval);
	MEnt_Get_AttVal(face,attfc1,&remote_regid1,&rval,&pval);
      
	region0 = MF_Region(face,0);
	local_regid0 = region0 ? MEnt_GlobalID(region0) : 0;
	region1 = MF_Region(face,1);
	local_regid1 = region1 ? MEnt_GlobalID(region1) : 0;
      
	if (remote_regid1 == local_regid0 || 
	    remote_regid0 == local_regid1) {
	  int lid = MEnt_ID(face);
	  faceflip[lid-1] = true;
	}
	else { // Sanity Check
	
	  if (remote_regid1 != local_regid1 &&
	      remote_regid0 != local_regid0) {
	  
            std::stringstream mesg_stream;
            mesg_stream << "Face cells mismatch between master and ghost (processor " << myprocid << ")";
            Errors::Message mesg(mesg_stream.str());
            amanzi_throw(mesg);
	  }
	}
      }
    }
    else if (cell_dimension() == 2) {
      double rval;
      void *pval;

      idx = 0;
      while ((edge = MSet_Next_Entry(NotOwnedFaces,&idx))) {
      
	MEnt_Get_AttVal(edge,attfc0,&remote_faceid0,&rval,&pval);
	MEnt_Get_AttVal(edge,attfc1,&remote_faceid1,&rval,&pval);
      
	List_ptr efaces = ME_Faces(edge);
	face0 = List_Entry(efaces,0);
	face1 = List_Entry(efaces,1);
	if (MF_EdgeDir(face0,edge) != 1) {
	  face0 = List_Entry(efaces,1);
	  face1 = List_Entry(efaces,0);
	}
	local_faceid0 = face0 ? MEnt_GlobalID(face0) : 0;
	local_faceid1 = face1 ? MEnt_GlobalID(face1) : 0;
      
	if (remote_faceid1 == local_faceid0 || 
	    remote_faceid0 == local_faceid1) {
	  int lid = MEnt_ID(edge);
	  faceflip[lid-1] = true;
	}
	else { // Sanity Check
	
	  if (remote_faceid1 != local_faceid1 &&
	      remote_faceid0 != local_faceid0) {
	  
            std::stringstream mesg_stream;
            mesg_stream << "Face cells mismatch between master and ghost (processor " << myprocid << ")";
            Errors::Message mesg(mesg_stream.str());
            amanzi_throw(mesg);
	  }
	}
	List_Delete(efaces);
      }

    }
  }    
} // Mesh_MSTK::init_pface_dirs


void Mesh_MSTK::init_pcell_lists() {
  int idx = 0;

  if (cell_dimension() == 3) {
    MRegion_ptr region;

    OwnedCells = MSet_New(mesh,"OwnedCells",MREGION);
    GhostCells = MSet_New(mesh,"GhostCells",MREGION);

    idx = 0;
    while ((region = MESH_Next_Region(mesh,&idx))) {
      if (MR_PType(region) == PGHOST)
	MSet_Add(GhostCells,region);
      else
	MSet_Add(OwnedCells,region);
    }
  }
  else if (cell_dimension() == 2) {
    MFace_ptr face;

    OwnedCells = MSet_New(mesh,"OwnedCells",MFACE);
    GhostCells = MSet_New(mesh,"GhostCells",MFACE);

    idx = 0;
    while ((face = MESH_Next_Face(mesh,&idx))) {
      if (MF_PType(face) == PGHOST)
	MSet_Add(GhostCells,face);
      else
	MSet_Add(OwnedCells,face);
    }
  }
  else {
    Errors::Message mesg("Implemented only for 2D and 3D");
    amanzi_throw(mesg);
  }

  return;
} // Mesh_MSTK::init_pcell_lists





void Mesh_MSTK::init_set_info() {
  MSet_ptr mset;
  char setname[256];
  
  AmanziGeometry::GeometricModelPtr gm = Mesh::geometric_model();

  if (gm == NULL) { 
    Errors::Message mesg("Need region definitions to initialize sets");
    amanzi_throw(mesg);
  }
    

  unsigned int ngr = gm->Num_Regions();

  for (int i = 0; i < ngr; ++i) {
    AmanziGeometry::RegionPtr rgn = gm->Region_i(i);

    MType entdim;
    if (rgn->type() == AmanziGeometry::LABELEDSET) {

      AmanziGeometry::LabeledSetRegionPtr lsrgn =
        dynamic_cast<AmanziGeometry::LabeledSetRegionPtr> (rgn);

      std::string internal_name;
      std::string label = lsrgn->label();
      std::string entity_type_str = lsrgn->entity_str();

      if (entity_type_str == "CELL")
        internal_name = internal_name_of_set(rgn,CELL);
      else if (entity_type_str == "FACE")
        internal_name = internal_name_of_set(rgn,FACE);
      else if (entity_type_str == "NODE")
        internal_name = internal_name_of_set(rgn,NODE);

      mset = MESH_MSetByName(mesh,internal_name.c_str());
   
      if (!mset) {
	Errors::Message mesg("Missing labeled set \"" + label + "\" or error in input");
	amanzi_throw(mesg);
      }

      entdim = MSet_EntDim(mset);
      if (Mesh::cell_dimension() == 3) {

        if ((entity_type_str == "CELL" && entdim != MREGION) ||
            (entity_type_str == "FACE" && entdim != MFACE) ||
            (entity_type_str == "NODE" && entdim != MVERTEX)) {
          Errors::Message mesg("Mismatch of entity type in labeled set region and mesh set");
          amanzi_throw(mesg);
        }
      }
      else if (Mesh::cell_dimension() == 2) {
        if ((entity_type_str == "CELL" && entdim != MFACE) ||
            (entity_type_str == "FACE" && entdim != MEDGE) ||
            (entity_type_str == "NODE" && entdim != MVERTEX)) {
          std::cerr << "Mismatch of entity type in labeled set region and mesh set" << std::endl;
          throw std::exception();
        }
      }

      if (mset) {
        if (entities_deleted) {
          entdim = MSet_EntDim(mset);
          int idx=0;
          switch (entdim) {
          case MREGION:
            MRegion_ptr region;
            while ((region = List_Next_Entry(deleted_regions,&idx)))
              MSet_Rem(mset,region);
            break;
          case MFACE:
            MFace_ptr face;
            while (face = List_Next_Entry(deleted_faces,&idx))
              MSet_Rem(mset,face);
            break;
          case MEDGE:
            MEdge_ptr edge;
            while (edge = List_Next_Entry(deleted_edges,&idx))
              MSet_Rem(mset,edge);
            break;
          case MVERTEX:
            MVertex_ptr vertex;
            while (vertex = List_Next_Entry(deleted_vertices,&idx))
              MSet_Rem(mset,vertex);
            break;
          }
        }
      }
    }
    else { /* General region - we have to account for all kinds of
              entities being queried in a set defined by this 
              region */
      Entity_kind int_to_kind[3] = {NODE,FACE,CELL};

      for (int k = 0; k < 3; ++k) {
        Entity_kind kind = int_to_kind[k];
        
        std::string internal_name = internal_name_of_set(rgn,kind);

        mset = MESH_MSetByName(mesh,internal_name.c_str());

        if (mset) {
          if (entities_deleted) {
            entdim = MSet_EntDim(mset);
            int idx=0;
            switch (entdim) {
            case MREGION:
              MRegion_ptr region;
              while ((region = List_Next_Entry(deleted_regions,&idx)))
                MSet_Rem(mset,region);
              break;
            case MFACE:
              MFace_ptr face;
              while ((face = List_Next_Entry(deleted_faces,&idx)))
                MSet_Rem(mset,face);
              break;
            case MEDGE:
              MEdge_ptr edge;
              while ((edge = List_Next_Entry(deleted_edges,&idx)))
                MSet_Rem(mset,edge);
              break;
            case MVERTEX:
              MVertex_ptr vertex;
              while ((vertex = List_Next_Entry(deleted_vertices,&idx)))
                MSet_Rem(mset,vertex);
              break;
            }
          }
        }
      }

    }

  }
  

}  /* Mesh_MSTK::init_set_info */



void Mesh_MSTK::collapse_degen_edges() {
  const int topoflag=0; // Don't worry about violation of model classification
  int idx, idx2, evgid0, evgid1;
  MVertex_ptr ev0, ev1, vkeep, vdel;
  MEdge_ptr edge;
  MFace_ptr face;
  MRegion_ptr region;
  List_ptr eregs, efaces, vregs, vfaces;
  double len2;
  int ival;
  void *pval;
  Cell_type celltype;  

  idx = 0;
  while ((edge = MESH_Next_Edge(mesh,&idx))) {

    len2 = ME_LenSqr(edge);

    if (len2 <= 1.0e-32) {

      /* Degenerate edge  - must collapse */

      /* If its the first time, we have to allocate
         these lists */
      if (!entities_deleted) {
        deleted_vertices = List_New(0);
        deleted_edges = List_New(0);
        deleted_faces = List_New(0);
        deleted_regions = List_New(0);
      }

      entities_deleted = true;

      /* Collapse, choosing the vertex to be deleted and vertex to
	 be kept consistently. If topological constraints permit,
	 collapse the vertex with the higher global ID to the vertex
	 with the lower global ID. If they do not, reverse the
	 order. Since global IDs and topological constraints are the
	 same for master and slave edges and their nodes, we will not
	 have conflict between processors */

      int vdelid=0;

      ev0 = ME_Vertex(edge,0); evgid0 = MEnt_GlobalID(ev0);
      ev1 = ME_Vertex(edge,1); evgid1 = MEnt_GlobalID(ev1);

      if (evgid0 < evgid1) {
	vkeep = ev0;
	vdel = ev1;
        vdelid = MV_ID(vdel);
      }
      else {
	vkeep = ev1;
	vdel = ev0;
        vdelid = MV_ID(vdel);
      }

#if defined (MSTK_2_20rc1_OR_NEWER) || defined (MSTK_2_21rc1_OR_NEWER)

      List_ptr deleted_ents;
      vkeep = ME_Collapse(edge, vkeep, topoflag, &deleted_ents);

      if (!vkeep) {
	vkeep = vdel;
	vdel = (vkeep == ev0) ? ev1 : ev1;
        vdelid = MV_ID(vdel);

	vkeep = ME_Collapse(edge, vkeep, topoflag, &deleted_ents);
      }

      if (!vkeep) {
        Errors::Message mesg("Could not collapse degenerate edge. Expect computational issues with connected elements");
        amanzi_throw(mesg);
      }
      
      MEntity_ptr ent;
      int idx1 = 0;
      while ((ent = List_Next_Entry(deleted_ents,&idx1))) {
        switch (MEnt_Dim(ent)) {
        case MREGION:
          List_Add(deleted_regions,ent);
          break;
        case MFACE:
          List_Add(deleted_faces,ent);
          break;
        case MEDGE:
          List_Add(deleted_edges,ent);
          break;
        case MVERTEX:
          List_Add(deleted_vertices,ent);
          break;
        }
      }
      List_Delete(deleted_ents);

#else

      eregs = ME_Regions(edge);
      efaces = ME_Faces(edge);

      vkeep = ME_Collapse(edge, vkeep, topoflag);

      if (!vkeep) {
	vkeep = vdel;
	vdel = (vkeep == ev0) ? ev1 : ev1;
        vdelid = MV_ID(vdel);

	vkeep = ME_Collapse(edge, vkeep, topoflag);
      }

      if (!vkeep) {
        Errors::Message mesg("Could not collapse degenerate edge. Expect computational issues with connected elements");
        amanzi_throw(mesg);
      }
      
      vregs = MV_Regions(vkeep);
      vfaces = MV_Faces(vkeep);

      if (eregs) {
        MRegion_ptr reg;
        int idx1 = 0;
        while ((reg = List_Next_Entry(eregs,&idx1))) {
          if (!List_Contains(vregs,reg)) 
            List_Add(deleted_regions,reg);
        }
      }
      
      if (efaces) {
        MFace_ptr face;
        int idx1 = 0;
        while ((face = List_Next_Entry(efaces,&idx1))) {
          if (!List_Contains(vfaces,face))
            List_Add(deleted_faces,face);
        }
      }

      List_Add(deleted_edges,edge);

      List_Add(deleted_vertices,vdel);

      if (vregs) List_Delete(vregs);
      if (vfaces) List_Delete(vfaces);
      if (eregs) List_Delete(eregs);
      if (efaces) List_Delete(efaces);

#endif
    }

  }

} /* end Mesh_MSTK::collapse_degen_edges */


Cell_type Mesh_MSTK::MFace_Celltype(MFace_ptr face) {
  int nfv = MF_Num_Vertices(face);

  switch (nfv) {
  case 3:
    return TRI;
    break;
  case 4:
    return QUAD;
    break;
  default:
    return POLYGON;
  }
}

Cell_type Mesh_MSTK::MRegion_Celltype(MRegion_ptr region) {
  List_ptr rverts, rfaces;
  MFace_ptr face;
  int nrv, nrf, idx2, nquads;

  rverts = MR_Vertices(region);
  nrv = List_Num_Entries(rverts);
  List_Delete(rverts);
  
  nrf = MR_Num_Faces(region);
  
  switch (nrf) {
  case 4:
    return TET;
    break;
  case 5:
    
    nquads = 0;
    rfaces = MR_Faces(region);	  
    idx2 = 0;
    while ((face = List_Next_Entry(rfaces,&idx2)))
      if (MF_Num_Vertices(face) == 4)
        nquads++;
    List_Delete(rfaces);
    
    switch (nquads) {
    case 1:
      return PYRAMID;
      break;
    case 3:
      return PRISM;
      break;
    default:
      return POLYHED;
    }

    break;

  case 6:

    nquads = 0;
    rfaces = MR_Faces(region);	  
    idx2 = 0;
    while ((face = List_Next_Entry(rfaces,&idx2)))
      if (MF_Num_Vertices(face) == 4)
        nquads++;
    List_Delete(rfaces);
    
    if (nquads == 6) 
      return HEX;
    else 
      return POLYHED;

    break;

  default:
    return POLYHED;
  }

}

void Mesh_MSTK::label_celltype() {
  Cell_type ctype;
  int idx;
  MFace_ptr face;
  MRegion_ptr region;

  if (cell_dimension() == 2) 
    celltype_att = MAttrib_New(mesh,"Cell_type",INT,MFACE);
  else
    celltype_att = MAttrib_New(mesh,"Cell_type",INT,MREGION);

  if (cell_dimension() == 2) {

    idx = 0;
    while ((face = MESH_Next_Face(mesh,&idx))) {
      ctype = MFace_Celltype(face);
      MEnt_Set_AttVal(face,celltype_att,ctype,0.0,NULL);
    }
      
  }
  else if (cell_dimension() == 3) {

    idx = 0;
    while ((region = MESH_Next_Region(mesh,&idx))) {
      ctype = MRegion_Celltype(region);
      MEnt_Set_AttVal(region,celltype_att,ctype,0.0,NULL);

    }

  }

} /* Mesh_MSTK::label_celltypes */


//
// Epetra maps
//------------
    
    
inline 
const Epetra_Map& Mesh_MSTK::cell_map (const bool include_ghost) const
{
  if (serial_run)
    return *cell_map_wo_ghosts_;
  else
    return (include_ghost ? *cell_map_w_ghosts_ : *cell_map_wo_ghosts_);
}
    
inline 
const Epetra_Map& Mesh_MSTK::face_map (const bool include_ghost) const
{
  if (serial_run)
    return *face_map_wo_ghosts_;
  else
    return (include_ghost ? *face_map_w_ghosts_ : *face_map_wo_ghosts_);
}
    
inline 
const Epetra_Map& Mesh_MSTK::node_map (const bool include_ghost) const
{
  if (serial_run)
    return *node_map_wo_ghosts_;
  else
    return (include_ghost ? *node_map_w_ghosts_ : *node_map_wo_ghosts_);
}


inline
const Epetra_Map& Mesh_MSTK::exterior_face_map(void) const
{
  return *extface_map_wo_ghosts_;
}

inline
const Epetra_Import& Mesh_MSTK::exterior_face_importer(void) const
{
  return *owned_to_extface_importer_;
}

int Mesh_MSTK::generate_regular_mesh(Mesh_ptr mesh, double x0, double y0, 
				     double z0, double x1, double y1, 
				     double z1, int nx, int ny, int nz)
{
/*

  Index directions for classification templates

  k   j
  |  /
  | /
  |/___ i


  Model vertex, edge and face enumeration for classification templates 


         MODEL                   MODEL                  MODEL
         VERTICES                EDGES                  FACES

     7 ____________ 8          ______7_____           ____________  
      /|          /|          /|          /|         /|      2   /| 
     / |         / |       12/ |8      11/ | 	    / |  4      / | 
   5/___________/6 |        /_____3_____/  |6	   /___________/  | 
    |  |        |  |        |  |        |  | 	   |  |        | 5| 
    |  |________|__|        |  |_____5__|__| 	   |6 |_1______|__| 
    |  /3       |  /4      4|  /        |  / 	   |  /        |  / 
    | /         | /         | /9       2| /10	   | /      3  | /  
    |/__________|/          |/__________|/   	   |/__________|/   
   1             2                1
                                                   
                                                    Front  - Face1
						    Back   - Face2
						    Bottom - Face3
						    Top    - Face4
						    Left   - Face6
						    Right  - Face5

  Classification of mesh regions onto multiple material regions is not done
  here since the "geometric model" could have overlapping regions. Instead
  mesh sets are created as necessary based on point location in regions.

*/

  int i, j, k, ii, jj, kk, gid, gdim;
  double xyz[3], dx, dy, dz;
  MVertex_ptr ***verts, mv, rverts[8], fverts[4], everts[2];
  MEdge_ptr me;
  MFace_ptr mf;
  MRegion_ptr mr;
  int vgid_tmpl[3][3][3] = {{{1,4,5},{9,6,12},{3,8,7}},{{1,1,3},{3,1,4},{5,2,7}},{{2,2,6},{10,5,11},{4,6,8}}};
  int vgdim_tmpl[3][3][3]= {{{0,1,0},{1,2,1}, {0,1,0}},{{1,2,1},{2,3,2},{1,2,1}},{{0,1,0},{1,2,1},{0,1,0}}};
  int egdim_tmpl[3][3] = {{1,2,1},{2,3,2},{1,2,1}};
  int egid_tmpl2[3][3] = {{4,6,8},{1,1,2},{2,5,6}};  /* Y direction edges (iterating over i,k) */
  int egid_tmpl1[3][3] = {{9,6,12},{3,1,4},{10,5,11}}; /* Z direction edges (iterating over i,j)*/
  int egid_tmpl0[3][3] = {{1,1,3},{3,1,4},{5,2,7}}; /* X direction edges (iterating over j,k) */
  int fgdim_tmpl[3] = {2,3,2};
  int fgid_tmpl0[3] = {6,1,5};
  int fgid_tmpl1[3] = {1,1,2};
  int fgid_tmpl2[3] = {3,1,4};

  dx = (x1-x0)/nx;
  dy = (y1-y0)/ny;
  dz = (z1-z0)/nz;

  verts = (MVertex_ptr ***) malloc((nx+1)*sizeof(MVertex_ptr **));
  for (j = 0; j < nx+1; ++j) {
    verts[j] = (MVertex_ptr **) malloc((ny+1)*sizeof(MVertex_ptr *)); 
    for (k = 0; k < ny+1; ++k) 
      verts[j][k] = (MVertex_ptr *) malloc((nz+1)*sizeof(MVertex_ptr));
  }

  for (k = 0; k < nz+1; ++k) {
    xyz[2] = (k == nz) ? z1 : z0 + k*dz;
    kk =  (k%nz) ? 1 : (k ? 2 : 0);

    for (j = 0; j < ny+1; ++j) {
      xyz[1] = (j == ny) ? y1 : y0 + j*dy;      
      jj = (j%ny) ? 1 : (j ? 2 : 0);

      for (i = 0; i < nx+1; ++i) {
	xyz[0] = (i == nx) ? x1 : x0 + i*dx;
	ii = (i%nx) ? 1 : (i ? 2 : 0);
	
	mv = MV_New(mesh);
	MV_Set_Coords(mv,xyz);	
	verts[i][j][k] = mv;

	gdim  = vgdim_tmpl[ii][jj][kk];
	MV_Set_GEntDim(mv,gdim);

	gid = vgid_tmpl[ii][jj][kk];
	MV_Set_GEntID(mv,gid);
      }
    }
  }


  /* Create the edges explicitly to get the classification right */
  for (i = 0; i < nx+1; ++i) {
    for (j = 0; j < ny+1; ++j) {
      for (k = 0; k < nz; ++k) {
	me = ME_New(mesh);

	everts[0] = verts[i][j][k];
	everts[1] = verts[i][j][k+1];
	ME_Set_Vertex(me,0,everts[0]);
	ME_Set_Vertex(me,1,everts[1]);

	ii = (i%nx) ? 1 : (i ? 2 : 0);
	jj = (j%ny) ? 1 : (j ? 2 : 0);
	gdim = egdim_tmpl[ii][jj];
	gid = egid_tmpl2[ii][jj];

	ME_Set_GEntDim(me,gdim);
	ME_Set_GEntID(me,gid);
      }
    }
  }
	
  for (i = 0; i < nx+1; ++i) {
    for (k = 0; k < nz+1; ++k) {
      for (j = 0; j < ny; ++j) {
	me = ME_New(mesh);

	everts[0] = verts[i][j][k];
	everts[1] = verts[i][j+1][k];
	ME_Set_Vertex(me,0,everts[0]);
	ME_Set_Vertex(me,1,everts[1]);

	ii = (i%nx) ? 1 : (i ? 2 : 0);
	kk = (k%nz) ? 1 : (k ? 2 : 0);
	gdim = egdim_tmpl[ii][kk];
	gid = egid_tmpl1[ii][kk];

	ME_Set_GEntDim(me,gdim);
	ME_Set_GEntID(me,gid);
      }
    }
  }
	
  for (j = 0; j < ny+1; ++j) {
    for (k = 0; k < nz+1; ++k) {
      for (i = 0; i < nx; ++i) {
	me = ME_New(mesh);

	everts[0] = verts[i][j][k];
	everts[1] = verts[i+1][j][k];
	ME_Set_Vertex(me,0,everts[0]);
	ME_Set_Vertex(me,1,everts[1]);

	jj = (j%ny) ? 1 : (j ? 2 : 0);
	kk = (k%nz) ? 1 : (k ? 2 : 0);
	gdim = egdim_tmpl[jj][kk];
	gid = egid_tmpl0[jj][kk];

	ME_Set_GEntDim(me,gdim);
	ME_Set_GEntID(me,gid);
      }
    }
  }
	


  /* Create the faces explicitly to get the classification right */
  for (i = 0; i < nx+1; ++i) {
    for (j = 0; j < ny; ++j) {
      for (k = 0; k < nz; ++k) {
	mf = MF_New(mesh);

	fverts[0] = verts[i][j][k];
	fverts[1] = verts[i][j+1][k];
	fverts[2] = verts[i][j+1][k+1];
	fverts[3] = verts[i][j][k+1];
	MF_Set_Vertices(mf,4,fverts);

	ii = (i%nx) ? 1 : (i ? 2 : 0);
	gdim = fgdim_tmpl[ii];
	gid = fgid_tmpl0[ii];

	MF_Set_GEntDim(mf,gdim);
	MF_Set_GEntID(mf,gid);
      }
    }
  }
	
  for (j = 0; j < ny+1; ++j) {
    for (i = 0; i < nx; ++i) {
      for (k = 0; k < nz; ++k) {
	mf = MF_New(mesh);

	fverts[0] = verts[i][j][k];
	fverts[1] = verts[i+1][j][k];
	fverts[2] = verts[i+1][j][k+1];
	fverts[3] = verts[i][j][k+1];
	MF_Set_Vertices(mf,4,fverts);

	jj = (j%ny) ? 1 : (j ? 2 : 0);
	gdim = fgdim_tmpl[jj];
	gid = fgid_tmpl1[jj];

	MF_Set_GEntDim(mf,gdim);
	MF_Set_GEntID(mf,gid);
      }
    }
  }
	
  for (k = 0; k < nz+1; ++k) {
    for (i = 0; i < nx; ++i) {
      for (j = 0; j < ny; ++j) {
	mf = MF_New(mesh);

	fverts[0] = verts[i][j][k];
	fverts[1] = verts[i+1][j][k];
	fverts[2] = verts[i+1][j+1][k];
	fverts[3] = verts[i][j+1][k];
	MF_Set_Vertices(mf,4,fverts);

	kk = (k%nz) ? 1 : (k ? 2 : 0);
	gdim = fgdim_tmpl[kk];
	gid = fgid_tmpl2[kk];

	MF_Set_GEntDim(mf,gdim);
	MF_Set_GEntID(mf,gid);
      }
    }
  }
	

  /* Not the most efficient way but the easiest to code */

  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      for (k = 0; k < nz; ++k) {
	mr = MR_New(mesh);
	MR_Set_GEntID(mr,1);
	
	rverts[0] = verts[i][j][k];       rverts[1] = verts[i+1][j][k]; 
	rverts[2] = verts[i+1][j+1][k];   rverts[3] = verts[i][j+1][k];
	rverts[4] = verts[i][j][k+1];     rverts[5] = verts[i+1][j][k+1]; 
	rverts[6] = verts[i+1][j+1][k+1]; rverts[7] = verts[i][j+1][k+1];

	MR_Set_Vertices(mr, 8, rverts, 6, NULL);
      }
    }
  }
      
  for (i = 0; i < nx+1; ++i) {
    for (j = 0; j < ny+1; ++j)
      free(verts[i][j]);
    free(verts[i]);
  }
  free(verts);

  return 1;
}


int Mesh_MSTK::generate_regular_mesh(Mesh_ptr mesh, double x0, double y0, 
				     double x1, double y1, int nx, int ny)
{
  int i, j, dir[4];
  double xyz[3], llx, lly, urx, ury, dx, dy;
  MVertex_ptr **verts, v0, v1, mv;
  MEdge_ptr fedges[4], me;
  MFace_ptr mf;

  dx = (x1-x0)/nx;
  dy = (y1-y0)/ny;

  verts = (MVertex_ptr **) malloc((nx+1)*sizeof(MVertex_ptr *));
  for (i = 0; i < nx+1; ++i)
    verts[i] = (MVertex_ptr *) malloc((ny+1)*sizeof(MVertex_ptr));
 
  xyz[2] = 0.0;
  for (j = 0; j < ny+1; ++j) {
    xyz[1] = (j == ny) ? y1 : y0 + j*dy;

    for (i = 0; i < nx+1; ++i) {
      xyz[0] = (i == nx) ? x1 : x0 + i*dx;

      mv = MV_New(mesh);
      MV_Set_Coords(mv,xyz);

      if (i == 0) {
	if (j == 0) {
	  MV_Set_GEntDim(mv,0);
	  MV_Set_GEntID(mv,1);
	}
	else if (j == ny) {
	  MV_Set_GEntDim(mv,0);
	  MV_Set_GEntID(mv,4);	  
	}
	else {
	  MV_Set_GEntDim(mv,1);
	  MV_Set_GEntID(mv,4);
	}
      }
      else if (i == nx) {
	if (j == 0) {
	  MV_Set_GEntDim(mv,0);
	  MV_Set_GEntID(mv,2);
	}
	else if (j == ny) {
	  MV_Set_GEntDim(mv,0);
	  MV_Set_GEntID(mv,3);	  
	}
	else {
	  MV_Set_GEntDim(mv,1);
	  MV_Set_GEntID(mv,2);
	}
      }
      else {
	if (j == 0) {
	  MV_Set_GEntDim(mv,1);
	  MV_Set_GEntID(mv,1);
	}
	else if (j == ny) {
	  MV_Set_GEntDim(mv,1);
	  MV_Set_GEntID(mv,3);
	}
	else {
	  MV_Set_GEntDim(mv,2);
	  MV_Set_GEntID(mv,1);
	}
      }

      verts[i][j] = mv;
    }
  }


  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      mf = MF_New(mesh);
      
      /* edge 0 */
      v0 = verts[i][j];
      v1 = verts[i+1][j];
      fedges[0] = MVs_CommonEdge(v0,v1);
      if (fedges[0])
	dir[0] = (ME_Vertex(fedges[0],0) == v0) ? 1 : 0;
      else {
	me = ME_New(mesh);
	
	ME_Set_Vertex(me,0,v0);
	ME_Set_Vertex(me,1,v1);
	
	if (j == 0) {
	  ME_Set_GEntDim(me,1);
	  ME_Set_GEntID(me,1);
	}
	else {
	  ME_Set_GEntDim(me,2);
	  ME_Set_GEntID(me,1);
	}
	
	fedges[0] = me;
	dir[0] = 1;
      }
      
      
      /* edge 1 */
      v0 = verts[i+1][j];
      v1 = verts[i+1][j+1];
      fedges[1] = MVs_CommonEdge(v0,v1);
      if (fedges[1])
	dir[1] = (ME_Vertex(fedges[1],0) == v0) ? 1 : 0;
      else {
	me = ME_New(mesh);
	
	ME_Set_Vertex(me,0,v0);
	ME_Set_Vertex(me,1,v1);
	
	if (i+1 == nx) {
	  ME_Set_GEntDim(me,1);
	  ME_Set_GEntID(me,2);
	}
	else {
	  ME_Set_GEntDim(me,2);
	  ME_Set_GEntID(me,1);
	}
	
	fedges[1] = me;
	dir[1] = 1;
      }
      
      
      /* edge 2 */
      v0 = verts[i+1][j+1];
      v1 = verts[i][j+1];
      fedges[2] = MVs_CommonEdge(v0,v1);
      if (fedges[2])
	dir[2] = (ME_Vertex(fedges[2],0) == v0) ? 1 : 0;
      else {
	me = ME_New(mesh);
	
	ME_Set_Vertex(me,0,v0);
	ME_Set_Vertex(me,1,v1);
	
	if (j+1 == nx) {
	  ME_Set_GEntDim(me,1);
	  ME_Set_GEntID(me,3);
	}
	else {
	  ME_Set_GEntDim(me,2);
	  ME_Set_GEntID(me,1);
	}
	
	fedges[2] = me;
	dir[2] = 1;
      }
      
      
      /* edge 3 */
      v0 = verts[i][j+1];
      v1 = verts[i][j];
      fedges[3] = MVs_CommonEdge(v0,v1);
      if (fedges[3])
	dir[3] = (ME_Vertex(fedges[3],0) == v0) ? 1 : 0;
      else {
	me = ME_New(mesh);
	
	ME_Set_Vertex(me,0,v0);
	ME_Set_Vertex(me,1,v1);
	
	if (i == 0) {
	  ME_Set_GEntDim(me,1);
	  ME_Set_GEntID(me,4);
	}
	else {
	  ME_Set_GEntDim(me,2);
	  ME_Set_GEntID(me,1);
	}
	
	fedges[3] = me;
	dir[3] = 1;
      }


      MF_Set_Edges(mf,4,fedges,dir);

      MF_Set_GEntDim(mf,2);
      MF_Set_GEntID(mf,1);
    }
  }
   
  for (i = 0; i < nx+1; ++i)
    free(verts[i]);
  free(verts);

  return 1;
}

void Mesh_MSTK::pre_create_steps_(const int space_dimension, 
                                  const Epetra_MpiComm *comm, 
                                  const AmanziGeometry::GeometricModelPtr& gm) 
{
  clear_internals_();

  MSTK_Init();

  Mesh::set_comm(comm);
  Mesh::set_geometric_model(gm);

  mpicomm = comm->GetMpiComm();

  set_space_dimension(space_dimension);

  MPI_Comm_rank(mpicomm,&myprocid);
  MPI_Comm_size(mpicomm,&numprocs);

  serial_run =  (!mpicomm || numprocs == 1) ? true : false;

  parent_mesh = NULL;

  OwnedVerts = NotOwnedVerts = NULL;
  OwnedFaces = NotOwnedFaces = NULL;
  OwnedCells = GhostCells = NULL;
  deleted_vertices = deleted_edges = deleted_faces = deleted_regions = NULL;
  entities_deleted = false;
}


void Mesh_MSTK::inherit_labeled_sets(MAttrib_ptr copyatt) {

  int idx, idx2, diffdim;
  MSet_ptr mset;
  char setname[256];
  
  AmanziGeometry::GeometricModelPtr gm = Mesh::geometric_model();

  if (gm == NULL) { 
    std::cerr << "Need region definitions to initialize sets" << std::endl;
    return;
  }

  Mesh_ptr parent_mstk_mesh = parent_mesh->mesh;

  // Difference in cell dimension of this mesh and its parent
  // Labeled set entity dimensions will be similarly dialed down
  
  diffdim = parent_mesh->cell_dimension() - cell_dimension();
  if (diffdim > 1) {
    Errors::Message mesg("Dimension of mesh and its parent differ by more than 1");
    amanzi_throw(mesg);
  }
    
  unsigned int ngr = gm->Num_Regions();

  for (int i = 0; i < ngr; ++i) {
    AmanziGeometry::RegionPtr rgn = gm->Region_i(i);

    if (rgn->type() == AmanziGeometry::LABELEDSET) {

      // Get the set from the parent mesh

      AmanziGeometry::LabeledSetRegionPtr lsrgn =
        dynamic_cast<AmanziGeometry::LabeledSetRegionPtr> (rgn);

      std::string internal_name;
      std::string label = lsrgn->label();
      
      if (lsrgn->entity_str() == "CELL")
        internal_name = internal_name_of_set(rgn,CELL);
      else if (lsrgn->entity_str() == "FACE")
        internal_name = internal_name_of_set(rgn,FACE);
      else if (lsrgn->entity_str() == "NODE")
        internal_name = internal_name_of_set(rgn,NODE);


      MSet_ptr mset_parent = MESH_MSetByName(parent_mstk_mesh,internal_name.c_str());
      if (!mset_parent) {
        Errors::Message mesg("Cannot find labeled set in parent mesh");
        amanzi_throw(mesg);
      }

      // Create the set in this mesh

      MType subentdim;
      MType entdim = MSet_EntDim(mset_parent);
      if (entdim == MVERTEX)
        subentdim = MVERTEX;
      else
        subentdim = (MType) (entdim-diffdim);
      
      MSet_ptr mset = MSet_New(mesh,internal_name.c_str(),subentdim);


      // Populate the set

      int mkid = MSTK_GetMarker();

      MEntity_ptr ent;
      idx = 0;
      while ((ent = MSet_Next_Entry(mset_parent,&idx))) {

        MEntity_ptr copyent;
        int ival;
        double rval;

        if (subentdim == entdim) {
          MEnt_Get_AttVal(ent,copyatt,&ival,&rval,&copyent);
          if (!copyent) continue;
          
          MSet_Add(mset,copyent);
        }
        else {
          if (entdim == MREGION) {
            MFace_ptr rf;
            List_ptr rfaces = MR_Faces((MRegion_ptr)ent);
            idx2 = 0;
            while ((rf = List_Next_Entry(rfaces,&idx2))) {
              MEnt_Get_AttVal(rf,copyatt,&ival,&rval,&copyent);
              if (!copyent) continue;

              if (!MEnt_IsMarked(copyent,mkid)) {
                MSet_Add(mset,copyent);
                MEnt_Mark(copyent,mkid);
              }
            }
            List_Delete(rfaces);
          }
          else if (entdim == MFACE) {
            MEdge_ptr fe;
            List_ptr fedges = MF_Edges((MFace_ptr)ent,1,0);
            idx2 = 0;
            while ((fe = List_Next_Entry(fedges,&idx2))) {
              MEnt_Get_AttVal(fe,copyatt,&ival,&rval,&copyent);
              if (!copyent) continue;

              if (!MEnt_IsMarked(copyent,mkid)) {
                MSet_Add(mset,copyent);
                MEnt_Mark(copyent,mkid);
              }
            }
            List_Delete(fedges);
          }
        }

      }

      MSet_Unmark(mset,mkid);
      MSTK_FreeMarker(mkid);
        
    }
  }

} // inherit_labeled_sets



// Write mesh out to exodus file

void
Mesh_MSTK::write_to_exodus_file(const std::string filename) const {
  MESH_ExportToExodusII(mesh,filename.c_str(),-1,NULL,NULL,mpicomm);
}


} // close namespace AmanziMesh
} // close namespace Amanzi
