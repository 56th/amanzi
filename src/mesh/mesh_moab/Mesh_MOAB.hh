#ifndef _MESH_MOAB_H_
#define _MESH_MOAB_H_

// MOAB include files - for some reason they have to be first -
// if not the compiler cannot find MPI_COMM_WORLD

#include "MBmpi.h"
#include "MBTypes.h"
#include "MBCore.hpp"
#include "MBRange.hpp"
#include "MBTagConventions.hpp"
#include "MBReadUtilIface.hpp"
#include "MBParallelConventions.h"
#include "MBParallelComm.hpp"

#include <Epetra_Map.h>
#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>

#include "Mesh.hh"
#include "GeometricModel.hh"

#include <memory>
#include <vector>

namespace Amanzi
{
namespace AmanziMesh
{

class Mesh_MOAB : public Mesh
{

  private:

    MBCore *mbcore;
    MBParallelComm *mbcomm;

    int serial_run;

    int spacedim;
    int celldim; // Topological dimension of highest level entities
    int facedim; // Topological dimension of 2nd highest level entities



    // Local handles to entity lists (Vertices, "Faces", "Cells")

    // For a surface mesh, "Faces" refers to mesh edges and "Cells"
    // refers to mesh faces
    //
    // For a solid mesh, "Faces" refers to mesh faces and "Cells"
    // refers to mesh regions
    

    // These are MOAB's definitions of types of parallel mesh entities
    // These definitions are slightly different from what Amanzi has defined
    //
    // There are 2 types of cells - Owned and Ghost

    // There are 4 types of lower dimensional entities 
    //
    // 1. OWNED - owned by this processor
    //
    // 2. NOTOWNED - not owned by this processor
    //
    // 3. USED - connected to at least one cell owned by this
    // processor (may or may not be owned by this processor)
    //
    // 4. GHOST - neither the entity nor a cell connected to the
    // entity is owned by the processor

    // UNFORTUNATELY, THE TERMINOLOGY USED BY THE API USES GHOST TO
    // MEAN NOTOWNED

    // ALL = OWNED + NOTOWNED or USED + GHOST


    MBRange AllVerts, OwnedVerts, NotOwnedVerts;

    MBRange AllFaces, OwnedFaces, NotOwnedFaces;

    MBRange AllCells, OwnedCells, GhostCells;


    // tag handles

    MBTag lid_tag;        // Local ID
    MBTag gid_tag;        // Global ID
    MBTag mattag;         // Material tag
    MBTag sstag;          // Sideset tag
    MBTag nstag;          // Nodeset tag


    // Local ID to MOAB handle map

    std::vector<MBEntityHandle> vtx_id_to_handle;
    std::vector<MBEntityHandle> face_id_to_handle;
    std::vector<MBEntityHandle> cell_id_to_handle;


    // Maps

    Epetra_Map *cell_map_wo_ghosts_, *face_map_wo_ghosts_, *node_map_wo_ghosts_;
    Epetra_Map *cell_map_w_ghosts_, *face_map_w_ghosts_, *node_map_w_ghosts_;


    // Sets (material sets, sidesets, nodesets)
    // We store the number of sets in the whole problem regardless of whether
    // they are represented on this processor or not
    // We also store the IDs of the sets and the dimension of entities 
    // in those sets
  
    int nsets;
    int *setids_, *setdims_;


    // Minimum and maximum global IDs of faces

    unsigned int minFGID, maxFGID;


    // flag whether to flip a face dir or not when returning nodes of a face

    bool *faceflip;
    
    // Private methods
    // ----------------------------

    bool valid_entity_kind_ (int kind) const;


    void clear_internals_();

    void init_pvert_lists();
    void init_pface_lists();
    void init_pcell_lists();
    void init_pface_dirs();

    void init_id_handle_maps();
    void init_global_ids();

    void init_cell_map();
    void init_face_map();
    void init_node_map();

    void init_set_info();

public:
  
  Mesh_MOAB (const char *filename, const Epetra_MpiComm *comm, 
	     const AmanziGeometry::GeometricModelPtr& gm = 
	     (AmanziGeometry::GeometricModelPtr) NULL);

  // Construct a mesh by extracting a subset of entities from another
  // mesh. In some cases like extracting a surface mesh from a volume
  // mesh, constructor can be asked to "flatten" the mesh to a lower
  // dimensional space or to extrude the mesh to give higher
  // dimensional cells

  Mesh_MOAB(const Mesh *inmesh,
            const std::vector<std::string>& setnames,
            const Entity_kind setkind,
            const bool flatten = false,
            const bool extrude = false);

  Mesh_MOAB(const Mesh_MOAB& inmesh,
            const std::vector<std::string>& setnames,
            const Entity_kind setkind,
            const bool flatten = false,
            const bool extrude = false);

  ~Mesh_MOAB();
  
  void update ();
  
  // Get parallel type of entity
    
  Parallel_type entity_get_ptype(const Entity_kind kind, 
				 const Entity_ID entid) const;


  // Get cell type
    
  Cell_type cell_get_type(const Entity_ID cellid) const;
        
   
  //
  // General mesh information
  // -------------------------
  //
    
  // Number of entities of any kind (cell, face, node) and in a
  // particular category (OWNED, GHOST, USED)
    
  unsigned int num_entities (const Entity_kind kind,
			     const Parallel_type ptype) const;
    
    
  // Global ID of any entity
    
  Entity_ID GID(const Entity_ID lid, const Entity_kind kind) const;
    
    
    
  //
  // Mesh Entity Adjacencies 
  //-------------------------


  // Downward Adjacencies
  //---------------------
    
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

  virtual
  void cell_get_faces_and_dirs (const Entity_ID cellid,
                                Entity_ID_List *faceids,
                                std::vector<int> *face_dirs,
				const bool ordered=false) const;

    
  // Get nodes of cell 
  // On a distributed mesh, all nodes (OWNED or GHOST) of the cell 
  // are returned
  // Nodes are returned in a standard order (Exodus II convention)
  // STANDARD CONVENTION WORKS ONLY FOR STANDARD CELL TYPES in 3D
  // For a general polyhedron this will return the nodes in
  // arbitrary order
  // In 2D, the nodes of the polygon will be returned in ccw order 
  // consistent with the face normal
    
  void cell_get_nodes (const Entity_ID cellid, 
		       Entity_ID_List *nodeids) const;
    
    
  // Get nodes of face 
  // On a distributed mesh, all nodes (OWNED or GHOST) of the face 
  // are returned
  // In 3D, the nodes of the face are returned in ccw order consistent
  // with the face normal
  // In 2D, nfnodes is 2
    
  void face_get_nodes (const Entity_ID faceid, 
		       Entity_ID_List *nodeids) const;
    


  // Upward adjacencies
  //-------------------
    
  // Cells of type 'ptype' connected to a node
    
  void node_get_cells (const Entity_ID nodeid, 
		       const Parallel_type ptype,
		       Entity_ID_List *cellids) const;
    
  // Faces of type 'ptype' connected to a node
    
  void node_get_faces (const Entity_ID nodeid, 
		       const Parallel_type ptype,
		       Entity_ID_List *faceids) const;
    
  // Get faces of ptype of a particular cell that are connected to the
  // given node
    
  void node_get_cell_faces (const Entity_ID nodeid, 
			    const Entity_ID cellid,
			    const Parallel_type ptype,
			    Entity_ID_List *faceids) const;    
    
  // Cells connected to a face
    
  void face_get_cells (const Entity_ID faceid, 
		       const Parallel_type ptype,
		       Entity_ID_List *cellids) const;
    


  // Same level adjacencies
  //-----------------------

  // Face connected neighboring cells of given cell of a particular ptype
  // (e.g. a hex has 6 face neighbors)

  // The order in which the cellids are returned cannot be
  // guaranteed in general except when ptype = USED, in which case
  // the cellids will correcpond to cells across the respective
  // faces given by cell_get_faces

  void cell_get_face_adj_cells(const Entity_ID cellid,
			       const Parallel_type ptype,
			       Entity_ID_List *fadj_cellids) const;

  // Node connected neighboring cells of given cell
  // (a hex in a structured mesh has 26 node connected neighbors)
  // The cells are returned in no particular order

  void cell_get_node_adj_cells(const Entity_ID cellid,
			       const Parallel_type ptype,
			       Entity_ID_List *nadj_cellids) const;

    
  //
  // Mesh entity geometry
  //--------------
  //
    
  // Node coordinates - 3 in 3D and 2 in 2D
    
  void node_get_coordinates (const Entity_ID nodeid, 
			     AmanziGeometry::Point *ncoord) const;
    
    
  // Face coordinates - conventions same as face_to_nodes call 
  // Number of nodes is the vector size divided by number of spatial dimensions
    
  void face_get_coordinates (const Entity_ID faceid, 
			     std::vector<AmanziGeometry::Point> *fcoords) const; 
    
  // Coordinates of cells in standard order (Exodus II convention)
  // STANDARD CONVENTION WORKS ONLY FOR STANDARD CELL TYPES IN 3D
  // For a general polyhedron this will return the node coordinates in
  // arbitrary order
  // Number of nodes is vector size divided by number of spatial dimensions
    
  void cell_get_coordinates (const Entity_ID cellid, 
			     std::vector<AmanziGeometry::Point> *ccoords) const;
    
  // Modify the coordinates of a node

  void node_set_coordinates (const Entity_ID nodeid, const AmanziGeometry::Point coords);

  void node_set_coordinates (const Entity_ID nodeid, const double *coords);

    
  //
  // Epetra maps
  //------------
    
    
  const Epetra_Map& cell_map (bool include_ghost) const;
    
  const Epetra_Map& face_map (bool include_ghost) const; 

  const Epetra_Map& node_map (bool include_ghost) const;
    
  const Epetra_Map& exterior_face_map (void) const; 
    
  // Epetra importer that will allow apps to import values from a
  // Epetra vector defined on all owned faces into an Epetra vector
  // defined only on exterior faces
  
  const Epetra_Import& exterior_face_importer (void) const;
    
    
    
  //
  // Boundary Conditions or Sets
  //----------------------------
  //
    
  // Get number of entities of type 'category' in set

  unsigned int get_set_size (const Set_ID setid, 
			     const Entity_kind kind,
			     const Parallel_type ptype) const;

  unsigned int get_set_size (const Set_Name setname, 
			     const Entity_kind kind,
			     const Parallel_type ptype) const;

  unsigned int get_set_size (const char *setname, 
			     const Entity_kind kind,
			     const Parallel_type ptype) const;


  // Get list of entities of type 'category' in set

  void get_set_entities (const Set_ID setid, 
			 const Entity_kind kind, 
			 const Parallel_type ptype, 
			 Entity_ID_List *entids) const;   

  void get_set_entities (const Set_Name setname, 
			 const Entity_kind kind, 
			 const Parallel_type ptype, 
			 std::vector<Entity_ID> *entids) const; 

  void get_set_entities (const char *setname, 
			 const Entity_kind kind, 
			 const Parallel_type ptype, 
			 std::vector<Entity_ID> *entids) const; 

  // Deform a mesh so that cell volumes conform as closely as possible
  // to target volumes without dropping below the minimum volumes.  If
  // move_vertical = true, nodes will be allowed to move only in the
  // vertical direction (right now arbitrary node movement is not allowed)

  int deform(const std::vector<double>& target_cell_volumes_in, 
             const std::vector<double>& min_cell_volumes_in, 
             const std::vector<std::string>& fixed_set_names,
             const bool move_vertical);

 
 // Miscellaneous
  
  void write_to_exodus_file (const std::string filename) const;

};

} // close namespace AmanziMesh
} // close namespace Amanzi

#endif /* _MESH_MAPS_MOAB_H_ */
