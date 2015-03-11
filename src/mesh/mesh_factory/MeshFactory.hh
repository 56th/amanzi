/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */
// -------------------------------------------------------------
/**
 * @file   MeshFactory.hh
 * @author William A. Perkins
 * @date Wed Sep 28 09:10:15 2011
 * 
 * @brief  declaration of the MeshFactory class
 * 
 * 
 */
// -------------------------------------------------------------
// -------------------------------------------------------------
// Created March 10, 2011 by William A. Perkins
// Last Change: Wed Sep 28 09:10:15 2011 by William A. Perkins <d3g096@PE10900.pnl.gov>
// -------------------------------------------------------------

#ifndef _MeshFactory_hh_
#define _MeshFactory_hh_

#include <string>
#include <vector>
#include <Epetra_MpiComm.h>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ParameterList.hpp>

#include "MeshException.hh"
#include "MeshFramework.hh"
#include "Mesh.hh"

#include "GeometricModel.hh"
#include "VerboseObject.hh"

namespace Amanzi {
namespace AmanziMesh {

// -------------------------------------------------------------
//  class MeshFactory
// -------------------------------------------------------------
class MeshFactory {
 protected:

  /// The parallel environment
  const Epetra_MpiComm *my_comm;

  /// A list of preferred mesh frameworks to consider
  FrameworkPreference my_preference;

 private:

  /// private, undefined copy constructor to avoid unwanted copies
  MeshFactory(MeshFactory& old);

  /// Object encoding the level of verbosity and output stream for
  /// diagnostic messages

  const VerboseObject *verbosity_obj;

 public:

  /// Default constructor.
  explicit MeshFactory(const Epetra_MpiComm *communicator, 
                       const VerboseObject *meshverbobj = NULL);

  /// Destructor
  ~MeshFactory(void);

  /// Get the framework preference
  const FrameworkPreference& preference(void) const
  { return my_preference; }

  /// Set the framework preference
  void preference(const FrameworkPreference& pref);

  /// Create a mesh by reading the specified file (or set of files)
  Teuchos::RCP<Mesh> create(const std::string& filename, 
                            const AmanziGeometry::GeometricModelPtr &gm = 
                            (AmanziGeometry::GeometricModelPtr) NULL,
                            const bool request_faces = true,
                            const bool request_edges = false);


  /// Create a hexahedral mesh of the specified dimensions
  Teuchos::RCP<Mesh> create(double x0, double y0, double z0,
                            double x1, double y1, double z1,
                            int nx, int ny, int nz, 
                            const AmanziGeometry::GeometricModelPtr &gm = 
                            (AmanziGeometry::GeometricModelPtr) NULL,
                            const bool request_faces = true,
                            const bool request_edges = false);

    
  /// Create a quadrilateral mesh of the specified dimensions
  Teuchos::RCP<Mesh> create(double x0, double y0,
                            double x1, double y1,
                            int nx, int ny,
                            const AmanziGeometry::GeometricModelPtr &gm = 
                            (AmanziGeometry::GeometricModelPtr) NULL,
                            const bool request_faces = true,
                            const bool request_edges = false);

    
  /// Create a quadrilateral/hexahedral mesh using the specified parameter list
  Teuchos::RCP<Mesh> create(Teuchos::ParameterList &parameter_list, 
                            const AmanziGeometry::GeometricModelPtr &gm = 
                            (AmanziGeometry::GeometricModelPtr) NULL,
                            const bool request_faces = true,
                            const bool request_edges = false);

  /// Create a mesh by extract subsets of entities from an existing mesh
  Teuchos::RCP<Mesh> create(const Mesh *inmesh,
                            const std::vector<std::string> setnames,
                            const Entity_kind setkind,
                            const bool flatten = false,
                            const bool extrude = false,
                            const bool request_faces = true,
                            const bool request_edges = false);


  /// Create a mesh by reading the specified file (or set of files) -- operator
  Teuchos::RCP<Mesh> operator() (const std::string& filename, 
                                 const AmanziGeometry::GeometricModelPtr &gm = 
                                 (AmanziGeometry::GeometricModelPtr) NULL,
                                 const bool request_faces = true,
                                 const bool request_edges = false) {

    return create(filename, gm, request_faces, request_edges);
  }
  
  /// Create a hexahedral mesh of the specified dimensions -- operator
  Teuchos::RCP<Mesh> operator() (double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 int nx, int ny, int nz, 
                                 const AmanziGeometry::GeometricModelPtr &gm = 
                                 (AmanziGeometry::GeometricModelPtr) NULL,
                                 const bool request_faces = true,
                                 const bool request_edges = false) {

    return create(x0, y0, z0, x1, y1, z1, nx, ny, nz, gm, 
                  request_faces, request_edges);
  }

  /// Create a quadrilateral mesh of the specified dimensions -- operator
  Teuchos::RCP<Mesh> operator() (double x0, double y0,
                                 double x1, double y1,
                                 int nx, int ny,
                                 const AmanziGeometry::GeometricModelPtr &gm = 
                                 (AmanziGeometry::GeometricModelPtr) NULL,
                                 const bool request_faces = true,
                                 const bool request_edges = false)  {
 
    return create(x0, y0, x1, y1, nx, ny, gm, request_faces, request_edges);
  }

  /// Create a quadrilateral/hexahedral mesh using the specified parameter list
  Teuchos::RCP<Mesh> operator() (Teuchos::ParameterList &parameter_list, 
                                 const AmanziGeometry::GeometricModelPtr &gm = 
                                 (AmanziGeometry::GeometricModelPtr) NULL,
                                 const bool request_faces = true,
                                 const bool request_edges = false) {

    return create(parameter_list, gm, request_faces, request_edges);
  }

  /// Create a mesh by extract subsets of entities from an existing mesh
  Teuchos::RCP<Mesh> operator() (const Mesh *inmesh,
                                 const std::vector<std::string> setnames,
                                 const Entity_kind setkind,
                                 const bool flatten = false,
                                 const bool extrude = false,
                                 const bool request_faces = true,
                                 const bool request_edges = false) {

    return create(inmesh, setnames, setkind, flatten, extrude,
                  request_faces, request_edges);
  }

};

} // namespace AmanziMesh
} // namespace Amanzi

#endif
