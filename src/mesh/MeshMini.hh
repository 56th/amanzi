/*
  Author: Alexander Zhiliakov alex@math.uh.edu
  Abstract mini-mesh interface
  This is used by ASC(n) method
*/

#ifndef AMANZI_MESH_MINI_HH_
#define AMANZI_MESH_MINI_HH_

#include "Mesh.hh"

namespace Amanzi {
    namespace AmanziMesh {
        class MeshMini {
        protected:
            Teuchos::RCP<const Mesh> mesh_;
        public:
            MeshMini(Teuchos::RCP<const Mesh> const & mesh) : mesh_(mesh) {}
            virtual ~MeshMini() = default;
            Teuchos::RCP<const Mesh> macroMesh() const {
                return mesh_;
            }
            // C := global index of macro-cell, c := global index of mini-cell w.r.t. mini-mesh
            // F := local index of macro-face of the cell C, g := global index of mini-face
            // Assumptions:
            // * external mini-faces are enumerated first: 0, 1, ..., numbOfExtFaces() - 1
            //   and internal mini-faces have indicies numbOfExtFaces(), numbOfExtFaces() + 1, ..., numbOfFaces() - 1
            // * normals of external mini-faces point outside
            virtual size_t numbOfFaces(size_t C) const = 0;
            virtual size_t numbOfExtFaces(size_t C) const = 0;
            virtual size_t numbOfMaterials(size_t C) const = 0;
            virtual AmanziGeometry::Point centroid(size_t C, size_t c) const = 0;
            virtual size_t materialIndex(size_t C, size_t c) const = 0;
            virtual double volume(size_t C, size_t c) const = 0;
            virtual Entity_ID_List facesGlobalIndicies(size_t C, size_t c) const = 0;     
            virtual AmanziGeometry::Point faceCentroid(size_t C, size_t g) const = 0;
            virtual double area(size_t C, size_t g) const = 0;
            virtual AmanziGeometry::Point normal(size_t C, size_t g) const = 0;
            virtual size_t parentFaceLocalIndex(size_t C, size_t g) const = 0;
            Entity_ID_List childrenFacesGlobalIndicies(size_t C, size_t F) const {
                Entity_ID_List res;
                auto n = numbOfExtFaces(C);
                for (size_t i = 0; i < n; ++i)
                    if (parentFaceLocalIndex(C, i) == F)
                        res.push_back(i);
                return res;
            }
        };
    }
}

#endif

