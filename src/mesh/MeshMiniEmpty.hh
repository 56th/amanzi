/*
  Author: Alexander Zhiliakov alex@math.uh.edu
  Abstract mini-mesh interface
  This is used by ASC(n) method
  This class implements a "mini-mesh" w/ no multi-material cells
*/

#ifndef AMANZI_MESH_MINI_EMPTY_HH_
#define AMANZI_MESH_MINI_EMPTY_HH_

#include "MeshMini.hh"

namespace Amanzi {
    namespace AmanziMesh {
        class MeshMiniEmpty : public MeshMini {
        public:
            MeshMiniEmpty(Teuchos::RCP<const Mesh> const & mesh) : MeshMini(mesh) {}
            size_t numbOfFaces(size_t C) const final {
                Entity_ID_List macroFacesIndicies;
                mesh_->cell_get_faces(C, &macroFacesIndicies);
                return macroFacesIndicies.size();
            }
            size_t numbOfExtFaces(size_t C) const final {
                return numbOfFaces(C);
            }
            size_t numbOfMaterials(size_t C) const final {
                return 1;
            }
            AmanziGeometry::Point centroid(size_t C, size_t c) const final {
                if (c != 0) throw std::invalid_argument("__function__: numb of mini-cells must be 1");
                return mesh_->cell_centroid(C);
            }
            size_t materialIndex(size_t C, size_t c) const final {
                if (c != 0) throw std::invalid_argument("__function__: numb of mini-cells must be 1");
                return 0;
            }
            double volume(size_t C, size_t c) const final {
                if (c != 0) throw std::invalid_argument("__function__: numb of mini-cells must be 1");
                return mesh_->cell_volume(C);
            }
            Entity_ID_List facesGlobalIndicies(size_t C, size_t c) const final {
                if (c != 0) throw std::invalid_argument("__function__: numb of mini-cells must be 1");
                Entity_ID_List res(numbOfFaces(C));
                for (size_t i = 0; i < res.size(); ++i) res[i] = i;
                return res;
            }    
            AmanziGeometry::Point faceCentroid(size_t C, size_t g) const final {
                Entity_ID_List macroFacesIndicies;
                mesh_->cell_get_faces(C, &macroFacesIndicies);
                return mesh_->face_centroid(macroFacesIndicies[g]);
            }
            virtual double area(size_t C, size_t g) const final {
                Entity_ID_List macroFacesIndicies;
                mesh_->cell_get_faces(C, &macroFacesIndicies);
                return mesh_->face_area(macroFacesIndicies[g]);
            }
            AmanziGeometry::Point normal(size_t C, size_t g) const final {
                AmanziMesh::Entity_ID_List macroFacesIndicies;
                std::vector<int> macroFacesNormalsDirs;
                mesh_->cell_get_faces_and_dirs(C, &macroFacesIndicies, &macroFacesNormalsDirs);
                auto n = macroFacesNormalsDirs[g] * mesh_->face_normal(macroFacesIndicies[g]);
                return n / AmanziGeometry::norm(n);
            }
            size_t parentFaceLocalIndex(size_t C, size_t g) const final {
                return g;
            }
        };
    }
}

#endif

