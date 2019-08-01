/*
  Author: Alexander Zhiliakov alex@math.uh.edu
  Abstract mini-mesh interface
  This is used by ASC(n) method
*/

#ifndef AMANZI_MESH_MINI_TANGRAM_HH_
#define AMANZI_MESH_MINI_TANGRAM_HH_

#include "MeshMini.hh"
#include "tangram/driver/CellMatPoly.h"

namespace Amanzi {
    namespace AmanziMesh {
        class MeshMiniTangram : public MeshMini {
        private:
            std::vector<std::shared_ptr<Tangram::CellMatPoly<3>>> polyCells_;
        public:
            MeshMiniTangram(
                Teuchos::RCP<const Mesh> const & mesh, 
                std::vector<std::shared_ptr<Tangram::CellMatPoly<3>>> const & polyCells
            ) 
            : MeshMini(mesh)
            , polyCells_(polyCells) {}
            size_t numbOfFaces(size_t C) const final {
                return polyCells_[C]->num_matfaces();
            }
            size_t numbOfExtFaces(size_t C) const final {
                return numbOfFaces(C);
            }
            size_t numbOfMaterials(size_t C) const final {
                return polyCells_[C]->num_matpolys();
            }
            AmanziGeometry::Point centroid(size_t C, size_t c) const final {
                auto a = polyCells_[C]->matpoly_centroid(c);
                return AmanziGeometry::Point(a[0], a[1], a[2]);
            }
            size_t materialIndex(size_t C, size_t c) const final {
                return polyCells_[C]->matpoly_matid(c);
            }
            double volume(size_t C, size_t c) const final {
                return polyCells_[C]->matpoly_volume(c);
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
                auto f = macroFacesIndicies[g];
                return macroFacesNormalsDirs[g] * mesh_->face_normal(f);
            }
            size_t parentFaceLocalIndex(size_t C, size_t g) const final {
                return g;
            }
        };
    }
}

#endif

