/*
  Author: Alexander Zhiliakov alex@math.uh.edu
  Abstract mini-mesh interface
  This is used by ASC(n) method
*/

#ifndef AMANZI_MESH_MINI_HH_
#define AMANZI_MESH_MINI_HH_

#include <unordered_set>
#include "Mesh.hh"

namespace Amanzi {
    namespace AmanziMesh {
        class MeshMini {
        protected:
            Teuchos::RCP<const Mesh> mesh_;
            int sgn_(double val) const {
                return (0. < val) - (val < 0.);
            }
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
            virtual size_t numbOfMaterials(size_t C) const = 0; // in a macro-cell
            size_t numbOfMaterials() const { // in a mesh
                std::unordered_set<size_t> matIDs;
                auto n = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::OWNED);
                for (size_t i = 0; i < n; ++i)
                    for (size_t j = 0; j < numbOfMaterials(i); ++j)
                        matIDs.insert(materialIndex(i, j));
                return matIDs.size();
            } 
            size_t maxNumbOfMaterials() const { // over all macro-cells
                auto n = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::OWNED);
                size_t max = 0;
                for (size_t i = 0; i < n; ++i)
                    max = std::max(max, numbOfMaterials(i));
                return max;
            }
            virtual AmanziGeometry::Point centroid(size_t C, size_t c) const = 0;
            virtual size_t materialIndex(size_t C, size_t c) const = 0;
            virtual double volume(size_t C, size_t c) const = 0;
            virtual Entity_ID_List facesGlobalIndicies(size_t C, size_t c) const = 0;     
            virtual AmanziGeometry::Point faceCentroid(size_t C, size_t g) const = 0;
            virtual double area(size_t C, size_t g) const = 0;
            virtual AmanziGeometry::Point normal(size_t C, size_t g) const = 0;
            // +1 if outwards, -1 if inwards
            int normalSign(size_t C, size_t c, size_t g) const {
                auto n = normal(C, g);
                auto u = centroid(C, c) - faceCentroid(C, g);
                auto s = -sgn_(n * u);
                if (s == 0) {
                    std::stringstream err;
                    err << __func__ << ": cell #" << C << ": mini-face #" << g << " has invalid normal (maybe the cell is not convex)";
                    throw std::invalid_argument(err.str());
                }
                return s;
            }
            virtual int parentFaceLocalIndex(size_t C, size_t g) const = 0;
            Entity_ID_List childrenFacesGlobalIndicies(size_t C, size_t F) const {
                Entity_ID_List res;
                auto n = numbOfExtFaces(C);
                for (size_t i = 0; i < n; ++i)
                    if (parentFaceLocalIndex(C, i) == F)
                        res.push_back(i);
                return res;
            }
            Entity_ID_List macroFacesIndicies(size_t C) const {
                Entity_ID_List macroFacesIndicies;
                mesh_->cell_get_faces(C, &macroFacesIndicies);
                return macroFacesIndicies;
            }
            std::vector<int> macroFacesNormalsDirs(size_t C) const {
                Entity_ID_List macroFacesIndicies;
                std::vector<int> macroFacesNormalsDirs;
                mesh_->cell_get_faces_and_dirs(C, &macroFacesIndicies, &macroFacesNormalsDirs);
                return macroFacesNormalsDirs;
            }
            size_t faceMaterialIndex(size_t C, size_t g) const {
                std::string err = __func__;
                if (parentFaceLocalIndex(C, g) < 0) {
                    throw std::invalid_argument(err + ": face is an interface face");
                }
                for (size_t c = 0; c < numbOfMaterials(C); ++c) {
                    auto ind = facesGlobalIndicies(C, c);
                    if (std::find(ind.begin(), ind.end(), g) != ind.end())
                        return materialIndex(C, c);
                }
                throw std::invalid_argument(err + ": face material is not found");
            }
        };
    }
}

#endif

