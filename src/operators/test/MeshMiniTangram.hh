/*
  Author: Alexander Zhiliakov alex@math.uh.edu
  Abstract mini-mesh interface
  This is used by ASC(n) method
*/

#ifndef AMANZI_MESH_MINI_TANGRAM_HH_
#define AMANZI_MESH_MINI_TANGRAM_HH_

#include <unordered_map>
#include "MeshMini.hh"
#include "MeshMiniEmpty.hh"
#include "tangram/driver/CellMatPoly.h"
#include "SingletonLogger.hpp"

namespace Amanzi {
    namespace AmanziMesh {
        // helper
        template<typename T>
        std::ostream& operator<<(std::ostream& out, std::vector<T> const & vec) {
            if (vec.size()) {
                for (size_t i = 0; i < vec.size() - 1; ++i) out << vec[i] << ' ';
                out << vec.back();
            }
            return out;
        }
        class MeshMiniTangram : public MeshMini {
        private:
            std::vector<std::shared_ptr<Tangram::CellMatPoly<3>>> polyCells_;
            std::vector<std::unordered_map<size_t, size_t>> faceRenum_;
            std::vector<size_t> numbOfExtFaces_;
            std::vector<Tangram::Point<3>> vertices_(size_t C) const {
                std::vector<Tangram::Point<3>> pts;
                auto nv = polyCells_[C]->num_matvertices();
                pts.reserve(nv);
                for (size_t i = 0; i < nv; ++i)
                    pts.push_back(polyCells_[C]->matvertex_point(i));
                // auto& logger = SingletonLogger::instance();
                // for (auto p : pts)
                //     logger.buf << "{ " << p << " }, ";    
                // logger.buf << '\n' << polyCells_[C]->matface_vertices(g);
                // logger.log();
                return pts;
            }
            bool pointProjectionLiesInTriangle_(
                AmanziGeometry::Point const & p,
                AmanziGeometry::Point const & t1,
                AmanziGeometry::Point const & t2,
                AmanziGeometry::Point const & t3
            ) const {
                return true;
            }
            bool fpEqual_(double a, double b, double tol = 1e-8) const {
                    return fabs(a - b) < tol;
            }
        public:
            MeshMiniTangram(
                Teuchos::RCP<const Mesh> const & mesh, 
                std::vector<std::shared_ptr<Tangram::CellMatPoly<3>>> const & polyCells,
                bool check = false
            ) 
            : MeshMini(mesh)
            , polyCells_(polyCells) {
                auto& logger = SingletonLogger::instance();
                auto n = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::OWNED);
                faceRenum_.resize(n);
                numbOfExtFaces_.resize(n);
                for (size_t C = 0; C < n; ++C) {
                    std::unordered_map<size_t, size_t> freq;
                    for (size_t c = 0; c < numbOfMaterials(C); ++c)
                        for (auto const & i : polyCells_[C]->matpoly_faces(c)) ++freq[i];
                    std::vector<size_t> fInt;
                    for (auto const & f : freq) {
                        faceRenum_[C][f.first] = f.first;
                        if (f.second > 1) fInt.push_back(f.first);
                    }
                    for (size_t i = 0; i < fInt.size(); ++i) std::swap(faceRenum_[C][fInt[i]], faceRenum_[C][faceRenum_[C].size() - 1 - i]);
                    numbOfExtFaces_[C] = polyCells_[C]->num_matfaces() - fInt.size();
                }
                if (!check) return;
                MeshMiniEmpty x(mesh);
                logger.beg("check mini-mesh");
                    for (size_t C = 0; C < n; ++C) {
                        if (numbOfMaterials(C) == 1) {
                            // global mini-faces indicies consistency
                            auto f1 = facesGlobalIndicies(C, 0);
                            auto f2 = x.facesGlobalIndicies(C, 0);
                            if (f1 != f2) 
                                logger.buf << "SMC #" << C << ": faces global indicies {" << f1 << "} != {" << f2 << "}\n";
                            // face areas, centroids, and normals
                            for (auto g : f1) {
                                auto diff = fabs(area(C, g) - x.area(C, g));
                                if (!fpEqual_(diff, 0.))
                                    logger.buf << "SMC #" << C << ": face #" << g << " area     diff = " << diff << ",\t" << area(C, g) << " vs. " << x.area(C, g) << '\n';
                                auto c1 = faceCentroid(C, g);
                                auto c2 = x.faceCentroid(C, g);
                                diff = AmanziGeometry::norm(c1 - c2);
                                if (!fpEqual_(diff, 0.))
                                    logger.buf << "SMC #" << C << ": face #" << g << " centroid diff = " << diff << ",\t" << c1 << " vs. " << c2 << '\n';
                                auto n1 = normal(C, g);
                                auto n2 = x.normal(C, g);
                                diff = AmanziGeometry::norm(n1 - n2);
                                if (!fpEqual_(diff, 0.))
                                    logger.buf << "SMC #" << C << ": face #" << g << " normal   diff = " << diff << ",\t" << n1 << " vs. " << n2 << '\n';
                            }
                            // centroids
                            auto c1 = centroid(C, 0);
                            auto c2 = x.centroid(C, 0);
                            if (!fpEqual_(AmanziGeometry::norm(c1 - c2), 0.))
                                logger.buf << "SMC #" << C << ": centroids err = " << AmanziGeometry::norm(c1 - c2) << '\n';
                            // volumes
                            auto v1 = volume(C, 0);
                            auto v2 = x.volume(C, 0);
                            if (!fpEqual_(v1 - v2, 0.))
                                logger.buf << "SMC #" << C << ": volume err = " << fabs(v1 - v2) << '\n';
                            if (logger.buf.tellp() != std::streampos(0)) logger.wrn();
                        }
                        else for (size_t c = 0; c < numbOfMaterials(C); ++c) {
                            // auto fInd = facesGlobalIndicies(C, c);
                            // if (numbOfMaterials(C) == 3) logger.buf << "MMC #" << C << "." << c << ": faces global indicies {" << fInd << "}\n";
                            // logger.log();
                        }
                    }
                logger.end();
            }
            size_t numbOfFaces(size_t C) const final {
                return polyCells_[C]->num_matfaces();
            }
            size_t numbOfExtFaces(size_t C) const final {
                return numbOfExtFaces_[C];
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
                auto a = polyCells_[C]->matpoly_faces(c);
                for (auto& i : a) i = faceRenum_[C].at(i);
                return a;
            }    
            AmanziGeometry::Point faceCentroid(size_t C, size_t g) const final {
                g = faceRenum_[C].at(g);
                std::vector<double> a;
                Tangram::polygon3d_moments(vertices_(C), polyCells_[C]->matface_vertices(g), a);
                return AmanziGeometry::Point(a[1] / a[0], a[2] / a[0], a[3] / a[0]);
            }
            double area(size_t C, size_t g) const final {
                g = faceRenum_[C].at(g);
                return Tangram::polygon3d_area(vertices_(C), polyCells_[C]->matface_vertices(g));
            }
            AmanziGeometry::Point normal(size_t C, size_t g) const final {
                g = faceRenum_[C].at(g);
                auto a = Tangram::polygon3d_normal(vertices_(C), polyCells_[C]->matface_vertices(g));
                // check dir
                return AmanziGeometry::Point(a[0], a[1], a[2]);
            }
            size_t parentFaceLocalIndex(size_t C, size_t g) const final {
                auto& logger = SingletonLogger::instance();
                auto p = faceCentroid(C, g);
                std::vector<AmanziGeometry::Point> coords;
                auto ind = macroFacesIndicies(C);
                std::vector<double> dist(ind.size(), std::numeric_limits<double>::max());
                for (size_t j = 0; j < ind.size(); ++j) {
                    auto f = ind[j];
                    auto n = mesh_->face_normal(f);
                    n /= AmanziGeometry::norm(n);
                    auto centroid = mesh_->face_centroid(f);
                    mesh_->face_get_coordinates(f, &coords);
                    for (size_t i = 1; i < coords.size(); ++i)
                        if (pointProjectionLiesInTriangle_(p, centroid, coords[i - 1], coords[i]))
                            dist[j] = std::min(dist[j], std::fabs(n * (centroid - p)));
                }
                auto min = std::min_element(dist.begin(), dist.end());
                if (!fpEqual_(*min, 0.)) {
                    logger.buf << __func__ << ": cell #" << C << ": mini-face #" << g << " centroid / macro-face distance is " << *min;
                    logger.wrn();
                }
                return std::distance(dist.begin(), min);
            }
        };
    }
}

#endif