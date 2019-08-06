/*
  Author: Alexander Zhiliakov alex@math.uh.edu
  Abstract mini-mesh interface
  This is used by ASC(n) method
*/

#ifndef AMANZI_MESH_MINI_TANGRAM_HH_
#define AMANZI_MESH_MINI_TANGRAM_HH_

#include <unordered_map>
#include <unordered_set>
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
            std::vector<std::unordered_map<size_t, size_t>> faceRenumInv_;
            std::vector<std::unordered_map<size_t, size_t>> gluedIntFaces_;
            std::vector<size_t> numbOfFaces_, numbOfExtFaces_;
            std::vector<std::vector<int>> parentFaceLocalIndicies_; // -1 for internal
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
                AmanziGeometry::Point const & t0,
                AmanziGeometry::Point const & t1,
                AmanziGeometry::Point const & t2
            ) const { // https://math.stackexchange.com/a/2579920/231246
                auto u = t1 - t0;
                auto v = t2 - t0;
                auto n = u ^ v;
                auto w = p - t0;
                auto gamma = (u ^ w) * n / (n * n);
                auto beta  = (w ^ v) * n / (n * n);
                auto alpha = 1. - gamma - beta;
                return fpInRange_(alpha, 0., 1.) &&
                       fpInRange_(beta,  0., 1.) &&
                       fpInRange_(gamma, 0., 1.);
            }
            bool fpEqual_(double a, double b, double tol = 1e-6) const {
                return std::fabs(a - b) < tol;
            }
            bool fpInRange_(double val, double a, double b, double tol = 1e-6) const {
                return a - tol <= val && val <= b + tol;
            }
        public:
            MeshMiniTangram(
                Teuchos::RCP<const Mesh> const & mesh, 
                std::vector<std::shared_ptr<Tangram::CellMatPoly<3>>> const & polyCells,
                double deleteEmptyFacesTol = -1., bool check = false
            ) 
            : MeshMini(mesh)
            , polyCells_(polyCells) {
                auto& logger = SingletonLogger::instance();
                auto n = mesh_->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::OWNED);
                faceRenum_.resize(n);
                faceRenumInv_.resize(n);
                numbOfFaces_.resize(n);
                numbOfExtFaces_.resize(n);
                gluedIntFaces_.resize(n);
                parentFaceLocalIndicies_.resize(n);
                logger.beg("glue mini-faces if needed, renumber mini-faces, and build mini-face to macro-face (child to parent) map");
                    for (size_t C = 0; C < n; ++C) {
                        logger.pro(C + 1, n);
                        auto m = numbOfMaterials(C);
                        std::string cellStr = m > 1 ? "MMC" : "SMC";
                        cellStr += " #" + std::to_string(C);
                        // remove empty tangram faces
                        std::unordered_set<size_t> deletedFaces;
                        for (size_t c = 0; c < m; ++c) {
                            auto& ind = const_cast<std::vector<int>&>(polyCells_[C]->matpoly_faces(c));
                            for (auto it = ind.begin(); it != ind.end(); ++it) {
                                auto area = Tangram::polygon3d_area(vertices_(C), polyCells_[C]->matface_vertices(*it));
                                if (fpEqual_(area, 0., deleteEmptyFacesTol)) {
                                    deletedFaces.insert(*it);
                                    ind.erase(it);
                                }
                            }
                        }
                        if (deletedFaces.size()) {
                            logger.buf << cellStr << ": deleted (empty) mini-faces = { ";
                            for (auto const & f : deletedFaces)
                                logger.buf << f << ' ';
                            logger.buf << "}\n";
                            // for (size_t c = 0; c < m; ++c) {
                            //     logger.buf << "new mini-faces = { ";
                            //     for (auto const & f : polyCells_[C]->matpoly_faces(c))
                            //         logger.buf << f << ' ';
                            //     logger.buf << "}\n";
                            // }
                        }
                        auto numMatfaces = polyCells_[C]->num_matfaces();
                        // build mini-face to macro-face (child to parent) map
                        for (size_t i = 0; i < numMatfaces; ++i) 
                            faceRenum_[C][i] = i; // no renum initially
                        parentFaceLocalIndicies_[C].resize(numMatfaces);
                        for (size_t g = 0; g < numMatfaces; ++g) {
                            if (deletedFaces.find(g) != deletedFaces.end()) {
                                parentFaceLocalIndicies_[C][g] = -2; // deleted face
                                continue;
                            }
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
                                for (size_t i = 1; i <= coords.size(); ++i)
                                    if (pointProjectionLiesInTriangle_(p, centroid, coords[i - 1], coords[i % coords.size()]))
                                        dist[j] = std::min(dist[j], std::fabs(n * (centroid - p)));
                            }
                            auto min = std::min_element(dist.begin(), dist.end());
                            parentFaceLocalIndicies_[C][g] = 
                                fpEqual_(*min, 0.)
                                ? std::distance(dist.begin(), min) // g is external
                                : -1; // g is internal
                        }
                        // glue internal faces if needed
                        std::vector<size_t> fInt;
                        for (size_t i = 0; i < parentFaceLocalIndicies_[C].size(); ++i)
                            if (parentFaceLocalIndicies_[C][i] == -1)
                                fInt.push_back(i);
                        for (size_t i = 0; i < fInt.size(); ++i) {
                            auto f1 = fInt[i];
                            if (gluedIntFaces_[C].find(f1) == gluedIntFaces_[C].end()) {
                                gluedIntFaces_[C][f1] = f1;
                                auto p1 = faceCentroid(C, f1);
                                for (size_t j = i + 1; j < fInt.size(); ++j) {
                                    auto f2 = fInt[j];
                                    auto p2 = faceCentroid(C, f2);
                                    if (gluedIntFaces_[C].find(f2) == gluedIntFaces_[C].end() && AmanziGeometry::norm(p1 - p2) < 1e-6)
                                        gluedIntFaces_[C][f2] = f1;
                                }
                            }
                        }
                        size_t nInt = 0;
                        for (auto const & kvp : gluedIntFaces_[C])
                            nInt += kvp.first == kvp.second;
                        if (nInt != fInt.size())
                            logger.buf << cellStr << ": numb of int faces = " << fInt.size() << " -> " << nInt;
                        if (m > 1) { // renumber faces
                            std::multimap<int, size_t, std::greater<int>> F2f; // sorted
                            for (size_t i = 0; i < parentFaceLocalIndicies_[C].size(); ++i)
                                F2f.insert(std::pair<int, size_t>(parentFaceLocalIndicies_[C][i], i));
                            size_t i = 0;
                            for (auto const & kvp : F2f) 
                                faceRenum_[C][i++] = kvp.second;
                        }
                        for (auto const & kvp : faceRenum_[C])
                            faceRenumInv_[C][kvp.second] = kvp.first;
                        numbOfExtFaces_[C] = numMatfaces - fInt.size() - deletedFaces.size();
                        numbOfFaces_[C] = numbOfExtFaces_[C] + nInt;
                        if (logger.buf.tellp() != std::streampos(0)) logger.log();
                    }
                logger.end();
                if (!check) return;
                MeshMiniEmpty x(mesh);
                logger.beg("check mini-mesh");
                    for (size_t C = 0; C < n; ++C) {
                        std::string cellStr = numbOfMaterials(C) > 1 ? "MMC" : "SMC";
                        cellStr += " #" + std::to_string(C);
                        auto m = numbOfMaterials(C);
                        auto nExt = numbOfExtFaces(C);
                        auto nInt = numbOfFaces(C) - nExt;
                        size_t i;
                        bool ok = true;
                        for (i = 0; i < nExt; ++i)
                            if (parentFaceLocalIndex(C, i) == -1)
                                ok = false;
                        for (; i < nExt + nInt; ++i)
                            if (parentFaceLocalIndex(C, i) != -1)
                                ok = false;
                        if (!ok) {
                            logger.buf << cellStr << ": inconsistent numbering of int/ext mini-faces:\n";
                            for (i = 0; i < nExt + nInt; ++i)
                                logger.buf << parentFaceLocalIndex(C, i) << ' ';
                            logger.buf << '\n';
                        }
                        if (m == 1) {
                            if (nInt != 0)
                                logger.buf << cellStr << ": numb of int faces = " << nInt << " != 0\n";
                            // global mini-faces indicies consistency
                            auto f1 = facesGlobalIndicies(C, 0);
                            auto f2 = x.facesGlobalIndicies(C, 0);
                            if (f1 != f2) 
                                logger.buf << cellStr << ": faces global indicies {" << f1 << "} != {" << f2 << "}\n";
                            // face areas, centroids, and normals
                            double surfArea1 = 0., surfArea2 = 0.;
                            for (auto g : f1) {
                                surfArea1 += area(C, g);
                                surfArea2 += x.area(C, g);
                                auto diff = std::fabs(area(C, g) - x.area(C, g));
                                if (!fpEqual_(diff, 0.))
                                    logger.buf << cellStr << ": face #" << g << " area     diff = " << diff << ",\t" << area(C, g) << " vs. " << x.area(C, g) << '\n';
                                auto c1 = faceCentroid(C, g);
                                auto c2 = x.faceCentroid(C, g);
                                diff = AmanziGeometry::norm(c1 - c2);
                                if (!fpEqual_(diff, 0.))
                                    logger.buf << cellStr << ": face #" << g << " centroid diff = " << diff << ",\t" << c1 << " vs. " << c2 << '\n';
                                auto n1 = normal(C, g);
                                auto n2 = x.normal(C, g);
                                diff = AmanziGeometry::norm(n1 - n2);
                                if (!fpEqual_(diff, 0.))
                                    logger.buf << cellStr << ": face #" << g << " normal   diff = " << diff << ",\t" << n1 << " vs. " << n2 << '\n';
                            }
                            auto diff = std::fabs(surfArea1 - surfArea2);
                            if (!fpEqual_(diff, 0.))
                                logger.buf << cellStr << ": cell surf area diff = " << diff << ",\t" << surfArea1 << " vs. " << surfArea2 << '\n';
                            // centroids
                            auto c1 = centroid(C, 0);
                            auto c2 = x.centroid(C, 0);
                            if (!fpEqual_(AmanziGeometry::norm(c1 - c2), 0.))
                                logger.buf << cellStr << ": centroids err = " << AmanziGeometry::norm(c1 - c2) << '\n';
                            // volumes
                            auto v1 = volume(C, 0);
                            auto v2 = x.volume(C, 0);
                            if (!fpEqual_(v1 - v2, 0.))
                                logger.buf << cellStr << ": volume err = " << std::fabs(v1 - v2) << '\n';
                        }
                        else {
                            if (m == 2 && nInt != 1)
                                logger.buf << cellStr << ", 2 mat: numb of int faces = " << nInt << " != 1\n";
                            if (m == 3 && nInt != 2)
                                logger.buf << cellStr << ", 3 mat: numb of int faces = " << nInt << " != 2\n";
                            if (m > 3)
                                logger.buf << cellStr << ": numb of mat = " << m << " > 3\n";
                            double surfArea1 = 0., surfArea2 = 0.;
                            std::vector<AmanziGeometry::Point> n2;
                            for (size_t f = 0; f < x.numbOfExtFaces(C); ++f) {
                                surfArea2 += x.area(C, f);
                                n2.push_back(x.normal(C, f));
                            }
                            for (size_t f = 0; f < nExt; ++f) {
                                surfArea1 += area(C, f);
                                auto F = parentFaceLocalIndex(C, f);
                                auto n1 = normal(C, f);
                                auto diff = AmanziGeometry::norm(n1 - n2[F]);
                                if (!fpEqual_(diff, 0.))
                                    logger.buf << cellStr << ": face #" << f << " normal diff = " << diff << ",\t" << n1 << " vs. " << n2[F] << '\n';
                            }
                            auto diff = std::fabs(surfArea1 - surfArea2);
                            if (!fpEqual_(diff, 0.))
                                logger.buf << cellStr << ": cell surf area diff = " << diff << ",\t" << surfArea1 << " vs. " << surfArea2 << '\n';
                            auto v1 = 0.;
                            auto v2 = x.volume(C, 0);
                            for (size_t c = 0; c < m; ++c)
                                v1 += volume(C, c);
                            if (!fpEqual_(v1 - v2, 0.))
                                logger.buf << cellStr << ": volume err = " << std::fabs(v1 - v2) << '\n';    
                        }
                        if (logger.buf.tellp() != std::streampos(0)) logger.wrn();
                    }
                logger.end();
            }
            size_t numbOfFaces(size_t C) const final {
                return numbOfFaces_[C];
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
                std::vector<int> a = polyCells_[C]->matpoly_faces(c);
                for (auto& i : a) {
                    i = gluedIntFaces_[C].find(i) != gluedIntFaces_[C].end() ? gluedIntFaces_[C].at(i) : i;
                    i = faceRenumInv_[C].at(i);
                    if (parentFaceLocalIndex(C, i) == -2)
                        throw std::logic_error("deleted faces shall not be used");
                }
                return a;
            }    
            AmanziGeometry::Point faceCentroid(size_t C, size_t g) const final {
                g = faceRenum_[C].at(g);
                std::vector<double> a;
                auto pts = vertices_(C);
                auto vrt = polyCells_[C]->matface_vertices(g);
                Tangram::polygon3d_moments(pts, vrt, a);
                if (a[0] != 0.) return AmanziGeometry::Point(a[1] / a[0], a[2] / a[0], a[3] / a[0]);
                return AmanziGeometry::Point(pts[vrt[0]][0], pts[vrt[0]][1], pts[vrt[0]][2]);
            }
            double area(size_t C, size_t g) const final {
                g = faceRenum_[C].at(g);
                return Tangram::polygon3d_area(vertices_(C), polyCells_[C]->matface_vertices(g));
            }
            AmanziGeometry::Point normal(size_t C, size_t g) const final {
                auto i = parentFaceLocalIndex(C, g);
                if (i == -1) { // int mini-face
                    auto tmp = Tangram::polygon3d_normal(vertices_(C), polyCells_[C]->matface_vertices(faceRenum_[C].at(g)));
                    return AmanziGeometry::Point(tmp[0], tmp[1], tmp[2]);
                }
                auto b = macroFacesNormalsDirs(C)[i] * mesh_->face_normal(macroFacesIndicies(C)[i]);
                return b / AmanziGeometry::norm(b);
            }
            int parentFaceLocalIndex(size_t C, size_t g) const final {
                return parentFaceLocalIndicies_[C][faceRenum_[C].at(g)];
            }
        };
    }
}

#endif