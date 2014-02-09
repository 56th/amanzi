/*
This is the flow component of the Amanzi code.

 
Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
Amanzi is released under the three-clause BSD License. 
The terms of use and "as is" disclaimer for this license are 
provided in the top-level COPYRIGHT file.

Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <vector>
#include <string>
#include <algorithm>

#include "Teuchos_RCP.hpp"

#include "FlowDefs.hh"
#include "Flow_PK.hh"


namespace Amanzi {
namespace AmanziFlow {

/* ******************************************************************
* Process parameter for special treatment of static head b.c.                                           
****************************************************************** */
void Flow_PK::ProcessShiftWaterTableList(const Teuchos::ParameterList& list)
{
  std::string name("relative position of water table");
  if (list.isParameter(name)) {
    Errors::Message msg;
    msg << "\nFlow_PK: \"relative position of water table\" is obsolete.\n"
        << "         see section \"Boundary Conditions\" in the Native Spec.\n";
    Exceptions::amanzi_throw(msg);
  }

  const std::vector<Amanzi::Functions::Action>& actions = bc_head->actions();
  int nactions = actions.size();
  if (nactions > 0) { 
    const Epetra_BlockMap& fmap = mesh_->face_map(false);
    shift_water_table_ = Teuchos::rcp(new Epetra_Vector(fmap));
  }

  for (int i = 0; i < nactions; i++) {
    int method = actions[i].second;

    if (method == Functions::BOUNDARY_FUNCTION_ACTION_HEAD_RELATIVE)
        CalculateShiftWaterTable(actions[i].first);
  }
}


/* ******************************************************************
* Calculate distance to the top of a given surface where the water 
* table is set up. 
* WARNING: works only in 3D.                                            
****************************************************************** */
void Flow_PK::CalculateShiftWaterTable(const std::string region)
{
  double tol = 1e-6;
  Errors::Message msg;

  if (dim == 2) {
    msg << "Flow PK: \"relative/absolute\" action on static head BC is not supported in 2D.\n";
    Exceptions::amanzi_throw(msg);
  }

  AmanziMesh::Entity_ID_List cells, faces, ss_faces;
  AmanziMesh::Entity_ID_List nodes1, nodes2, common_nodes;
  std::vector<int> fdirs;

  AmanziGeometry::Point p1(dim), p2(dim), p3(dim);
  std::vector<AmanziGeometry::Point> edges;

  mesh_->get_set_entities(region, AmanziMesh::FACE, AmanziMesh::OWNED, &ss_faces);
  int n = ss_faces.size();

  for (int i = 0; i < n; i++) {
    int f1 = ss_faces[i];
    mesh_->face_get_cells(f1, AmanziMesh::USED, &cells);

    mesh_->face_get_nodes(f1, &nodes1);
    std::sort(nodes1.begin(), nodes1.end());

    int c = cells[0];
    mesh_->cell_get_faces_and_dirs(c, &faces, &fdirs);
    int nfaces = faces.size();

    // find all edges that intersection of boundary faces f1 and f2
    for (int j = 0; j < nfaces; j++) {
      int f2 = faces[j];
      if (f2 != f1) {
        mesh_->face_get_cells(f2, AmanziMesh::USED, &cells);
        int ncells = cells.size();
        if (ncells == 1) {
          mesh_->face_get_nodes(f2, &nodes2);
          std::sort(nodes2.begin(), nodes2.end());
          set_intersection(nodes1, nodes2, &common_nodes);

          int m = common_nodes.size();
          if (m > dim-1) {
            msg << "Flow PK: Unsupported configuration: two or more common edges.";
            Exceptions::amanzi_throw(msg);
          } else if (m == 1 && dim == 2) {
            int v1 = common_nodes[0];
            mesh_->node_get_coordinates(v1, &p1);
            edges.push_back(p1);
          } else if (m == 2 && dim == 3) {
            int v1 = common_nodes[0], v2 = common_nodes[1];
            mesh_->node_get_coordinates(v1, &p1);
            mesh_->node_get_coordinates(v2, &p2);

            p3 = p1 - p2;
            if (p3[0] * p3[0] + p3[1] * p3[1] > tol * L22(p3)) {  // filter out vertical edges
              edges.push_back(p1);
              edges.push_back(p2);
            }
          }
        }
      }
    }
  }
  int nedges = edges.size();

#ifdef HAVE_MPI
  int gsize;
  const MPI_Comm& comm = mesh_->get_comm()->Comm();
  MPI_Comm_size(comm, &gsize);
  int* edge_counts = new int[gsize];
  MPI_Allgather(&nedges, 1, MPI_INT, edge_counts, 1, MPI_INT, comm);

  // prepare send buffer
  int sendcount = nedges * dim;
  double* sendbuf = NULL;
  if (nedges > 0) sendbuf = new double[sendcount];
  for (int i = 0; i < nedges; i++) {
    for (int k = 0; k < dim; k++) sendbuf[dim * i + k] = edges[i][k];
  }

  // prepare receive buffer
  for (int i = 0; i < gsize; i++) edge_counts[i] *= dim;
  int recvcount = 0;
  for (int i = 0; i < gsize; i++) recvcount += edge_counts[i];
  double* recvbuf = new double[recvcount];

  int* displs = new int[gsize];
  displs[0] = 0;
  for (int i = 1; i < gsize; i++) displs[i] = edge_counts[i-1] + displs[i-1];

  MPI_Allgatherv(sendbuf, sendcount, MPI_DOUBLE,
                 recvbuf, edge_counts, displs, MPI_DOUBLE, comm);

  // process receive buffer
  edges.clear();
  nedges = recvcount / dim;
  for (int i = 0; i < nedges; i++) {
    for (int k = 0; k < dim; k++) p1[k] = recvbuf[dim * i + k];
    edges.push_back(p1);
  }
#endif

  // calculate head shift
  double rho = *(S_->GetScalarData("fluid_density"));
  double edge_length, tol_edge, a, b;
  double rho_g = -rho * gravity_[dim - 1];

  for (int i = 0; i < n; i++) {
    int f = ss_faces[i];
    const AmanziGeometry::Point& xf = mesh_->face_centroid(f);

    int flag = 0;
    for (int j = 0; j < nedges; j += 2) {
      p1 = edges[j + 1] - edges[j];
      p2 = xf - edges[j];

      edge_length = p1[0] * p1[0] + p1[1] * p1[1];
      tol_edge = tol * edge_length;

      a = (p1[0] * p2[0] + p1[1] * p2[1]) / edge_length;
      b = fabs(p1[0] * p2[1] - p1[1] * p2[0]);

      if (b < tol_edge && a > -0.01 && a < 1.01) {
        double z = edges[j][2] + a * p1[2];
        (*shift_water_table_)[f] = z * rho_g;
        if (z > xf[2]) {
          flag = 1;
          break;
        }
      }
    }

    if (flag == 0) {
      // msg << "Flow PK: The boundary region \"" << region.c_str() << "\" is not piecewise flat.";
      // Exceptions::amanzi_throw(msg);
      // Instead, we take the closest mid-edge point with a higher z-coordinate.
      double z, d, dmin = 1e+99;
      for (int j = 0; j < nedges; j += 2) {
        p1 = (edges[j] + edges[j + 1]) / 2;
        d = L22(p1 - xf);
        if (p1[2] > xf[2] && d < dmin) {
          dmin = d;
          z = p1[2];
        }
      }
      (*shift_water_table_)[f] = z * rho_g;
    }
  }

#ifdef HAVE_MPI
  delete [] edge_counts;
  delete [] displs;
  delete [] recvbuf;
  if (sendbuf != NULL) delete [] sendbuf;
#endif

  if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM) {
    Teuchos::OSTab tab = vo_->getOSTab();
    *vo_->os() << "found " << nedges/2 << " boundary edges for side set " << region.c_str() << std::endl;
  }
}


/* ******************************************************************
* New implementation of the STL function.                                              
****************************************************************** */
void Flow_PK::set_intersection(const std::vector<AmanziMesh::Entity_ID>& v1,
                               const std::vector<AmanziMesh::Entity_ID>& v2, std::vector<AmanziMesh::Entity_ID>* vv)
{
  int i(0), j(0), n1, n2;

  n1 = v1.size();
  n2 = v2.size();
  vv->clear();

  while (i < n1 && j < n2) {
    if (v1[i] < v2[j]) {
      i++;
    } else if (v2[j] < v1[i]) {
      j++;
    } else {
      vv->push_back(v1[i]);
      i++;
      j++;
    }
  }
}

}  // namespace AmanziFlow
}  // namespace Amanzi


