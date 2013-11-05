/*
The transport component of the Amanzi code, serial unit tests.
License: BSD
Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#include "UnitTest++.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "MeshFactory.hh"
#include "MeshAudit.hh"

#include "Mesh.hh"
#include "Point.hh"

#include "mfd3d_diffusion.hh"
#include "tensor.hh"


/* **************************************************************** */
TEST(DARCY_MASS) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "Test: Mass matrix for Darcy" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(Simple);

  MeshFactory meshfactory(comm);
  meshfactory.preference(pref);
  RCP<Mesh> mesh = meshfactory(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1); 
 
  MFD3D_Diffusion mfd(mesh);

  int nfaces = 6, cell = 0;
  Tensor T(3, 1);
  T(0, 0) = 1;

  DenseMatrix M(nfaces, nfaces);
  for (int method = 0; method < 1; method++) {
    mfd.MassMatrix(cell, T, M);

    printf("Mass matrix for cell %3d\n", cell);
    for (int i=0; i<nfaces; i++) {
      for (int j=0; j<nfaces; j++ ) printf("%8.4f ", M(i, j)); 
      printf("\n");
    }

    // verify SPD propery
    for (int i=0; i<nfaces; i++) CHECK(M(i, i) > 0.0);

    // verify exact integration property
    AmanziMesh::Entity_ID_List faces;
    std::vector<int> dirs;
    mesh->cell_get_faces_and_dirs(cell, &faces, &dirs);
    
    double xi, yi, xj;
    double vxx = 0.0, vxy = 0.0, volume = mesh->cell_volume(cell); 
    for (int i = 0; i < nfaces; i++) {
      int f = faces[i];
      xi = mesh->face_normal(f)[0] * dirs[i];
      yi = mesh->face_normal(f)[1] * dirs[i];
      for (int j = 0; j < nfaces; j++) {
        f = faces[j];
        xj = mesh->face_normal(f)[0] * dirs[j];
        vxx += M(i, j) * xi * xj;
        vxy += M(i, j) * yi * xj;
      }
    }
    CHECK_CLOSE(vxx, volume, 1e-10);
    CHECK_CLOSE(vxy, 0.0, 1e-10);
  }

  delete comm;
}


/* **************************************************************** */
TEST(DARCY_INVERSE_MASS_3D) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: Inverse mass matrix for Darcy" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(Simple);

  MeshFactory factory(comm);
  factory.preference(pref);
  RCP<Mesh> mesh = factory(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 2, 3); 
 
  MFD3D_Diffusion mfd(mesh);

  int nfaces = 6, cell = 0;
  Tensor T(3, 1);  // tensor of rank 1
  T(0, 0) = 1.0;

  DenseMatrix W(nfaces, nfaces);
  for (int method = 0; method < 5; method++) {
    if (method == 0) 
      mfd.MassMatrixInverse(cell, T, W);
    else if (method == 1)
      mfd.MassMatrixInverseScaled(cell, T, W);
    else if (method == 2)
      mfd.MassMatrixInverseOptimizedScaled(cell, T, W);
    else if (method == 3)
      mfd.MassMatrixInverseSO(cell, T, W);
    else if (method == 4)
      mfd.MassMatrixInverseMMatrixHex(cell, T, W);

    printf("Inverse of mass matrix for method=%d\n", method);
    for (int i=0; i<6; i++) {
      for (int j=0; j<6; j++ ) printf("%8.4f ", W(i, j)); 
      printf("\n");
    }

    // verify SPD propery
    for (int i=0; i<nfaces; i++) CHECK(W(i, i) > 0.0);

    // verify exact integration property
    W.Inverse();

    AmanziMesh::Entity_ID_List faces;
    std::vector<int> dirs;
    mesh->cell_get_faces_and_dirs(cell, &faces, &dirs);
    
    double xi, yi, xj;
    double vxx = 0.0, vxy = 0.0, volume = mesh->cell_volume(cell); 
    for (int i = 0; i < nfaces; i++) {
      int f = faces[i];
      xi = mesh->face_normal(f)[0] * dirs[i];
      yi = mesh->face_normal(f)[1] * dirs[i];
      for (int j = 0; j < nfaces; j++) {
        f = faces[j];
        xj = mesh->face_normal(f)[0] * dirs[j];
        vxx += W(i, j) * xi * xj;
        vxy += W(i, j) * yi * xj;
      }
    }
    CHECK_CLOSE(vxx, volume, 1e-10);
    CHECK_CLOSE(vxy, 0.0, 1e-10);
  }

  delete comm;
}


/* **************************************************************** */
TEST(DARCY_FULL_TENSOR) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: Inverse mass matrix and full tensor" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(Simple);

  MeshFactory factory(comm);
  factory.preference(pref);
  RCP<Mesh> mesh = factory(0.0, 0.0, 0.0, 1.1, 1.0, 1.0, 3, 2, 1); 
 
  MFD3D_Diffusion mfd(mesh);

  int nfaces = 6, cell = 0;
  Tensor T(3, 2);  // tensor of rank 2
  T(0, 0) = 1.0;
  T(1, 1) = 2.0;
  T(2, 2) = 3.0;
  T(0, 1) = T(1, 0) = 1.0;
  T(1, 2) = T(2, 1) = 1.0;

  DenseMatrix W(nfaces, nfaces);
  for (int method = 0; method < 5; method++) {
    if (method == 0) 
      mfd.MassMatrixInverse(cell, T, W);
    else if (method == 1)
      mfd.MassMatrixInverseScaled(cell, T, W);
    else if (method == 2)
      mfd.MassMatrixInverseOptimizedScaled(cell, T, W);
    else if (method == 3)
      mfd.MassMatrixInverseSO(cell, T, W);
    else if (method == 4)
      mfd.MassMatrixInverseMMatrixHex(cell, T, W);

    printf("Inverse of mass matrix for method=%d\n", method);
    for (int i=0; i<6; i++) {
      for (int j=0; j<6; j++ ) printf("%8.4f ", W(i, j)); 
      printf("\n");
    }

    // verify SPD propery
    for (int i=0; i<nfaces; i++) CHECK(W(i, i) > 0.0);

    // verify exact integration property
    W.Inverse();

    AmanziMesh::Entity_ID_List faces;
    std::vector<int> dirs;
    mesh->cell_get_faces_and_dirs(cell, &faces, &dirs);
    
    AmanziGeometry::Point v(1.0, 2.0, 3.0);
    double xi, xj;
    double vxx = 0.0, volume = mesh->cell_volume(cell); 
    for (int i = 0; i < nfaces; i++) {
      int f = faces[i];
      xi = (v * mesh->face_normal(f)) * dirs[i];
      for (int j = 0; j < nfaces; j++) {
        f = faces[j];
        xj = (v * mesh->face_normal(f)) * dirs[j];
        vxx += W(i, j) * xi * xj;
      }
    }
    CHECK_CLOSE(vxx, 4 * volume, 1e-10);
  }

  delete comm;
}


/* **************************************************************** */
TEST(DARCY_STIFFNESS_2D) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: Stiffness matrix for Darcy in 2D" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory meshfactory(comm);
  meshfactory.preference(pref);
  RCP<Mesh> mesh = meshfactory(0.0, 0.0, 1.0, 1.0, 1, 1); 
 
  MFD3D_Diffusion mfd(mesh);

  int nnodes = 4, cell = 0;
  Tensor T(2, 1);
  T(0, 0) = 1;

  DenseMatrix A(nnodes, nnodes);
  mfd.StiffnessMatrix(cell, T, A);

  printf("Stiffness matrix for cell %3d\n", cell);
  for (int i=0; i<nnodes; i++) {
    for (int j=0; j<nnodes; j++ ) printf("%8.4f ", A(i, j)); 
    printf("\n");
  }

  // verify SPD propery
  for (int i=0; i<nnodes; i++) CHECK(A(i, i) > 0.0);

  // verify exact integration property
  AmanziMesh::Entity_ID_List nodes;
  std::vector<int> dirs;
  mesh->cell_get_nodes(cell, &nodes);
    
  int d = mesh->space_dimension();
  Point p(d);

  double xi, yi, xj;
  double vxx = 0.0, vxy = 0.0, volume = mesh->cell_volume(cell); 
  for (int i = 0; i < nnodes; i++) {
    int v = nodes[i];
    mesh->node_get_coordinates(v, &p);
    xi = p[0];
    yi = p[1];
    for (int j = 0; j < nnodes; j++) {
      v = nodes[j];
      mesh->node_get_coordinates(v, &p);
      xj = p[0];
      vxx += A(i, j) * xi * xj;
      vxy += A(i, j) * yi * xj;
    }
  }
  CHECK_CLOSE(vxx, volume, 1e-10);
  CHECK_CLOSE(vxy, 0.0, 1e-10);

  delete comm;
}


/* **************************************************************** */
TEST(DARCY_STIFFNESS_3D) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: Stiffness matrix for Darcy in 3D" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory meshfactory(comm);
  meshfactory.preference(pref);
  // RCP<Mesh> mesh = meshfactory(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1); 
  RCP<Mesh> mesh = meshfactory("test/one_cell.exo"); 
 
  MFD3D_Diffusion mfd(mesh);

  int nnodes = 8, cell = 0;
  Tensor T(3, 1);
  T(0, 0) = 1;

  DenseMatrix A(nnodes, nnodes);
  mfd.StiffnessMatrix(cell, T, A);

  printf("Stiffness matrix for cell %3d\n", cell);
  for (int i=0; i<nnodes; i++) {
    for (int j=0; j<nnodes; j++ ) printf("%8.4f ", A(i, j)); 
    printf("\n");
  }

  // verify SPD propery
  for (int i=0; i<nnodes; i++) CHECK(A(i, i) > 0.0);

  // verify exact integration property
  AmanziMesh::Entity_ID_List nodes;
  std::vector<int> dirs;
  mesh->cell_get_nodes(cell, &nodes);
    
  int d = mesh->space_dimension();
  Point p(d);

  double xi, yi, xj;
  double vxx = 0.0, vxy = 0.0, volume = mesh->cell_volume(cell); 
  for (int i = 0; i < nnodes; i++) {
    int v = nodes[i];
    mesh->node_get_coordinates(v, &p);
    xi = p[0];
    yi = p[1];
    for (int j = 0; j < nnodes; j++) {
      v = nodes[j];
      mesh->node_get_coordinates(v, &p);
      xj = p[0];
      vxx += A(i, j) * xi * xj;
      vxy += A(i, j) * yi * xj;
    }
  }
  CHECK_CLOSE(vxx, volume, 1e-10);
  CHECK_CLOSE(vxy, 0.0, 1e-10);

  delete comm;
}


/* **************************************************************** */
TEST(RECOVER_GRADIENT_MIXED) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: Recover gradient from Darcy fluxes" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory meshfactory(comm);
  meshfactory.preference(pref);
  RCP<Mesh> mesh = meshfactory("test/one_cell.exo"); 
 
  MFD3D_Diffusion mfd(mesh);

  // create Darcy fluxes
  AmanziMesh::Entity_ID_List faces;
  std::vector<int> dirs;

  int nfaces = 6, cell = 0;
  mesh->cell_get_faces_and_dirs(cell, &faces, &dirs);

  Point flux(1.0, 2.0,3.0);
  std::vector<double> solution(nfaces);

  for (int n = 0; n < nfaces; n++) {
    int f = faces[n];
    const Point& normal = mesh->face_normal(f);
    solution[n] = -normal * flux;
  }
  
  // gradient recovery
  Point gradient(3);
  mfd.RecoverGradient_MassMatrix(cell, solution, gradient);

  printf("Gradient %f %f %f\n", gradient[0], gradient[1], gradient[2]);

  CHECK_CLOSE(gradient[0], 1.0, 1e-10);
  CHECK_CLOSE(gradient[1], 2.0, 1e-10);
  CHECK_CLOSE(gradient[2], 3.0, 1e-10);

  delete comm;
}


/* **************************************************************** */
TEST(RECOVER_GRADIENT_NODAL) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: Recover gradient from nodal pressures" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory meshfactory(comm);
  meshfactory.preference(pref);
  RCP<Mesh> mesh = meshfactory("test/one_cell.exo"); 
 
  MFD3D_Diffusion mfd(mesh);

  // create pressure solution
  AmanziMesh::Entity_ID_List nodes;
  int nnodes = 8, cell = 0;
  mesh->cell_get_nodes(cell, &nodes);

  Point slope(1.0, 2.0,3.0);
  std::vector<double> solution(nnodes);
  Point xv(3);

  for (int n = 0; n < nnodes; n++) {
    int v = nodes[n];
    mesh->node_get_coordinates(v, &xv);
    solution[n] = slope * xv;
  }
  
  // gradient recovery
  Point gradient(3);
  mfd.RecoverGradient_StiffnessMatrix(cell, solution, gradient);

  printf("Gradient %f %f %f\n", gradient[0], gradient[1], gradient[2]);

  CHECK_CLOSE(gradient[0], 1.0, 1e-10);
  CHECK_CLOSE(gradient[1], 2.0, 1e-10);
  CHECK_CLOSE(gradient[2], 3.0, 1e-10);

  delete comm;
}


/* **************************************************************** */
TEST(DARCY_INVERSE_MASS_2D) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: Inverse mass matrix for Darcy, 2D" << endl;
#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);

  MeshFactory factory(comm);
  factory.preference(pref);
  // RCP<Mesh> mesh = factory.create(0.0, 0.0, 1.0, 1.0, 1, 1); 
  // RCP<Mesh> mesh = factory.create(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 2, 3); 
  // RCP<Mesh> mesh = factory("test/one_cell2.exo"); 
  RCP<Mesh> mesh = factory("test/dodecahedron.exo"); 
 
  MFD3D_Diffusion mfd(mesh);

  int ok, nfaces = 12, cell = 0, dim = mesh->space_dimension();
  Tensor T(dim, 2);  // tensor of rank 1
  T(0, 0) = 1.0;
  T(1, 1) = 1.0;
  T(2, 2) = 1.0;
  T(0, 1) = T(1, 0) = 0.5;

  DenseMatrix W(nfaces, nfaces);
  for (int method = 0; method < 1; method++) {
    if (method == 0) {
      ok = mfd.MassMatrixInverseMMatrix(cell, T, W);
    }

    printf("Inverse of mass matrix for method=%d  ierr=%d\n", method, ok);
    for (int i = 0; i < nfaces; i++) {
      for (int j = 0; j < nfaces; j++ ) printf("%8.4f ", W(i, j)); 
      printf("\n");
    }

    // verify monotonicity propery
    for (int i = 0; i < nfaces; i++) {
      CHECK(W(i, i) > 0.0);
      for (int j = i + 1; j < nfaces; j++) CHECK(W(i, j) < 1e-10);
    }

    // verify exact integration property
    W.Inverse();
    T.Inverse();

    AmanziMesh::Entity_ID_List faces;
    std::vector<int> dirs;
    mesh->cell_get_faces_and_dirs(cell, &faces, &dirs);
    
    double xi, yi, xj;
    double vxx = 0.0, vxy = 0.0, volume = mesh->cell_volume(cell); 
    for (int i = 0; i < nfaces; i++) {
      int f = faces[i];
      xi = mesh->face_normal(f)[0] * dirs[i];
      yi = mesh->face_normal(f)[1] * dirs[i];
      for (int j = 0; j < nfaces; j++) {
        f = faces[j];
        xj = mesh->face_normal(f)[0] * dirs[j];
        vxx += W(i, j) * xi * xj;
        vxy += W(i, j) * yi * xj;
      }
    }
    CHECK_CLOSE(vxx, volume * T(0, 0), 1e-10);
    CHECK_CLOSE(vxy, volume * T(0, 1), 1e-10);
  }

  delete comm;
}


