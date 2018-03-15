/*
  WhetStone

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#include "Teuchos_RCP.hpp"
#include "UnitTest++.h"

#include "Mesh.hh"
#include "MeshFactory.hh"

#include "MFD3D_CrouzeixRaviart.hh"
#include "MFD3D_Diffusion.hh"
#include "MFD3D_Lagrange.hh"
#include "MFD3D_LagrangeSerendipity.hh"
#include "Tensor.hh"


/* **************************************************************** */
TEST(HIGH_ORDER_CROUZEIX_RAVIART) {
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "Test: High-order Crouzeix Raviart in 2D" << std::endl;
  auto comm = std::make_shared<Epetra_MpiComm>(MPI_COMM_WORLD);

  MeshFactory meshfactory(comm.get());
  meshfactory.preference(FrameworkPreference({MSTK}));
  // Teuchos::RCP<Mesh> mesh = meshfactory(0.0, 0.0, 1.2, 1.1, 1, 1, Teuchos::null, true, true); 
  Teuchos::RCP<Mesh> mesh = meshfactory("test/one_pentagon.exo", Teuchos::null, true, true); 
 
  MFD3D_CrouzeixRaviart mfd(mesh);

  int cell(0);
  DenseMatrix N, R, G, A1, Ak;

  Tensor T(2, 1);
  T(0, 0) = 1.0;

  // 1st-order scheme
  mfd.StiffnessMatrix(cell, T, A1);

  // 1st-order scheme (new algorithm)
  mfd.StiffnessMatrixHO(cell, 1, T, R, G, Ak);

  printf("Stiffness matrix for order = 1\n");
  A1.PrintMatrix("%8.4f ");

  A1 -= Ak;
  CHECK(A1.NormInf() <= 1e-10);

  // high-order scheme (new algorithm)
  for (int k = 2; k < 4; ++k) {
    mfd.H1consistencyHO(cell, k, T, N, R, G, Ak);
    mfd.StiffnessMatrixHO(cell, k, T, R, G, Ak);

    printf("Stiffness matrix for order = %d\n", k);
    Ak.PrintMatrix("%8.4f ");

    // verify SPD propery
    int nrows = Ak.NumRows();
    for (int i = 0; i < nrows; i++) CHECK(Ak(i, i) > 0.0);

    // verify exact integration property
    DenseMatrix G1(G);
    G1.Multiply(N, R, true);
    G1(0, 0) = 1.0;
    G1.Inverse();
    G1(0, 0) = 0.0;
    G1 -= G;
    CHECK(G1.NormInf() <= 1e-10);
  }
}


/* **************************************************************** */
void HighOrderLagrange(std::string file_name) {
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: High-order Lagrange element in 2D, file=" << file_name << std::endl;
  auto comm = std::make_shared<Epetra_MpiComm>(MPI_COMM_WORLD);

  MeshFactory meshfactory(comm.get());
  meshfactory.preference(FrameworkPreference({MSTK}));
  // Teuchos::RCP<Mesh> mesh = meshfactory(0.0, 0.0, 1.0, 1.0, 1, 1, Teuchos::null, true, true); 
  Teuchos::RCP<Mesh> mesh = meshfactory(file_name, Teuchos::null, true, true); 
 
  int ncells = mesh->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::ALL);

  MFD3D_Diffusion mfd_lo(mesh);
  MFD3D_Lagrange mfd_ho(mesh);

  for (int c = 0; c < ncells; ++c) {
    DenseMatrix N, A1, Ak;

    Tensor T(2, 1);
    T(0, 0) = 1.0;

    // 1st-order scheme
    mfd_lo.StiffnessMatrix(c, T, A1);

    // 1st-order scheme (new algorithm)
    mfd_ho.set_order(1);
    mfd_ho.StiffnessMatrix(c, T, Ak);

    printf("Stiffness matrix for order=1, cell=%d\n", c);
    A1.PrintMatrix("%8.4f ");

    A1 -= Ak;
    CHECK(A1.NormInf() <= 1e-10);

    // high-order scheme (new algorithm)
    for (int k = 2; k < 4; ++k) {
      mfd_ho.set_order(k);
      mfd_ho.H1consistency(c, T, N, Ak);
      mfd_ho.StiffnessMatrix(c, T, Ak);

      printf("Stiffness matrix for order=%d, cell=%d\n", k, c);
      Ak.PrintMatrix("%8.4f ");

      // verify SPD propery
      int nrows = Ak.NumRows();
      for (int i = 0; i < nrows; i++) CHECK(Ak(i, i) > 0.0);

      // verify exact integration property
      const DenseMatrix& R = mfd_ho.R();
      const DenseMatrix& G = mfd_ho.G();
      DenseMatrix G1(G);

      G1.Multiply(N, R, true);
      G1(0, 0) = 1.0;
      G1.Inverse();
      G1(0, 0) = 0.0;
      G1 -= G;
      CHECK(G1.NormInf() <= 1e-12 * G.NormInf());
    }
  }
}


TEST(HIGH_ORDER_LAGRANGE) {
  HighOrderLagrange("test/one_pentagon.exo");
  HighOrderLagrange("test/two_cell2_dist.exo");
} 


/* **************************************************************** */
void HighOrderLagrangeSerendipity(std::string file_name) {
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::WhetStone;

  std::cout << "\nTest: High-order Lagrange Serendipity element in 2D, file=" << file_name << std::endl;
  auto comm = std::make_shared<Epetra_MpiComm>(MPI_COMM_WORLD);

  MeshFactory meshfactory(comm.get());
  meshfactory.preference(FrameworkPreference({MSTK}));
  Teuchos::RCP<const AmanziGeometry::GeometricModel> gm;
  // Teuchos::RCP<Mesh> mesh = meshfactory(0.0, 0.0, 2.0, 2.0, 1, 1, gm, true, true); 
  Teuchos::RCP<Mesh> mesh = meshfactory(file_name, gm, true, true); 
 
  int ncells = mesh->num_entities(AmanziMesh::CELL, AmanziMesh::Parallel_type::ALL);

  MFD3D_Diffusion mfd_lo(mesh);
  MFD3D_LagrangeSerendipity mfd_ho(mesh);

  for (int c = 0; c < ncells; ++c) {
    if (mesh->cell_get_num_faces(c) < 4) continue;

    Tensor T(2, 1);
    T(0, 0) = 1.0;

    // 1st-order scheme
    DenseMatrix N, A1, Ak;
    mfd_lo.StiffnessMatrix(c, T, A1);

    mfd_ho.set_order(1);
    mfd_ho.StiffnessMatrix(c, T, Ak);

    printf("Stiffness matrix for order=1, cell=%d\n", c);
    Ak.PrintMatrix("%8.4f ");

    A1 -= Ak;
    CHECK(A1.NormInf() <= 1e-10);

    // high-order schemes
    for (int k = 2; k < 4; ++k) {
      mfd_ho.set_order(k);
      mfd_ho.H1consistency(c, T, N, Ak);
      mfd_ho.StiffnessMatrix(c, T, Ak);

      printf("Stiffness matrix for order=%d, cell=%d\n", k, c);
      Ak.PrintMatrix("%8.3f ");

      // verify SPD propery
      int nrows = Ak.NumRows();
      for (int i = 0; i < nrows; i++) CHECK(Ak(i, i) > 0.0);

      // verify exact integration property
      const DenseMatrix& R = mfd_ho.R();
      const DenseMatrix& G = mfd_ho.G();

      DenseMatrix G1(G);
      G1.PutScalar(0.0);
      G1.Multiply(N, R, true);
      G1(0, 0) = 1.0;
      G1.Inverse();
      G1(0, 0) = 0.0;
      G1 -= G;
      std::cout << "Norm of dG=" << G1.NormInf() << " " << G.NormInf() << std::endl;      
      CHECK(G1.NormInf() <= 1e-12 * G.NormInf());
    }
  }
}


TEST(HIGH_ORDER_LAGRANGE_SERENDIPITY) {
  HighOrderLagrangeSerendipity("test/two_cell2_dist.exo");
  HighOrderLagrangeSerendipity("test/one_pentagon.exo");
} 
