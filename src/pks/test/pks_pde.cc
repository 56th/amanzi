/* -*-  mode: c++; indent-tabs-mode: nil -*- */
/*
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.
*/

/* Test basic implicit and explicit PDEs

At this point PKs manage memory and interface time integrators with the DAG.
These tests that functionality with a series of ODEs.

*/

#include "Epetra_MpiComm.h"
#include "Epetra_Vector.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "UnitTest++.h"

#include "Mesh.hh"
#include "MeshFactory.hh"
#include "State.hh"
#include "CompositeVector.hh"
#include "TreeVector.hh"

#include "PK.hh"
#include "PK_Adaptors.hh"
#include "PK_Default.hh"
#include "PK_MixinExplicit.hh"
#include "PK_MixinExplicitSubcycled.hh"
#include "PK_MixinImplicit.hh"
#include "PK_MixinImplicitSubcycled.hh"
#include "PK_MixinLeaf.hh"
#include "PK_MixinPredictorCorrector.hh"

#include "pks_test_harness.hh"
#include "test_pk_pde.hh"

using namespace Amanzi;

static const double PI_2 = 1.5707963267948966;

SUITE(PKS_PDE) {

  // // Forward Euler tests with each of 3 PKs
  // TEST(DIFFUSION_FE_EXPLICIT) {
  //   using PK_t = PK_Explicit_Adaptor<PK_PDE_Explicit<PK_MixinExplicit<PK_MixinLeafCompositeVector<PK_Default>>>>;
  //   auto run = createRunPDE<PK_t>("diffusion FE", "test/pks_pde.xml");
  //   auto nsteps = run_test(run->S, run->pk);

  //   auto& u = *run->S->Get<CompositeVector>("u").ViewComponent("cell", false);

  //   auto m = run->S->GetMesh();
  //   int ncells = m->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED);
  //   for (int c=0; c!=ncells; ++c) {
  //     auto p = m->cell_centroid(c);
  //     double val = 2.0 * std::pow(PI_2,2) * std::cos(PI_2 * p[0]) * std::cos(PI_2*p[1]);
  //     CHECK_CLOSE(val, u[0][c], 1.e-10);
  //   }
  //   CHECK_EQUAL(10, nsteps.first);
  //   CHECK_EQUAL(0, nsteps.second);
  // }

  // Forward Euler tests with each of 3 PKs
  TEST(DIFFUSION_FE_IMPLICIT) {
    using PK_t = PK_Implicit_Adaptor<PK_PDE_Implicit<PK_MixinImplicit<PK_MixinLeafCompositeVector<PK_Default>>>>;
    auto run = createRunPDE<PK_t>("diffusion FE", "test/pks_pde.xml");
    auto nsteps = run_test(run->S, run->pk);

    auto& u = *run->S->Get<CompositeVector>("u").ViewComponent("cell", false);

    auto m = run->S->GetMesh();
    int ncells = m->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED);
    for (int c=0; c!=ncells; ++c) {
      auto p = m->cell_centroid(c);
      double val = 2.0 * std::pow(PI_2,2) * std::cos(PI_2 * p[0]) * std::cos(PI_2*p[1]);
      CHECK_CLOSE(val, u[0][c], 1.e-10);
    }
    CHECK_EQUAL(10, nsteps.first);
    CHECK_EQUAL(0, nsteps.second);
  }
}
