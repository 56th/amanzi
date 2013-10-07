/*
  This is the mimetic discretization component of the Amanzi code. 

  Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Version: 2.0
  Release name: naka-to.
  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_WHETSTONE_NLFV_HH_
#define AMANZI_WHETSTONE_NLFV_HH_

#include "Teuchos_RCP.hpp"

#include "Mesh.hh"
#include "Point.hh"

#include "WhetStone_typedefs.hh"
#include "tensor.hh"


namespace Amanzi {
namespace WhetStone {

class NLFV { 
 public:
  NLFV(Teuchos::RCP<const AmanziMesh::Mesh> mesh) : mesh_(mesh) {};
  ~NLFV() {};

  void HarmonicAveragingPoint(int face, std::vector<Tensor>& T,
                              AmanziGeometry::Point& p, double& weight);

  int PositiveDecomposition(
      int id1, const std::vector<AmanziGeometry::Point>& tau,
      const AmanziGeometry::Point& conormal, double* ws, int* ids);

 private:
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
};

}  // namespace WhetStone
}  // namespace Amanzi

#endif

