/*
  This is the flow component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov

  A collection of multiscale porosity models along with a mesh partition.
*/

#include "dbc.hh"
#include "MultiscalePorosityFactory.hh"
#include "MultiscalePorosityPartition.hh"

namespace Amanzi {
namespace Flow {

/* ******************************************************************
* Non-member factory.
****************************************************************** */
Teuchos::RCP<MultiscalePorosityPartition> CreateMultiscalePorosityPartition(
    Teuchos::RCP<const AmanziMesh::Mesh>& mesh,
    Teuchos::RCP<Teuchos::ParameterList> plist)
{
  MultiscalePorosityFactory factory;
  std::vector<Teuchos::RCP<MultiscalePorosity> > msp_list;
  std::vector<std::string> region_list;

  for (Teuchos::ParameterList::ConstIterator lcv = plist->begin(); lcv != plist->end(); ++lcv) {
    std::string name = lcv->first;
    if (plist->isSublist(name)) {
      Teuchos::ParameterList sublist = plist->sublist(name);
      region_list.push_back(sublist.get<std::string>("region"));
      msp_list.push_back(factory.Create(sublist));
    } else {
      ASSERT(0);
    }
  }

  Teuchos::RCP<Functions::MeshPartition> partition =
      Teuchos::rcp(new Functions::MeshPartition(AmanziMesh::CELL, region_list));
  partition->Initialize(mesh, -1);
  partition->Verify();

  return Teuchos::rcp(new MultiscalePorosityPartition(partition, msp_list));
}

}  // namespace Flow
}  // namespace Amanzi

