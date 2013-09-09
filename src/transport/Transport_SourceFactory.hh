/*
This is the transport component of the Amanzi code. 

Copyright 2010-2013 held jointly by LANS/LANL, LBNL, and PNNL. 
Amanzi is released under the three-clause BSD License. 
The terms of use and "as is" disclaimer for this license are 
provided Reconstruction.cppin the top-level COPYRIGHT file.

Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_TRANSPORT_SOURCE_FACTORY_HH_
#define AMANZI_TRANSPORT_SOURCE_FACTORY_HH_

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Point.hh"
#include "Mesh.hh"
#include "transport_domain_function.hh"

namespace Amanzi {
namespace AmanziTransport {

class TransportSourceFactory {
 public:
  TransportSourceFactory(const Teuchos::RCP<const AmanziMesh::Mesh> mesh,
                         const Teuchos::RCP<Teuchos::ParameterList> params)
     : mesh_(mesh), params_(params) {};
  ~TransportSourceFactory() {};
  
  Functions::TransportDomainFunction* CreateSource();

 private:
  void ProcessSourceSpec(Teuchos::ParameterList& list, 
                         const std::string& name, 
                         Functions::TransportDomainFunction* src) const;
  void ProcessStringActions(const std::string& name, int* method) const;
     
 private:
  const Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
  const Teuchos::RCP<Teuchos::ParameterList> params_;
};

}  // namespace AmanziTransport
}  // namespace Amanzi

#endif // TRANSPORT_SOURCE_FACTORY_HH_
