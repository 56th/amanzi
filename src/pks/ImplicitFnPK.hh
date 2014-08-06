/*
  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov, Ethan Coon

  This is a base virtual class for process kernels. All physical 
  kernels and MPCs must implement this interface for use within 
  weak and strongly coupled hierarchies.
*/

#ifndef IMPLICIT_FN_PK_HH_
#define IMPLICIT_FN_PK_HH_

#include "Teuchos_RCP.hpp"

#include "ImplicitFn.hh"
#include "PK.hh"

namespace Amanzi {

class ImplictFnPK : public PK, public ImplicitFn {};

}  // namespace Amanzi

#endif
