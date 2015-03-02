/*
  This is the flow component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)

  We use this class to test convergence of discretization
  schemes. It employs a simple model for relative permeability, 
  k_rel = 1 / (1 + p^2).
*/

#ifndef AMANZI_FAKE_MODEL_HH_
#define AMANZI_FAKE_MODEL_HH_

#include "Teuchos_ParameterList.hpp"

#include "factory.hh"

#include "WRM.hh"

namespace Amanzi {
namespace Flow {

class WRM_fake : public WRM {
 public:
  explicit WRM_fake(Teuchos::ParameterList& plist);
  explicit WRM_fake(std::string region);
  ~WRM_fake() {};
  
  // required methods from the base class
  double k_relative(double pc);
  double saturation(double pc);
  double dSdPc(double pc);  
  double capillaryPressure(double saturation);
  double residualSaturation() { return 0.0; }

 private:
  double m, n, alpha;

  static Utils::RegisteredFactory<WRM, WRM_fake> factory_;
};

}  // namespace Flow
}  // namespace Amanzi
 
#endif
