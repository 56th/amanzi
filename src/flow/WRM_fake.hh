/*
This is the flow component of the Amanzi code. 
License: BSD
Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_FAKE_MODEL_HH_
#define AMANZI_FAKE_MODEL_HH_

#include "WaterRetentionModel.hh"

/*
 We use this class to test convergence of discretization schemes.
 It uses the simplest model for relrative permeability, k_rel = 1 / (1 + p^2).
*/

namespace Amanzi {
namespace AmanziFlow {

class WRM_fake : public WaterRetentionModel {
 public:
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
};

}  // namespace AmanziFlow
}  // namespace Amanzi
 
#endif
