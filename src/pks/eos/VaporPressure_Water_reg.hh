/*
  This is the EOS component of the ATS and Amanzi codes.
   
  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)

  Saturated vapor pressure for vapor over water or ice, Sonntag (1990)
*/

#include <cmath>
#include "errors.hh"
#include "VaporPressure_Water.hh"

namespace Amanzi {
namespace EOS {

// registry of method
Utils::RegisteredFactory<VaporPressure_Base, VaporPressure_Water> VaporPressure_Water::factory_("water vapor over water/ice");

}  // namespace EOS
}  // namespace Amanzi
