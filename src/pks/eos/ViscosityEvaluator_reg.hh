/*
  This is the EOS component of the ATS and Amanzi codes.
   
  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)

  ViscosityEvaluator is the interface between state/data and the model, a VPM.
*/

#include "ViscosityBaseFactory.hh"
#include "ViscosityEvaluator.hh"

namespace Amanzi {
namespace EOS {

// registry of method
Utils::RegisteredFactory<FieldEvaluator, ViscosityEvaluator> ViscosityEvaluator::factory_("viscosity");

}  // namespace EOS
}  // namespace Amanzi
