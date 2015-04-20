/*
  This is the EOS component of the ATS and Amanzi codes.
   
  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)

  Constant density/viscosity EOS, defaults to reasonable values for water.
*/

#include "EOS_Constant.hh"

namespace Amanzi {
namespace EOS {

EOS_Constant::EOS_Constant(Teuchos::ParameterList& eos_plist) :
    eos_plist_(eos_plist) {
  InitializeFromPlist_();
};


void EOS_Constant::InitializeFromPlist_() {
  // defaults to water
  if (eos_plist_.isParameter("Molar mass [kg/mol]")) {
    M_ = eos_plist_.get<double>("Molar mass [kg/mol]");
  } else {
    M_ = eos_plist_.get<double>("Molar mass [g/mol]", 18.0153) * 1.e-3;
  }

  if (eos_plist_.isParameter("Density [mol/m^3]")) {
    rho_ = eos_plist_.get<double>("Density [mol/m^3]") * M_;
  } else {
    rho_ = eos_plist_.get<double>("Density [kg/m^3]", 1000.0);
  }
};

}  // namespace EOS
}  // namespace Amanzi
