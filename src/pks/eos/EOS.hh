/*
  This is the EOS component of the ATS and Amanzi codes.
   
  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)

  EOS -- purely virtual base class for an EOS.
*/

#ifndef AMANZI_EOS_HH_
#define AMANZI_EOS_HH_

namespace Amanzi {
namespace EOS {

class EOS {
 public:
  virtual ~EOS() {};

  // Virtual methods that form the EOS
  virtual double MassDensity(double T, double p) = 0;
  virtual double DMassDensityDT(double T, double p) = 0;
  virtual double DMassDensityDp(double T, double p) = 0;

  virtual double MolarDensity(double T, double p) = 0;
  virtual double DMolarDensityDT(double T, double p) = 0;
  virtual double DMolarDensityDp(double T, double p) = 0;

  // If molar mass is constant, we can take some shortcuts if we need both
  // molar and mass densities.  MolarMass() is undefined if
  // !IsConstantMolarMass()
  virtual bool IsConstantMolarMass() = 0;
  virtual double MolarMass() = 0;
};

}  // namespace EOS
}  // namespace Amanzi

#endif
