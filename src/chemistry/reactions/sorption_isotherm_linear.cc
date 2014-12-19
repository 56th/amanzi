/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */
#include "sorption_isotherm_linear.hh"

#include <iostream>
#include <iomanip>
#include <string>

#include "VerboseObject.hh"
#include "sorption_isotherm.hh"

namespace Amanzi {
namespace AmanziChemistry {

extern VerboseObject* chem_out;

SorptionIsothermLinear::SorptionIsothermLinear()
    : SorptionIsotherm("linear", SorptionIsotherm::LINEAR),
      KD_(0.0),
      params_(1, 0.0) {
}  // end SorptionIsothermLinear() constructor

SorptionIsothermLinear::SorptionIsothermLinear(const double KD)
    : SorptionIsotherm("linear", SorptionIsotherm::LINEAR),
      KD_(KD),
      params_(1, 0.0){
}  // end SorptionIsothermLinear() constructor

SorptionIsothermLinear::~SorptionIsothermLinear() {
}  // end SorptionIsothermLinear() destructor

void SorptionIsothermLinear::Init(const double KD) {
  set_KD(KD);
}

const std::vector<double>& SorptionIsothermLinear::GetParameters(void) {
  params_.at(0) = KD();
  return params_;
}  // end GetParameters()

void SorptionIsothermLinear::SetParameters(const std::vector<double>& params) {
  set_KD(params.at(0));
}  // end SetParameters()


double SorptionIsothermLinear::Evaluate(const Species& primarySpecies ) {
  // Csorb = KD * activity
  // Units:
  // sorbed_concentration [mol/m^3 bulk] = KD [kg water/m^3 bulk] * 
  //   activity [mol/kg water]
  return KD() * primarySpecies.activity();
}  // end Evaluate()

double SorptionIsothermLinear::EvaluateDerivative(const Species& primarySpecies) {
  // Csorb = KD * activity
  // dCsorb/dCaq = KD * activity_coef
  // Units:
  //  KD [kg water/m^3 bulk]
  return KD() * primarySpecies.act_coef();
}  // end EvaluateDerivative()

void SorptionIsothermLinear::Display(void) const {
  std::stringstream message;
  message << std::setw(5) << "KD:"
          << std::scientific << std::setprecision(5)
          << std::setw(15) << KD() << std::endl;
  chem_out->Write(Teuchos::VERB_HIGH, message);
}  // end Display()

}  // namespace AmanziChemistry
}  // namespace Amanzi
