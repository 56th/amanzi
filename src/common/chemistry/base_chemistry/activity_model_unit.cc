/* -*-  mode: c++; indent-tabs-mode: nil -*- */
#include "activity_model_unit.hh"

#include <cmath>

#include <iostream>

namespace Amanzi {
namespace AmanziChemistry {

ActivityModelUnit::ActivityModelUnit()
    : ActivityModel() {
}  // end ActivityModelUnit constructor


ActivityModelUnit::~ActivityModelUnit() {
}  // end ActivityModelUnit destructor

double ActivityModelUnit::Evaluate(const Species& species) {
  static_cast<void>(species);
  // log(gamma_i) = 0.0, gamma_i = 1.0

  return 1.0;
}  // end Evaluate()

void ActivityModelUnit::EvaluateVector(
    const std::vector<Species>& prim, 
    const std::vector<AqueousEquilibriumComplex>& sec,
    std::vector<double>* gamma, 
    double* actw) {
  //const double r1(1.0e0);
  for (std::vector<double>::iterator i = gamma->begin(); i != gamma->end(); ++i) {
    (*i) = 1.0;
  }
  *actw = 1.0;
}  // end EvaluateVector


void ActivityModelUnit::Display(void) const {
  std::cout << "Activity Model: unit activity coefficients (gamma = 1.0)."
            << std::endl;
}  // end Display()

}  // namespace AmanziChemistry
}  // namespace Amanzi
