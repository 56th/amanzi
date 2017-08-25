/* -*-  mode: c++; indent-tabs-mode: nil -*- */
#ifndef AMANZI_CHEMISTRY_UTILITIES_HH_
#define AMANZI_CHEMISTRY_UTILITIES_HH_

#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

//
// Common stand alone utility functions
//

#include "VerboseObject.hh"

namespace Amanzi {
namespace AmanziChemistry {

extern VerboseObject* chem_out;

namespace utilities {

/*******************************************************************************
 **
 **  Custom comparison operators
 **
 ******************************************************************************/
bool CaseInsensitiveStringCompare(const std::string& string1, 
                                  const std::string& string2);
bool CompareFabs(const double& a, const double& b);

/*******************************************************************************
 **
 **  String conversion utilities
 **
 ******************************************************************************/
void LowerCaseString(const std::string& in, std::string* out);
void RemoveLeadingAndTrailingWhitespace(std::string* line);

/*******************************************************************************
 **
 **  Math conversion utilities
 **
 ******************************************************************************/
double log_to_ln(double d);
double ln_to_log(double d);

/*******************************************************************************
 **
 **  Print Utilities
 **
 ******************************************************************************/
template <typename T>
void PrintVector(const std::string& name, 
                 const std::vector<T>& data,
                 const int precision = -1,
                 const bool comma_seperated = false) {
  std::stringstream output;
  if (precision > 0) {
    output << std::setprecision(precision);
  }
  output << name << " : { ";
  for (typename std::vector<T>::const_iterator i = data.begin();
       i != data.end(); ++i) {
    output << *i;
    if (i != --data.end()) {
      if (comma_seperated) {
        output << ", ";
      } else {
        output << "  ";
      }
    }
  }
  output << " }\n";
  chem_out->Write(Teuchos::VERB_HIGH, output);
}  // end PrintVector


}  // namespace utilities
}  // namespace AmanziChemistry
}  // namespace Amanzi
#endif  // AMANZI_CHEMISTRY_UTILITIES_HH_
