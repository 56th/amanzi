/*
  This is the operators component of the Amanzi code.

  License: BSD
  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_OPERATORS_BC_HH_
#define AMANZI_OPERATORS_BC_HH_

#include <vector>

namespace Amanzi {
namespace Operators {

/* *******************************************************************
* Three types of BCs are supported by this class:
*   [Dirichlet]                  u = u0 
*   [Neumann]     -K(u) grad u . n = g0
*   [Mixed] -K(u) grad u . n - c u = g1
*
* The right-hand side data (u0, g0, g1) must be placed in array 
* bc_value that has a proper size (see below). The type of BC 
* must be indicated in integer array bc_model using constants
* defined in file OperatorsDefs.hh. Arrays bc_value and bc_model
* must have the same size and contain ghost degrees of freedom.
*
* The coefficent c must be placed in array bc_mixed. This array
* can be empty; otherwise, its size must match that of bc_value.
*
* All three arrays are associated with degrees of freedom selected 
* for a problem discretization, see class Operators for more detail.
* For example, for the nodal discretization of elliptic equation,
* the dimension of the arrays equals to the total number of nodes 
* on a processor, including the ghost nodes.
*
* NOTE. Arrays bc_value and bc_model may be empty when homogeneous
*   Neumann boundary conditions are imposed on the domain boundary.
*
* NOTE. Suffient conditions for solution non-negativity are 
*   g0 <= 0, g1 <= 0 and c >=0.
*
* NOTE. All data in input arrays are given with respect to exterior
*   normal vector. Implementation of boundary conditions should take
*   into account that actual mesh normal may be oriented arbitrarily.
******************************************************************* */

class BCs {
 public:
  BCs() : type_(0) {};
  BCs(int type, std::vector<int>& bc_model, std::vector<double>& bc_value, std::vector<double>& bc_mixed) {
    Init(type, bc_model, bc_value, bc_mixed);
  }
  ~BCs() {};

  // main members
  void Init(int type, std::vector<int>& bc_model, std::vector<double>& bc_value, std::vector<double>& bc_mixed) {
    type_ = type;
    bc_model_ = &bc_model; 
    bc_value_ = &bc_value; 
    bc_mixed_ = &bc_mixed; 
  }

  bool CheckDataConsistency() {
    if (bc_value_->size() != bc_model_->size()) return false;
    if (bc_mixed_->size() != 0 && bc_mixed_->size() != bc_model_->size()) return false;
    return true; 
  }

  // access
  const std::vector<int>& bc_model() { return *bc_model_; }
  const std::vector<double>& bc_value() { return *bc_value_; }
  const std::vector<double>& bc_mixed() { return *bc_mixed_; }
  int type() { return type_; }

 private:
  int type_;
  std::vector<int>* bc_model_;
  std::vector<double>* bc_value_;
  std::vector<double>* bc_mixed_;
};

}  // namespace Operators
}  // namespace Amanzi


#endif


