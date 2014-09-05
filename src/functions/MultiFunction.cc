/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */
/* -------------------------------------------------------------------------
ATS

License: see $AMANZI_DIR/COPYRIGHT
Author Ethan Coon

Factory for vector functions which are composed of multiple scalar functions.

The expected plist is of the form:

<ParameterList name="constuctor plist">
  <Parameter name="Number of DoFs">

  <ParameterList name="Function 1">
    <ParameterList name="function-constant">
      ...
    </ParameterList>
  </ParameterList>

  <ParameterList name="Function 2">
    <ParameterList name="function-linear">
      ...
    </ParameterList>
  </ParameterList>

  ...
</ParameterList>


Where each of the "Function X" lists are valid input to the
function-factory Create() method (see ./function-factory.hh).

------------------------------------------------------------------------- */

#include "dbc.hh"
#include "MultiFunction.hh"

namespace Amanzi {

MultiFunction::MultiFunction(
        const std::vector<Teuchos::RCP<const Function> >& functions) :
    functions_(functions) {
  values_ = new double[functions_.size()];
};

MultiFunction::MultiFunction(const Teuchos::RCP<const Function>& function) :
    functions_(1, function)  {
  values_ = new double[1];
};


MultiFunction::MultiFunction(Teuchos::ParameterList& plist) {
  FunctionFactory factory;

  if (plist.isParameter("Number of DoFs")) {
    if (plist.isType<int>("Number of DoFs")) {
      int ndofs = plist.get<int>("Number of DoFs");

      if (ndofs < 1) {
        // ERROR -- invalid number of dofs
        ASSERT(0);
      }

      for (int lcv=1; lcv!=(ndofs+1); ++lcv) {
        std::stringstream sublist_name;
        sublist_name << "DoF " << lcv << " Function";
        functions_.push_back(Teuchos::rcp(factory.Create(plist.sublist(sublist_name.str()))));
      }
    } else {
      // ERROR -- invalid number of dofs
      ASSERT(0);
    }
  } else {
    // assume it is a single dof function
    functions_.push_back(Teuchos::rcp(factory.Create(plist)));
  };

  values_ = new double[functions_.size()];
}


MultiFunction::~MultiFunction() {
  delete [] values_;
};


int MultiFunction::size() const {
  return functions_.size();
};


double* MultiFunction::operator()(const std::vector<double>& xt) const {
  for (int i=0; i!=size(); ++i) {
    values_[i] = (*functions_[i])(xt);
  }
  return values_;
};



} // namespace

