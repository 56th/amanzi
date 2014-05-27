/*
  This is the Linear Solver component of the Amanzi code.

  License: BSD
  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_PRECONDITIONER_IDENTITY_HH_
#define AMANZI_PRECONDITIONER_IDENTITY_HH_

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_MultiVector.h"
#include "Epetra_RowMatrix.h"

#include "exceptions.hh"
#include "Preconditioner.hh"

namespace Amanzi {
namespace AmanziPreconditioners {

class PreconditionerIdentity : public Preconditioner {
 public:
  PreconditionerIdentity() {};
  ~PreconditionerIdentity() {};

  void Init(const std::string& name, const Teuchos::ParameterList& list) {};
  void Update(const Teuchos::RCP<Epetra_RowMatrix>& A) {};
  void Destroy() {};

  int ApplyInverse(const Epetra_MultiVector& v, Epetra_MultiVector& hv) {
    hv = v;
    return 0;
  }

  int returned_code() { return 0; }
};

}  // namespace AmanziPreconditioners
}  // namespace Amanzi



#endif
