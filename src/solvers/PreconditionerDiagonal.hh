/*
  This is the Linear Solver component of the Amanzi code.

  License: BSD
  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_PRECONDITIONER_DIAGONAL_HH_
#define AMANZI_PRECONDITIONER_DIAGONAL_HH_

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_MultiVector.h"
#include "Epetra_RowMatrix.h"

#include "exceptions.hh"
#include "Preconditioner.hh"

namespace Amanzi {
namespace AmanziPreconditioners {

class PreconditionerDiagonal : public Preconditioner {
 public:
  PreconditionerDiagonal() {};
  ~PreconditionerDiagonal() {};

  void Init(const std::string& name, const Teuchos::ParameterList& list) {};
  void Update(const Teuchos::RCP<Epetra_RowMatrix>& A) {
    diagonal_ = Teuchos::rcp(new Epetra_Vector(A->Map()));
    A->ExtractDiagonalCopy(*diagonal_);
  };
  void Destroy() {};

  int ApplyInverse(const Epetra_MultiVector& v, Epetra_MultiVector& hv) {
    return hv.ReciprocalMultiply(1.0, *diagonal_, v, 0.0); 
  }

  int returned_code() { return 0; }

 private:
  Teuchos::RCP<Epetra_Vector> diagonal_;
};

}  // namespace AmanziPreconditioners
}  // namespace Amanzi



#endif
