/*
  This is the Operator component of the Amanzi code.

  Copyright 2010-2013 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
           Ethan Coon (ecoon@lanl.gov)
*/

#ifndef AMANZI_OPERATOR_WITH_CONSISTENT_FACE_HH_
#define AMANZI_OPERATOR_WITH_CONSISTENT_FACE_HH_

#include "DenseMatrix.hh"
#include "OperatorDefs.hh"
#include "Operator.hh"

/* ******************************************************************
Special purpose operator, this takes FaceCell Ops and assembles a matrix
with only the Face entries, enabling the solution of consistent faces.
****************************************************************** */ 

namespace Amanzi {
namespace Operators {

class Operator_ConsistentFace : public Operator {
 public:
  // constuctors
  // main constructor
  //   The CVS is the domain and range of the operator
  Operator_ConsistentFace(const Teuchos::RCP<const CompositeVectorSpace>& cvs,
                    Teuchos::ParameterList& plist) :
      Operator(cvs, plist,
               OPERATOR_SCHEMA_BASE_CELL | OPERATOR_SCHEMA_DOFS_FACE) {};

  // visit methods for Apply
  virtual int ApplyMatrixFreeOp(const Op_Cell_FaceCell& op,
      const CompositeVector& X, CompositeVector& Y) const;

  // visit methods for symbolic assemble
  virtual void SymbolicAssembleMatrixOp(const Op_Cell_FaceCell& op,
          const SuperMap& map, GraphFE& graph,
          int my_block_row, int my_block_col) const;
  
  // visit methods for assemble
  virtual void AssembleMatrixOp(const Op_Cell_FaceCell& op,
          const SuperMap& map, MatrixFE& mat,
          int my_block_row, int my_block_col) const;
};

}  // namespace Operators
}  // namespace Amanzi

#endif

    

