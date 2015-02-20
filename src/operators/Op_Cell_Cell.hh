/*
  This is the Operator component of the Amanzi code.

  Copyright 2010-2013 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)
*/

#ifndef AMANZI_OP_CELL_CELL_HH_
#define AMANZI_OP_CELL_CELL_HH_

#include <vector>
#include "Operator.hh"
#include "Op.hh"

/*
  Op classes are small structs that play two roles:

  1. They provide a class name to the schema, enabling visitor patterns.
  2. They are a container for local matrices.
  
  This Op class is for storing local matrices of length ncells and with dofs
  on cells and faces.
*/

namespace Amanzi {
namespace Operators {

class Op_Cell_Cell : public Op {
 public:
  Op_Cell_Cell(std::string& name,
               const Teuchos::RCP<const AmanziMesh::Mesh> mesh) :
      Op(OPERATOR_SCHEMA_BASE_CELL | OPERATOR_SCHEMA_DOFS_CELL,
         name, mesh) {
    vals.resize(mesh->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED), 0.);
    vals_shadow = vals;
  }

  virtual void ApplyMatrixFreeOp(const Operator* assembler,
          const CompositeVector& X, CompositeVector& Y) const {
    assembler->ApplyMatrixFreeOp(*this, X, Y);
  }

  virtual void SymbolicAssembleMatrixOp(const Operator* assembler,
          const SuperMap& map, GraphFE& graph,
          int my_block_row, int my_block_col) const {
    assembler->SymbolicAssembleMatrixOp(*this,
            map, graph, my_block_row, my_block_col);
  }

  virtual void AssembleMatrixOp(const Operator* assembler,
          const SuperMap& map, MatrixFE& mat,
          int my_block_row, int my_block_col) const {
    assembler->AssembleMatrixOp(*this, map, mat,
            my_block_row, my_block_col);
  }
  
  virtual bool ApplyBC(BCs& bc,
                       const Teuchos::Ptr<CompositeVector>& rhs,                       
                       bool bc_previously_applied) {
    return false;
  }

  virtual void Rescale(const CompositeVector& scaling) {
    if (scaling.HasComponent("cell")) {
      const Epetra_MultiVector& s_c = *scaling.ViewComponent("cell",false);
      for (int i = 0; i != vals.size(); ++i) {
        vals[i] *= s_c[0][i];
      }
    }
  }
  

};

}  // namespace Operators
}  // namespace Amanzi


#endif


