/*
  This is the flow component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)
           Konstantin Lipnikov (lipnikov@lanl.gov)

  The WRM Evaluator simply calls the WRM with the correct arguments.
*/

#ifndef AMANZI_FLOW_WRM_EVALUATOR_HH_
#define AMANZI_FLOW_WRM_EVALUATOR_HH_

#include "factory.hh"
#include "secondary_variables_field_evaluator.hh"
#include "WRM.hh"

namespace Amanzi {
namespace Flow {

class WRMEvaluator : public SecondaryVariablesFieldEvaluator {
 public:
  // constructor format for all derived classes
  explicit
  WRMEvaluator(Teuchos::ParameterList& plist,
               Teuchos::RCP<Teuchos::ParameterList> wrm_list,
               Teuchos::RCP<const AmanziMesh::Mesh> mesh);
  WRMEvaluator(const WRMEvaluator& other);

  virtual Teuchos::RCP<FieldEvaluator> Clone() const;

 protected:
  void InitializeFromPlist_();

  // Required methods from SecondaryVariableFieldEvaluator
  virtual void EvaluateField_(const Teuchos::Ptr<State>& S,
          const std::vector<Teuchos::Ptr<CompositeVector> >& results);
  virtual void EvaluateFieldPartialDerivative_(const Teuchos::Ptr<State>& S,
          Key wrt_key, const std::vector<Teuchos::Ptr<CompositeVector> > & results);

 protected:
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
  std::vector<Teuchos::RCP<WRM> > wrm_;
  Teuchos::RCP<Epetra_IntVector> cell2region_;
  Key pressure_key_;

 private:
  void CreateWRM_(Teuchos::ParameterList& plist);
  void CreateCell2Region_();
  static Utils::RegisteredFactory<FieldEvaluator,WRMEvaluator> factory_;
};

}  // namespace Flow
}  // namespace Amanzi

#endif
