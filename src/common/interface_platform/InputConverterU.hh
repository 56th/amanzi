/*
  This is the input component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_INPUT_CONVERTER_UNSTRUCTURED_HH_
#define AMANZI_INPUT_CONVERTER_UNSTRUCTURED_HH_

// TPLs
#include "xercesc/dom/DOM.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"

// Amanzi's
#include "VerboseObject.hh"

#include "InputConverter.hh"
#include "InputConverterU_Defs.hh"

namespace Amanzi {
namespace AmanziInput {

typedef std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> > PK;
typedef std::map<std::string, std::vector<std::string> > Tree;

class InputConverterU : public InputConverter {
 public:
  InputConverterU() :
      vo_(NULL),
      flow_single_phase_(false),
      compressibility_(false) {};
  ~InputConverterU() { if (vo_ != NULL) delete vo_; }

  // main members
  Teuchos::ParameterList Translate();

 private:
  void ParseSolutes_();

  Teuchos::ParameterList TranslateVerbosity_();
  Teuchos::ParameterList TranslateMesh_();
  Teuchos::ParameterList TranslateRegions_();
  Teuchos::ParameterList TranslateOutput_();
  Teuchos::ParameterList TranslatePreconditioners_();
  Teuchos::ParameterList TranslateTrilinosML_();
  Teuchos::ParameterList TranslateHypreAMG_();
  Teuchos::ParameterList TranslateBILU_();
  Teuchos::ParameterList TranslateSolvers_();
  Teuchos::ParameterList TranslateState_();
  Teuchos::ParameterList TranslateMaterialsPartition_();
  Teuchos::ParameterList TranslateCycleDriver_();
  Teuchos::ParameterList TranslateTimePeriodControls_();
  Teuchos::ParameterList TranslatePKs_(const Teuchos::ParameterList& cd_list);
  Teuchos::ParameterList TranslateDiffusionOperator_(
      const std::string& disc_method, const std::string& pc_method,
      const std::string& nonlinear_solver, const std::string& rel_perm);
  Teuchos::ParameterList TranslateTimeIntegrator_(
      const std::string& err_options, const std::string& nonlinear_solver,
      bool modify_correction, const std::string& unstr_colntrols);
  Teuchos::ParameterList TranslateInitialization_(
      const std::string& unstr_controls);
  Teuchos::ParameterList TranslateFlow_(int regime = FLOW_BOTH_REGIMES);
  Teuchos::ParameterList TranslateWRM_();
  Teuchos::ParameterList TranslatePOM_();
  Teuchos::ParameterList TranslateFlowBCs_();
  Teuchos::ParameterList TranslateTransport_();
  Teuchos::ParameterList TranslateTransportBCs_();
  Teuchos::ParameterList TranslateTransportSources_();
  Teuchos::ParameterList TranslateChemistry_();
  Teuchos::ParameterList TranslateEnergy_();

  void ProcessMacros_(const std::string& prefix, char* text_content,
                      Teuchos::ParameterList& mPL, Teuchos::ParameterList& outPL);

  void RegisterPKsList_(Teuchos::ParameterList& pk_tree, Teuchos::ParameterList& pks_list);

 private:
  int dim_;
  Tree tree_;
  Tree phases_;

  std::map<std::string, std::string> pk_model_;

  // global flow constants
  bool flow_single_phase_;
  bool compressibility_;
  double rho_;

  // global transport and chemistry constants
  std::vector<std::string> comp_names_all_;

  // for analysis
  std::vector<std::string> vv_bc_regions_;
  std::vector<std::string> vv_src_regions_;

  Teuchos::ParameterList verb_list_;
  VerboseObject* vo_;
};

}  // namespace AmanziInput
}  // namespace Amanzi

#endif
