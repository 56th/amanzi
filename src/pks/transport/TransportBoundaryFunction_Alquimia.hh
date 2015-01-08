/* -------------------------------------------------------------------------
This is the Transport component of Amanzi

License: see $AMANZI_DIR/COPYRIGHT
Author (v1): Neil Carlson
       (v2): Ethan Coon

Function applied to a mesh component with at most one function 
application per entity.
------------------------------------------------------------------------- */

#ifndef AMANZI_TRANSPORT_BOUNDARY_FUNCTION_ALQUIMIA_HH_
#define AMANZI_TRANSPORT_BOUNDARY_FUNCTION_ALQUIMIA_HH_

#include <vector>
#include <map>
#include <string>

#include "Teuchos_RCP.hpp"

#include "Mesh.hh"
#include "MultiFunction.hh"
#include "TransportBoundaryFunction.hh"

#ifdef ALQUIMIA_ENABLED
#include "Chemistry_State.hh"
#include "ChemistryEngine.hh"

namespace Amanzi {
namespace Transport {

class TransportBoundaryFunction_Alquimia : public TransportBoundaryFunction {
 public:
  TransportBoundaryFunction_Alquimia(const std::vector<double>& times,
                                     const std::vector<std::string>& cond_names, 
                                     const Teuchos::RCP<const AmanziMesh::Mesh> &mesh,
                                     Teuchos::RCP<AmanziChemistry::Chemistry_State> chem_state,
                                     Teuchos::RCP<AmanziChemistry::ChemistryEngine> chem_engine);
  ~TransportBoundaryFunction_Alquimia();
  
  void Compute(double time);

  void Define(const std::vector<std::string> &regions);

  void Define(std::string region);

 private:
  // The computational mesh.
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;

  // The geochemical conditions we are enforcing, and the times we are enforcing them at.
  std::vector<double> times_;
  std::vector<std::string> cond_names_;

  // Chemistry state and engine.
  Teuchos::RCP<AmanziChemistry::Chemistry_State> chem_state_;
  Teuchos::RCP<AmanziChemistry::ChemistryEngine> chem_engine_;

  // Containers for interacting with the chemistry engine.
  AlquimiaState alq_state_;
  AlquimiaMaterialProperties alq_mat_props_;
  AlquimiaAuxiliaryData alq_aux_data_;
  AlquimiaAuxiliaryOutputData alq_aux_output_;

  // A mapping of boundary face indices to interior cells.
  std::map<int, int> cell_for_face_;

};

}  // namespace Transport
}  // namespace Amanzi

#endif


#endif
