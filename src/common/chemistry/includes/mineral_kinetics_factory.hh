/* -*-  mode: c++; indent-tabs-mode: nil -*- */
#ifndef AMANZI_CHEMISTRY_MINERAL_KINETICS_FACTORY_HH_

#define AMANZI_CHEMISTRY_MINERAL_KINETICS_FACTORY_HH_

/*******************************************************************************
 **
 **  File Name: MineralKineticsFactory.h
 **
 **  Description: factory class for building a mineral kinetic rate object
 **
 *******************************************************************************/
#include <vector>
#include <string>

#include "species.hh"
#include "mineral.hh"
#include "string_tokenizer.hh"

namespace Amanzi {
namespace AmanziChemistry {

class KineticRate;

class MineralKineticsFactory {
 public:
  MineralKineticsFactory(void);
  ~MineralKineticsFactory(void);

  KineticRate* Create(const std::string& rate_type,
                      const StringTokenizer& rate_data,
                      const Mineral& mineral,
                      const SpeciesArray& primary_species);

  SpeciesId VerifyMineralName(const std::string mineral_name,
                              const std::vector<Mineral>& minerals) const;


  void set_debug(const bool value) {
    this->debug_ = value;
  };
  bool debug(void) const {
    return this->debug_;
  };

 protected:

 private:
  bool debug_;
  static const std::string kTST;
};

}  // namespace AmanziChemistry
}  // namespace Amanzi
#endif     /* AMANZI_CHEMISTRY_MINERAL_KINETICS_FACTORY_HH_ */
