/* -*-  mode: c++; indent-tabs-mode: nil -*- */
#include "mineral.hh"

#include <sstream>
#include <iostream>
#include <iomanip>

#include "secondary_species.hh"
#include "matrix_block.hh"
#include "chemistry_verbosity.hh"

namespace Amanzi {
namespace AmanziChemistry {

Mineral::Mineral()
    : SecondarySpecies(),
      verbosity_(kSilent),
      molar_volume_(0.0),
      specific_surface_area_(0.0),
      volume_fraction_(0.0) {
}  // end Mineral() constructor

Mineral::Mineral(const SpeciesName in_name,
                 const SpeciesId in_id,
                 const std::vector<SpeciesName>& in_species,
                 const std::vector<double>& in_stoichiometries,
                 const std::vector<int>& in_species_ids,
                 const double in_h2o_stoich,
                 const double in_mol_wt,
                 const double in_logK,
                 const double molar_volume,
                 const double specific_surface_area)
    : SecondarySpecies(in_name, in_id,
                       in_species, in_stoichiometries, in_species_ids,
                       in_h2o_stoich, 0., in_mol_wt, 0., in_logK),
      verbosity_(kSilent),
      molar_volume_(molar_volume),
      specific_surface_area_(specific_surface_area),
      volume_fraction_(0.0) {
}  // end Mineral costructor


Mineral::~Mineral() {
}  // end Mineral() destructor

void Mineral::UpdateSpecificSurfaceArea(void) {
  // updating SSA not supported at this time!

}  // end UpdateSpecificSurfaceArea()

void Mineral::UpdateVolumeFraction(const double rate,
                                   const double delta_time) {
  // NOTE: the rate is a dissolution rate so either need to use -rate
  // or vol_frac -= .... inorder to get the correct
  // dissolution/precipitation behavior.

  // TODO(bandre): Right now we are just setting volume fraction to
  // zero if they go negative, introducing mass balance errors! Need
  // to adjust time step or reaction rate in the N-R solve!

  // delta_vf = [m^3/mole] * [moles/m^3/sec] * [sec]
  volume_fraction_ -= molar_volume() * rate * delta_time;
  if (volume_fraction_ < 0.0) {
    volume_fraction_ = 0.0;
  }
  if (false) {
    std::stringstream message;
    message << name() << "::UpdateVolumeFraction() : \n"
            << "molar_volume : " << molar_volume() << "\n"
            << "rate : " << rate << "\n"
            << "dt : " << delta_time << "\n"
            << "delta_vf : " << molar_volume() * rate * delta_time << std::endl;
    chem_out->Write(Teuchos::VERB_HIGH, message);
  }
}  // end UpdateVolumeFraction()

void Mineral::Update(const std::vector<Species>& primary_species, const Species& water_species) {
  double lnQK = -lnK_;
  for (int i = 0; i < ncomp(); i++) {
    lnQK += stoichiometry_.at(i) * primary_species.at(species_ids_.at(i)).ln_activity();
  }
  // Add the contribution of the water activity
  lnQK += SecondarySpecies::h2o_stoich_ * std::log(water_species.act_coef());
  lnQK_ = lnQK;
}  // end update()

void Mineral::AddContributionToTotal(std::vector<double> *total) {
  static_cast<void>(total);
}  // end addContributionToTotal()

void Mineral::AddContributionToDTotal(const std::vector<Species>& primary_species,
                                      MatrixBlock* dtotal) {
  static_cast<void>(primary_species);
  static_cast<void>(dtotal);
}  // end addContributionToDTotal()


/*
**
**  Display functions
**
*/
void Mineral::Display(void) const {
  std::stringstream message;
  message << "    " << name() << " = ";
  for (unsigned int i = 0; i < species_names_.size(); i++) {
    message << std::setprecision(2) << stoichiometry_[i] << " " << species_names_[i];
    if (i < species_names_.size() - 1) {
      message << " + ";
    }
  }
  if (SecondarySpecies::h2o_stoich_!=0.0) {
    message << " + ";
    message << std::setprecision(2) << h2o_stoich_ << " " << "H2O";
  }
  message << std::endl;
  message << std::setw(40) << " "
          << std::setw(10) << std::setprecision(5) << std::fixed << logK_
          << std::setw(13) << std::scientific << molar_volume()
          << std::setw(13) << std::fixed << gram_molecular_weight()
          << std::setw(13) << specific_surface_area()
          << std::setw(13) << std::fixed << volume_fraction()
          << std::endl;
  chem_out->Write(Teuchos::VERB_HIGH, message);
}  // end Display()

void Mineral::DisplayResultsHeader(void) const {
  std::stringstream message;
  message << std::setw(15) << "Name"
          << std::setw(15) << "Q/K"
          << std::setw(15) << "SI"
          << std::endl;
  chem_out->Write(Teuchos::VERB_HIGH, message);
}  // end DisplayResultsHeader()

void Mineral::DisplayResults(void) const {
  std::stringstream message;
  message << std::setw(15) << name()
          << std::scientific << std::setprecision(5)
          << std::setw(15) << Q_over_K()
          << std::fixed << std::setprecision(3)
          << std::setw(15) << saturation_index()
          << std::endl;
  chem_out->Write(Teuchos::VERB_HIGH, message);
}  // end DisplayResults()

}  // namespace AmanziChemistry
}  // namespace Amanzi
