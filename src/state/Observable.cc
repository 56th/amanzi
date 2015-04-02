/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */
/* -------------------------------------------------------------------------
Amanzi

License:
Author: Markus Berndt
        Ethan Coon (ecoon@lanl.gov)

Observable data object

------------------------------------------------------------------------- */

#include <string>
#include <algorithm>

#include <boost/filesystem/operations.hpp>

#include "errors.hh"
#include "Mesh.hh"
#include "State.hh"
#include "Field.hh"

#include "Observable.hh"


namespace Amanzi {

double ObservableSum(double a, double b, double vol) { return a + b*vol; }
double ObservableMin(double a, double b, double vol) { return std::min(a,b); }
double ObservableMax(double a, double b, double vol) { return std::max(a,b); }


Observable::Observable(Teuchos::ParameterList& plist, Epetra_MpiComm *comm) :
    IOEvent(plist, comm),
    count_(0) {
  // process the spec
  name_ = plist.name();
  variable_ = plist.get<std::string>("variable");
  region_ = plist.get<std::string>("region");
  delimiter_ = plist.get<std::string>("delimiter", ",");

  functional_ = plist.get<std::string>("functional");
  if (functional_ == "Observation Data: Point" ||
      functional_ == "Observation Data: Integral") {
    function_ = &ObservableSum;
  } else if (functional_ == "Observation Data: Minimum") {
    function_ = &ObservableMin;
  } else if (functional_ == "Observation Data: Maximum") {
    function_ = &ObservableMax;
  } else {
    Errors::Message msg;
    msg << "Observable: unrecognized functional " << functional_;
    Exceptions::amanzi_throw(msg);
  }
    
  
  // entity of region
  location_ = plist.get<std::string>("location name", "cell");

  // hack to orient flux to outward-normal along a boundary only
  flux_normalize_ = plist.get<bool>("direction normalized flux", false);

  // write mode
  interval_ = plist.get<int>("write interval", 0);
  write_ = interval_ > 0;

  if (write_) {
    filenamebase_ = plist.get<std::string>("observation output filename");

    // open file only on process 0
    if (!comm->MyPID()) {
      std::string safename(name_);
      std::replace(safename.begin(), safename.end(), ' ', '_');
      std::replace(safename.begin(), safename.end(), ':', '_');
      std::stringstream filename;
      filename << filenamebase_ << "_" << safename;
      ASSERT(boost::filesystem::portable_file_name(filenamebase_));
      out_ = Teuchos::rcp(new std::ofstream(filenamebase_.c_str()));
    }
  }
}

void Observable::Update(const State& S,
                        Amanzi::ObservationData::DataTriple& data) {
  if (count_ == 0) WriteHeader_();

  ++count_;
  Update_(S, data);

  if (out_.get()) {
    if (data.is_valid) {
      *out_ << data.time << delimiter_ << " " << data.value << std::endl;
    } else {
      *out_ << data.time << delimiter_ << " " << "NaN" << std::endl;
    }

    if (count_ % interval_ == 0) out_->flush();
  }
}

void Observable::Flush() {
  if (out_.get()) out_->flush();
}

void Observable::WriteHeader_() {
  if (out_.get()) {
    *out_ << "# Observation Name: " << name_ << std::endl;
    *out_ << "# Region: " << region_ << std::endl;
    *out_ << "# Functional: " << functional_ << std::endl;
    *out_ << "# Variable: " << variable_ << std::endl;
    *out_ << "# ==========================================================="
          << std::endl;
    *out_ << "#" << std::endl;
    out_->precision(16);
    *out_ << std::scientific;
  }
}

void Observable::Update_(const State& S,
                          Amanzi::ObservationData::DataTriple& data) {
  data.time = S.time();

  Teuchos::RCP<const Field> field = S.GetField(variable_);

  if (field->type() == CONSTANT_SCALAR) {
    // scalars, just return the value
    data.value = *field->GetScalarData();
    data.is_valid = true;

  } else if (field->type() == COMPOSITE_VECTOR_FIELD) {
    // vector field
    Teuchos::RCP<const CompositeVector> vec = field->GetFieldData();
    ASSERT(vec->HasComponent(location_));

    AmanziMesh::Entity_kind entity = vec->Location(location_);
    AmanziMesh::Entity_ID_List ids;
    vec->Mesh()->get_set_entities(region_, entity, AmanziMesh::OWNED, &ids);

    double value(0.);
    if (functional_ == "Observation Data: Minimum") {
      value = 1.e20;
    } else if (functional_ == "Observation Data: Maximum") {
      value = -1.e20;
    }

    double volume(0.);
    const Epetra_MultiVector& subvec = *vec->ViewComponent(location_, false);

    if (entity == AmanziMesh::CELL) {
      for (AmanziMesh::Entity_ID_List::const_iterator id=ids.begin();
           id!=ids.end(); ++id) {
        double vol = vec->Mesh()->cell_volume(*id);
        value = (*function_)(value, subvec[0][*id], vol);
        volume += vol;
      }
    } else if (entity == AmanziMesh::FACE) {
      for (AmanziMesh::Entity_ID_List::const_iterator id=ids.begin();
           id!=ids.end(); ++id) {
        double vol = vec->Mesh()->face_area(*id);

        // hack to orient flux to outward-normal along a boundary only
        if (flux_normalize_) {
          AmanziMesh::Entity_ID_List cells;
          vec->Mesh()->face_get_cells(*id, AmanziMesh::USED, &cells);
          ASSERT(cells.size() == 1);
          AmanziMesh::Entity_ID_List faces;
          std::vector<int> dirs;
          vec->Mesh()->cell_get_faces_and_dirs(cells[0], &faces, &dirs);
          int i = std::find(faces.begin(), faces.end(), *id) - faces.begin();
          vol *= dirs[i];
        }

        value = (*function_)(value, subvec[0][*id], vol);
        volume += std::abs(vol);
      }
    } else if (entity == AmanziMesh::NODE) {
      for (AmanziMesh::Entity_ID_List::const_iterator id=ids.begin();
           id!=ids.end(); ++id) {
        double vol = 1.0;
        value = (*function_)(value, subvec[0][*id], vol);
        volume += vol;
      }
    }

    // syncronize the result across processors
    if (functional_ == "Observation Data: Point" ||
        functional_ == "Observation Data: Integral") {
      double local[2], global[2];
      local[0] = value; local[1] = volume;
      S.GetMesh()->get_comm()->SumAll(local, global, 2);

      if (volume > 0) {
        if (functional_ == "Observation Data: Point") {
          data.value = global[0] / global[1];
          data.is_valid = true;
        } else if (functional_ == "Observation Data: Integral") {
          data.value = global[0];
          data.is_valid = true;
        }
      } else {
        data.value = 0.;
        data.is_valid = false;
      }
    } else if (functional_ == "Observation Data: Minimum") {
      double global;
      S.GetMesh()->get_comm()->MinAll(&value, &global, 1);
      data.value = global;
      data.is_valid = true;
    } else if (functional_ == "Observation Data: Maximum") {
      double global;
      S.GetMesh()->get_comm()->MaxAll(&value, &global, 1);
      data.value = global;
      data.is_valid = true;
    } else {
      data.value = 0.;
      data.is_valid = false;
    }
  } else {
    data.value = 0.;
    data.is_valid = false;
  }
}


} // namespace


