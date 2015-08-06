/*
  This is the input component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <sstream>
#include <string>

// TPLs
#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>

// Amanzi's
#include "errors.hh"
#include "exceptions.hh"
#include "dbc.hh"

#include "InputConverterU.hh"
#include "InputConverterU_Defs.hh"

namespace Amanzi {
namespace AmanziInput {

XERCES_CPP_NAMESPACE_USE

/* ******************************************************************
* STATE sublist
****************************************************************** */
Teuchos::ParameterList InputConverterU::TranslateState_()
{
  Teuchos::ParameterList out_list;

  if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH) {
    *vo_->os() << "Translating state" << std::endl;
  }

  // first we write initial conditions for scalars and vectors, not region-specific
  Teuchos::ParameterList& out_ev = out_list.sublist("field evaluators");
  Teuchos::ParameterList& out_ic = out_list.sublist("initial conditions");

  Errors::Message msg;
  char* tagname;
  char* text_content;
  
  // --- gravity
  Teuchos::Array<double> gravity(dim_);
  for (int i = 0; i != dim_-1; ++i) gravity[i] = 0.0;
  gravity[dim_-1] = -GRAVITY_MAGNITUDE;
  out_ic.sublist("gravity").set<Teuchos::Array<double> >("value", gravity);

  // --- viscosity
  bool flag;
  DOMNode* node = getUniqueElementByTagNames_("phases", "liquid_phase", "viscosity", flag);
  text_content = XMLString::transcode(node->getTextContent());
  double viscosity = std::strtod(text_content, NULL);
  out_ic.sublist("fluid_viscosity").set<double>("value", viscosity);
  XMLString::release(&text_content);

  // --- density
  node = getUniqueElementByTagNames_("phases", "liquid_phase", "density", flag);
  text_content = XMLString::transcode(node->getTextContent());
  double density = std::strtod(text_content, NULL);
  out_ic.sublist("fluid_density").set<double>("value", density);
  XMLString::release(&text_content);

  out_ic.sublist("water_density").sublist("function").sublist("All")
      .set<std::string>("region","All")
      .set<std::string>("component","cell")
      .sublist("function").sublist("function-constant")
      .set<double>("value", density);

  // --- region specific initial conditions from material properties
  std::map<std::string, int> reg2mat;
  int mat(0);

  DOMNodeList* node_list = doc_->getElementsByTagName(XMLString::transcode("materials"));
  DOMNodeList* childern = node_list->item(0)->getChildNodes();

  for (int i = 0; i < childern->getLength(); i++) {
    DOMNode* inode = childern->item(i);
    if (DOMNode::ELEMENT_NODE == inode->getNodeType()) {
      DOMNamedNodeMap* attr_map = inode->getAttributes();
      DOMNode* node = attr_map->getNamedItem(XMLString::transcode("name"));
      if (!node) {
        ThrowErrorMissattr_("materials", "attribute", "name", "material");
      }

      node = getUniqueElementByTagNames_(inode, "assigned_regions", flag);
      text_content = XMLString::transcode(node->getTextContent());
      std::vector<std::string> regions = CharToStrings_(text_content);
      XMLString::release(&text_content);

      // record the material ID for each region that this material occupies
      for (int k = 0; k < regions.size(); k++) {
        if (reg2mat.find(regions[k]) == reg2mat.end()) {
          reg2mat[regions[k]] = mat++;
        } else {
          std::stringstream ss;
          ss << "There is more than one material assinged to region " << regions[k] << ".";
          Exceptions::amanzi_throw(Errors::Message(ss.str().c_str()));
        }
      }

      // create regions string
      std::string reg_str;
      for (std::vector<std::string>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
        reg_str = reg_str + *it;
      }

      // -- porosity: skip if compressibility model was already provided.
      if (!compressibility_) {
        double porosity;
        node = getUniqueElementByTagNames_(inode, "mechanical_properties", "porosity", flag);
        if (flag) {
          DOMNamedNodeMap* attr_map = node->getAttributes();
          DOMNode* node_tmp = attr_map->getNamedItem(XMLString::transcode("value"));
          if (node_tmp) {
            text_content = XMLString::transcode(node_tmp->getNodeValue());
            porosity = std::strtod(text_content, NULL);
            XMLString::release(&text_content);
          } else {
            ThrowErrorMissattr_("mechanical_properties", "attribute", "value", "porosity");
          }
        } else {
          msg << "Porosity element must be specified under mechanical_properties";
          Exceptions::amanzi_throw(msg);
        }
        Teuchos::ParameterList& porosity_ev = out_ev.sublist("porosity");
        porosity_ev.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions",regions)
            .set<std::string>("component","cell")
            .sublist("function").sublist("function-constant")
            .set<double>("value", porosity);
        porosity_ev.set<std::string>("field evaluator type", "independent variable");
      }

      // -- permeability.
      double perm_x, perm_y, perm_z;
      bool perm_init_from_file(false), conductivity(false);
      std::string perm_file, perm_attribute, perm_format;

      node = getUniqueElementByTagNames_(inode, "permeability", flag);
      if (!flag) {
        conductivity = true;
        node = getUniqueElementByTagNames_(inode, "hydraulic_conductivity", flag);
      }

      // first we get eilter permeability values or the file name
      int file(0);
      char* file_name;
      char* attr_name;
      double kx, ky, kz;

      DOMNamedNodeMap* attr_tmp = node->getAttributes();
      for (int k=0; k < attr_tmp->getLength(); k++) {
        DOMNode* knode = attr_tmp->item(k);

        if (DOMNode::ATTRIBUTE_NODE == knode->getNodeType()) {
          tagname = XMLString::transcode(knode->getNodeName());
          text_content = XMLString::transcode(knode->getNodeValue());

          if (strcmp(tagname, "x") == 0) {
            kx = std::strtod(text_content, NULL);
          } else if (strcmp(tagname, "y") == 0) {
            ky = std::strtod(text_content, NULL);
          } else if (strcmp(tagname, "z") == 0) {
            kz = std::strtod(text_content, NULL);
          } else if (strcmp(tagname, "type") == 0) {
            file++;
          } else if (strcmp(tagname, "filename") == 0) {
            file++;
            file_name = new char[std::strlen(text_content)];
            std::strcpy(attr_name, text_content);
          } else if (strcmp(tagname, "attribute") == 0) {
            file++;
            attr_name = new char[std::strlen(text_content)];
            std::strcpy(attr_name, text_content);
          }
          XMLString::release(&text_content);
          XMLString::release(&tagname);
        }
      }

      if (conductivity) {
        kx *= viscosity / (density * GRAVITY_MAGNITUDE);
        ky *= viscosity / (density * GRAVITY_MAGNITUDE);
        kz *= viscosity / (density * GRAVITY_MAGNITUDE);
      }

      // Second, we copy collected data to XML file.
      // For now permeability is not dumped into checkpoint files.
      Teuchos::ParameterList& permeability_ic = out_ic.sublist("permeability");
      permeability_ic.set<bool>("write checkpoint", false);

      if (file == 3) {
        permeability_ic.sublist("exodus file initialization")
            .set<std::string>("file", file_name)
            .set<std::string>("attribute", attr_name);
        delete file_name;
        delete attr_name;
      } else if (file == 0) {
        Teuchos::ParameterList& aux_list = permeability_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions",regions)
            .set<std::string>("component","cell")
            .sublist("function");
        aux_list.set<int>("Number of DoFs", dim_)
            .set<std::string>("Function type","composite function");
        aux_list.sublist("DoF 1 Function").sublist("function-constant").set<double>("value", kx);
        aux_list.sublist("DoF 2 Function").sublist("function-constant").set<double>("value", ky);
        if (dim_ == 3) {
          aux_list.sublist("DoF 3 Function").sublist("function-constant").set<double>("value", kz);
        }
      } else {
        ThrowErrorIllformed_("materials", "element", "file/filename/attribute");
      }

      // -- specific_yield
      node = getUniqueElementByTagNames_(inode, "mechanical_properties", "specific_yield", flag);
      if (flag) {
        double specific_yield = GetAttributeValueD_(static_cast<DOMElement*>(node), "value");

        Teuchos::ParameterList& spec_yield_ic = out_ic.sublist("specific_yield");
        spec_yield_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions",regions)
            .set<std::string>("component","cell")
            .sublist("function").sublist("function-constant")
            .set<double>("value", specific_yield);
      }

      // -- specific storage
      node = getUniqueElementByTagNames_(inode, "mechanical_properties", "specific_storage", flag);
      if (flag) {
        double specific_storage = GetAttributeValueD_(static_cast<DOMElement*>(node), "value");

        Teuchos::ParameterList& spec_yield_ic = out_ic.sublist("specific_storage");
        spec_yield_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions",regions)
            .set<std::string>("component","cell")
            .sublist("function").sublist("function-constant")
            .set<double>("value", specific_storage);
      }

      // -- particle density
      node = getUniqueElementByTagNames_(inode, "particle_density", flag);
      if (flag) {
        double particle_density = GetAttributeValueD_(static_cast<DOMElement*>(node), "value");

        Teuchos::ParameterList& part_dens_ic = out_ic.sublist("particle_density");
        part_dens_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions",regions)
            .set<std::string>("component","cell")
            .sublist("function").sublist("function-constant")
            .set<double>("value", particle_density);
      }
    }
  }

  // initialization of fields via the initial_conditions list
  node_list = doc_->getElementsByTagName(XMLString::transcode("initial_conditions"));
  childern = node_list->item(0)->getChildNodes();

  for (int i = 0; i < childern->getLength(); i++) {
    DOMNode* inode = childern->item(i);
    if (DOMNode::ELEMENT_NODE == inode->getNodeType()) {
      node = getUniqueElementByTagNames_(inode, "assigned_regions", flag);
      text_content = XMLString::transcode(node->getTextContent());
      std::vector<std::string> regions = CharToStrings_(text_content);
      XMLString::release(&text_content);

      // create regions string
      std::string reg_str;
      for (std::vector<std::string>::const_iterator it = regions.begin(); it != regions.end(); ++it) {
        reg_str = reg_str + *it;
      }

      // -- uniform pressure
      node = getUniqueElementByTagsString_(inode, "liquid_phase, liquid_component, uniform_pressure", flag);
      if (flag) {
        double p = GetAttributeValueD_(static_cast<DOMElement*>(node), "value");

        Teuchos::ParameterList& pressure_ic = out_ic.sublist("pressure");
        pressure_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions",regions)
            .set<std::string>("component","cell")
            .sublist("function").sublist("function-constant")
            .set<double>("value", p);
      }

      // -- linear pressure
      node = getUniqueElementByTagsString_(inode, "liquid_phase, liquid_component, linear_pressure", flag);
      if (flag) {
        double p = GetAttributeValueD_(static_cast<DOMElement*>(node), "value");
        std::vector<double> grad = GetAttributeVector_(static_cast<DOMElement*>(node), "gradient");
        std::vector<double> refc = GetAttributeVector_(static_cast<DOMElement*>(node), "reference_coord");

        Teuchos::Array<double> grad_with_time(grad.size() + 1);
        Teuchos::Array<double> refc_with_time(grad.size() + 1);

        grad_with_time[0] = 0.0;
        refc_with_time[0] = 0.0;

        for (int j = 0; j != grad.size(); ++j) {
          grad_with_time[j + 1] = grad[j];
          refc_with_time[j + 1] = refc[j];
        }

        Teuchos::ParameterList& pressure_ic = out_ic.sublist("pressure");
        pressure_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions", regions)
            .set<std::string>("component", "cell")
            .sublist("function").sublist("function-linear")
            .set<double>("y0", p)
            .set<Teuchos::Array<double> >("x0", refc_with_time)
            .set<Teuchos::Array<double> >("gradient", grad_with_time);
      }

      // -- uniform saturation
      node = getUniqueElementByTagsString_(inode, "liquid_phase, liquid_component, uniform_saturation", flag);
      if (flag) {
        double s = GetAttributeValueD_(static_cast<DOMElement*>(node), "value");

        Teuchos::ParameterList& saturation_ic = out_ic.sublist("saturation_liquid");
        saturation_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions", regions)
            .set<std::string>("component", "cell")
            .sublist("function").sublist("function-constant")
            .set<double>("value", s);
      }

      // -- linear saturation
      node = getUniqueElementByTagsString_(inode, "liquid_phase, liquid_component, linear_saturation", flag);
      if (flag) {
        double s = GetAttributeValueD_(static_cast<DOMElement*>(node), "value");
        std::vector<double> grad = GetAttributeVector_(static_cast<DOMElement*>(node), "gradient");
        std::vector<double> refc = GetAttributeVector_(static_cast<DOMElement*>(node), "reference_coord");

        Teuchos::Array<double> grad_with_time(grad.size() + 1);
        Teuchos::Array<double> refc_with_time(grad.size() + 1);

        grad_with_time[0] = 0.0;
        refc_with_time[0] = 0.0;

        for (int j = 0; j != grad.size(); ++j) {
          grad_with_time[j + 1] = grad[j];
          refc_with_time[j + 1] = refc[j];
        }

        Teuchos::ParameterList& saturation_ic = out_ic.sublist("saturation_liquid");
        saturation_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions", regions)
            .set<std::string>("component", "cell")
            .sublist("function").sublist("function-linear")
            .set<double>("y0", s)
            .set<Teuchos::Array<double> >("x0", refc_with_time)
            .set<Teuchos::Array<double> >("gradient", grad_with_time);
      }

      // -- darcy_flux
      node = getUniqueElementByTagsString_(inode, "liquid_phase, liquid_component, velocity", flag);
      if (flag) {
        std::vector<double> velocity;
        velocity.push_back(GetAttributeValueD_(static_cast<DOMElement*>(node), "x"));
        velocity.push_back(GetAttributeValueD_(static_cast<DOMElement*>(node), "y"));
        if (dim_ == 3) velocity.push_back(GetAttributeValueD_(static_cast<DOMElement*>(node), "z"));

        Teuchos::ParameterList& darcy_flux_ic = out_ic.sublist("darcy_flux");
        Teuchos::ParameterList& tmp_list =
            darcy_flux_ic.set<bool>("dot with normal", true)
            .sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions", regions)
            .set<std::string>("component", "face")
            .sublist("function")
            .set<int>("Number of DoFs", dim_)
            .set<std::string>("Function type", "composite function");

        for (int k = 0; k != dim_; ++k) {
          std::stringstream dof_str;
          dof_str << "DoF " << k+1 << " Function";
          tmp_list.sublist(dof_str.str()).sublist("function-constant")
                                         .set<double>("value", velocity[k]);
        }
      }

      // -- total_component_concentration...
      node = getUniqueElementByTagsString_(inode, "liquid_phase, solute_component", flag);
      if (flag) {
        Teuchos::ParameterList& tcc_ic = out_ic.sublist("total_component_concentration");
        Teuchos::ParameterList& dof_list = tcc_ic.sublist("function").sublist(reg_str)
            .set<Teuchos::Array<std::string> >("regions", regions)
            .set<std::string>("component","cell")
            .sublist("function")
            .set<int>("Number of DoFs", comp_names_all_.size())
            .set<std::string>("Function type", "composite function");

        int m(0);
        int ncomp = phases_["water"].size();
        for (int k = 0; k < ncomp; k++, m++) {
          std::string name = phases_["water"][k];
          double conc = GetAttributeValueD_(static_cast<DOMElement*>(node), "value");
          std::stringstream dof_str;

          dof_str << "DoF " << m + 1 << " Function";
          dof_list.sublist(dof_str.str()).sublist("function-constant").set<double>("value", conc);
        }
      }
    }
  }

  // add mesh partitions to the state list
  out_list.sublist("mesh partitions") = TranslateMaterialsPartition_();

  return out_list;
}


/* ******************************************************************
* Mesh patition sublist based on materials
****************************************************************** */
Teuchos::ParameterList InputConverterU::TranslateMaterialsPartition_()
{
  Teuchos::ParameterList out_list;
  Teuchos::ParameterList& tmp_list = out_list.sublist("materials");

  DOMNodeList* node_list = doc_->getElementsByTagName(XMLString::transcode("materials"));
  DOMNodeList* childern = node_list->item(0)->getChildNodes();

  bool flag;
  std::vector<std::string> regions;

  for (int i = 0; i < childern->getLength(); i++) {
    DOMNode* inode = childern->item(i);
    if (DOMNode::ELEMENT_NODE == inode->getNodeType()) {
      DOMNamedNodeMap* attr_map = inode->getAttributes();

      DOMNode* node = getUniqueElementByTagNames_(inode, "assigned_regions", flag);
      if (flag) {
        char* text_content = XMLString::transcode(node->getTextContent());
        std::vector<std::string> names = CharToStrings_(text_content);
        XMLString::release(&text_content);

        for (int i = 0; i < names.size(); i++) {
          regions.push_back(names[i]);
        } 
      }
    }
  }
  tmp_list.set<Teuchos::Array<std::string> >("region list", regions);
  
  return out_list;
}

}  // namespace AmanziInput
}  // namespace Amanzi
