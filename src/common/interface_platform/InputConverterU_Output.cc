/*
  This is the input component of the Amanzi code. 

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Erin Barker (original version)
           Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/bind.hpp>
#include <boost/algorithm/string.hpp>

#include "errors.hh"
#include "exceptions.hh"
#include "dbc.hh"

#define  BOOST_FILESYTEM_NO_DEPRECATED
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/format.hpp"
#include "boost/lexical_cast.hpp"

// TPLs
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "xercesc/dom/DOM.hpp"
#include "xercesc/parsers/DOMLSParserImpl.hpp"

// Amanzi's
#include "ErrorHandler.hpp"
#include "InputConverterU.hh"


namespace Amanzi {
namespace AmanziInput {

XERCES_CPP_NAMESPACE_USE

/* ******************************************************************
* Empty
****************************************************************** */
Teuchos::ParameterList InputConverterU::TranslateOutput_()
{
  Teuchos::ParameterList out_list;

  if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH)
    *vo_->os() << "Translating output" << std::endl;
  Teuchos::OSTab tab = vo_->getOSTab();

  MemoryManager mm;

  char *tagname, *text;
  DOMNamedNodeMap* attr_map;
  DOMNodeList *node_list, *children;
  DOMNode* node;

  // get definitions node - this node MAY exist ONCE
  // this contains any time macros and cycle macros
  Teuchos::ParameterList tmPL, cmPL;
  node_list = doc_->getElementsByTagName(mm.transcode("macros"));
  children = node_list->item(0)->getChildNodes();

  for (int i = 0; i < children->getLength(); ++i) {
    DOMNode* inode = children->item(i);
    if (DOMNode::ELEMENT_NODE == inode->getNodeType()) {
      tagname = mm.transcode(inode->getNodeName());

      // process time macros
      if (strcmp(tagname, "time_macro") == 0) {
        Teuchos::ParameterList tm_parameter;
        std::string name = GetAttributeValueS_(static_cast<DOMElement*>(inode), "name");

        // deal differently if "times" or "start-period-stop"
        bool flag;
        std::string child_name;
        std::vector<DOMNode*> multi_list = GetSameChildNodes_(inode, child_name, flag, false);

        if (child_name == "time") {
          std::vector<double> times;
          for (int j = 0; j < multi_list.size(); j++) {
            DOMNode* jnode = multi_list[j];
            if (DOMNode::ELEMENT_NODE == jnode->getNodeType()) {
              char* text = mm.transcode(jnode->getTextContent());
              times.push_back(TimeCharToValue_(text));
            }
          }
          tm_parameter.set<Teuchos::Array<double> >("values", times);
        } else {
          DOMElement* element = static_cast<DOMElement*>(inode);
          DOMNodeList* list = element->getElementsByTagName(mm.transcode("start"));
          node = list->item(0);

          char* text = mm.transcode(node->getTextContent());
          Teuchos::Array<double> sps;
          sps.append(TimeCharToValue_(text));

          list = element->getElementsByTagName(mm.transcode("timestep_interval"));
          if (list->getLength() > 0) {
            text = mm.transcode(list->item(0)->getTextContent());
            sps.append(TimeCharToValue_(text));

            list = element->getElementsByTagName(mm.transcode("stop"));
            if (list->getLength() > 0) {
              text = mm.transcode(list->item(0)->getTextContent());
              sps.append(TimeCharToValue_(text));
            } else {
              sps.append(-1.0);
            }
            tm_parameter.set<Teuchos::Array<double> >("sps", sps);
          } else {
            tm_parameter.set<Teuchos::Array<double> >("values", sps);
          }
        }
        tmPL.sublist(name) = tm_parameter;
      }

      // process cycle macros
      else if (strcmp(tagname,"cycle_macro") == 0) {
        Teuchos::ParameterList cm_parameter;
        std::string name = GetAttributeValueS_(static_cast<DOMElement*>(inode), "name");      

        DOMElement* element = static_cast<DOMElement*>(inode);
        DOMNodeList* list = element->getElementsByTagName(mm.transcode("start"));
        node = list->item(0);

        text = mm.transcode(node->getTextContent());
        Teuchos::Array<int> sps;
        sps.append(std::strtol(text, NULL, 10));

        list = element->getElementsByTagName(mm.transcode("timestep_interval"));
        if (list->getLength() > 0) {
          text = mm.transcode(list->item(0)->getTextContent());
          sps.append(std::strtol(text, NULL, 10));

          list = element->getElementsByTagName(mm.transcode("stop"));
          if (list->getLength() > 0) {
            text = mm.transcode(list->item(0)->getTextContent());
            sps.append(std::strtol(text, NULL, 10));
          } else {
            sps.append(-1);
          }
          cm_parameter.set<Teuchos::Array<int> >("sps", sps);
        } else {
          cm_parameter.set<Teuchos::Array<int> >("values", sps);
        }
        cmPL.sublist(name) = cm_parameter;
      }
    }
  }
  if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH)
    *vo_->os() << "Found " << cmPL.numParams() << " cycle macros and "
               << tmPL.numParams() << " time macros." << std::endl;

  // get output->vis node - this node must exist ONCE
  bool flag;
  node = GetUniqueElementByTagsString_("output, vis", flag);

  if (flag && node->getNodeType() == DOMNode::ELEMENT_NODE) {
    if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH) {
      *vo_->os() << "Translating output: visualization" << std::endl;
    }

    DOMNodeList* children = node->getChildNodes();
    Teuchos::ParameterList visPL;
    for (int j = 0; j < children->getLength(); j++) {
      DOMNode* jnode = children->item(j);
      tagname = mm.transcode(jnode->getNodeName());

      if (strcmp(tagname, "base_filename") == 0) {
        text = mm.transcode(jnode->getTextContent());
        visPL.set<std::string>("file name base", TrimString_(text));
      } else if (strcmp(tagname, "cycle_macros") == 0 ||
                 strcmp(tagname, "cycle_macro") == 0) {
        text = mm.transcode(jnode->getTextContent());
        ProcessMacros_("cycles", text, cmPL, visPL);
      } else if (strcmp(tagname, "time_macros") == 0 ||
                 strcmp(tagname, "time_macro") == 0) {
        text = mm.transcode(jnode->getTextContent());
        ProcessMacros_("times", text, tmPL, visPL);
      } else if (strcmp(tagname, "write_regions") == 0) {
        DOMNodeList* kids = jnode->getChildNodes();
        for (int k = 0; k < kids->getLength(); ++k) { 
          DOMNode* knode = kids->item(k);
          if (knode->getNodeType() != DOMNode::ELEMENT_NODE) continue;

          DOMElement* element = static_cast<DOMElement*>(knode);
          std::string name = GetAttributeValueS_(element, "name");
          std::string regs = GetAttributeValueS_(element, "regions");
          std::vector<std::string> regions = CharToStrings_(regs.c_str());
          visPL.sublist("write regions").set<Teuchos::Array<std::string> >(name, regions);
        }
      }
    }
    out_list.sublist("Visualization Data") = visPL;
  }

  // get output->checkpoint node - this node must exist ONCE
  node = GetUniqueElementByTagsString_("output, checkpoint", flag);

  if (flag && node->getNodeType() == DOMNode::ELEMENT_NODE) {
    if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH)
      *vo_->os() << "Translating output: checkpoint" << std::endl;

    Teuchos::ParameterList chkPL;
    DOMNodeList* children = node->getChildNodes();
    for (int j = 0; j < children->getLength(); j++) {
      DOMNode* jnode = children->item(j) ;
      tagname = mm.transcode(jnode->getNodeName());
      text = mm.transcode(jnode->getTextContent());

      if (strcmp(tagname, "base_filename") == 0) {
        chkPL.set<std::string>("file name base", TrimString_(text));
      }
      else if (strcmp(tagname, "num_digits") == 0) {
        chkPL.set<int>("file name digits", std::strtol(text, NULL, 10));
      } else if (strcmp(tagname, "cycle_macros") == 0 ||
                 strcmp(tagname, "cycle_macro") == 0) {
        ProcessMacros_("cycles", text, cmPL, chkPL);
      }
    }
    out_list.sublist("Checkpoint Data") = chkPL;
  }

  // get output->walkabout node - this node must exist ONCE
  node = GetUniqueElementByTagsString_("output, walkabout", flag);

  if (flag && node->getNodeType() == DOMNode::ELEMENT_NODE) {
    if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH)
      *vo_->os() << "Translating output: walkabout" << std::endl;

    Teuchos::ParameterList chkPL;
    DOMNodeList* children = node->getChildNodes();
    for (int j = 0; j < children->getLength(); j++) {
      DOMNode* jnode = children->item(j);
      tagname = mm.transcode(jnode->getNodeName());
      text = mm.transcode(jnode->getTextContent());

      if (strcmp(tagname, "base_filename") == 0) {
        chkPL.set<std::string>("file name base", TrimString_(text));
      } else if (strcmp(tagname, "num_digits") == 0) {
        chkPL.set<int>("file name digits", std::strtol(text, NULL, 10));
      } else if (strcmp(tagname, "cycle_macros") == 0 ||
                 strcmp(tagname, "cycle_macro") == 0) {
        ProcessMacros_("cycles", text, cmPL, chkPL);
      }
    }
    out_list.sublist("Walkabout Data") = chkPL;
  }

  // get output->observations node - this node must exist ONCE
  node = GetUniqueElementByTagsString_("output, observations", flag);

  if (flag && node->getNodeType() == DOMNode::ELEMENT_NODE) {
    if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH)
      *vo_->os() << "Translating output: observations" << std::endl;

    Teuchos::ParameterList obsPL;
    DOMNodeList* OBList = node->getChildNodes();

    for (int i = 0; i < OBList->getLength(); i++) {
      DOMNode* inode = OBList->item(i);
      if (inode->getNodeType() != DOMNode::ELEMENT_NODE) continue;
      tagname = mm.transcode(inode->getNodeName());
      text = mm.transcode(inode->getTextContent());

      if (strcmp(tagname, "filename") == 0) {
        obsPL.set<std::string>("Observation Output Filename", TrimString_(text));
      } else if (strcmp(tagname, "liquid_phase") == 0) {
        attr_map = inode->getAttributes();
        node = attr_map->getNamedItem(mm.transcode("name"));
        std::string phaseName;

        if (node) {
          char* text_content = mm.transcode(node->getNodeValue());
          phaseName = std::string(text_content);
          if (phaseName=="water") {
            phaseName = "Water";
          }
        } else {
          ThrowErrorMissattr_("observations", "attribute", "name", "liquid_phase");
        }

        // loop over observations
        DOMNodeList* children = inode->getChildNodes();
        for (int j = 0; j < children->getLength(); j++) {
          Teuchos::ParameterList obPL;
          DOMNode* jnode = children->item(j);
          if (jnode->getNodeType() != DOMNode::ELEMENT_NODE) continue;
          char* obs_type = mm.transcode(jnode->getNodeName());

          if (strcmp(obs_type, "aqueous_pressure") == 0) {
            obPL.set<std::string>("variable", "Aqueous pressure");
          } else if (strcmp(obs_type, "integrated_mass") == 0) {
            // TODO: can't find matching version
          } else if (strcmp(obs_type, "volumetric_water_content") == 0) {
            obPL.set<std::string>("variable", "Volumetric water content");
          } else if (strcmp(obs_type, "gravimetric_water_content") == 0) {
            obPL.set<std::string>("variable", "Gravimetric water content");
          } else if (strcmp(obs_type, "x_aqueous_volumetric_flux") == 0) {
            obPL.set<std::string>("variable", "X-Aqueous volumetric flux");
          } else if (strcmp(obs_type, "y_aqueous_volumetric_flux") == 0) {
            obPL.set<std::string>("variable", "Y-Aqueous volumetric flux");
          } else if (strcmp(obs_type, "z_aqueous_volumetric_flux") == 0) {
            obPL.set<std::string>("variable", "Z-Aqueous volumetric flux");
          } else if (strcmp(obs_type, "material_id") == 0) {
            obPL.set<std::string>("variable", "MaterialID");
          } else if (strcmp(obs_type, "hydraulic_head") == 0) {
            obPL.set<std::string>("variable", "Hydraulic Head");
          } else if (strcmp(obs_type, "aqueous_mass_flow_rate") == 0) {
            obPL.set<std::string>("variable", "Aqueous mass flow rate");
          } else if (strcmp(obs_type, "aqueous_volumetric_flow_rate") == 0) {
            obPL.set<std::string>("variable", "Aqueous volumetric flow rate");
          } else if (strcmp(obs_type, "aqueous_saturation") == 0) {
            obPL.set<std::string>("variable", "Aqueous saturation");
          } else if (strcmp(obs_type, "aqueous_conc") == 0) {
            std::string solute_name = GetAttributeValueS_(static_cast<DOMElement*>(jnode), "solute");
            std::stringstream name;
            name<< solute_name << " Aqueous concentration";
            obPL.set<std::string>("variable", name.str());
          } else if (strcmp(obs_type, "drawdown") == 0) {
            obPL.set<std::string>("variable", "Drawdown");
          } else if (strcmp(obs_type, "solute_volumetric_flow_rate") == 0) {
            std::string solute_name = GetAttributeValueS_(static_cast<DOMElement*>(jnode), "solute");
            std::stringstream name;
            name << solute_name << " volumetric flow rate";
            obPL.set<std::string>("variable", name.str());
          }

          DOMNodeList* kids = jnode->getChildNodes();
          for (int k = 0; k < kids->getLength(); k++) {
            DOMNode* knode = kids->item(k);
            if (DOMNode::ELEMENT_NODE == knode->getNodeType()) {
              char* elem = mm.transcode(knode->getNodeName());
              char* value = mm.transcode(knode->getTextContent());

              if (strcmp(elem, "assigned_regions") == 0) {
                obPL.set<std::string>("region", TrimString_(value));
                vv_obs_regions_.push_back(TrimString_(value));
              } else if (strcmp(elem, "functional") == 0) {
                if (strcmp(value, "point") == 0) {
                  obPL.set<std::string>("functional", "Observation Data: Point");
                } else if (strcmp(value, "integral") == 0) {
                  obPL.set<std::string>("functional", "Observation Data: Integral");
                } else if (strcmp(value, "mean") == 0) {
                  obPL.set<std::string>("functional", "Observation Data: Mean");
                }
              // Keeping singular macro around to help users. This will go away
              } else if (strcmp(elem, "time_macros") == 0 ||
                       strcmp(elem, "time_macro") == 0) {
                ProcessMacros_("times", value, tmPL, obPL);
              } else if (strcmp(elem, "cycle_macros") == 0 ||
                         strcmp(elem, "cycle_macro") == 0) {
                ProcessMacros_("cycles", value, cmPL, obPL);
              }
            }
          }
          std::stringstream list_name;
          list_name << "observation-" << j + 1 << ":" << phaseName;
          obsPL.sublist(list_name.str()) = obPL;
        }

        out_list.sublist("Observation Data") = obsPL;

        if (vo_->getVerbLevel() >= Teuchos::VERB_HIGH)
          *vo_->os() << "Found " << children->getLength() << " observations" << std::endl;
      }
    }
  }

  return out_list;
}


/* ******************************************************************
* Converts macros and macro to multiple SPS parameters orthe single
* values parameter.
****************************************************************** */
void InputConverterU::ProcessMacros_(
    const std::string& prefix, char* text_content,
    Teuchos::ParameterList& mPL, Teuchos::ParameterList& outPL)
{
  std::vector<std::string> macro = CharToStrings_(text_content);
  Teuchos::Array<int> cycles;
  Teuchos::Array<double> times;

  std::list<int> cm_list;
  std::list<double> tm_list;

  int k(0);
  bool flag(false);
  for (int i = 0; i < macro.size(); i++) {
    Teuchos::ParameterList& mlist = mPL.sublist(macro[i]);
    if (mlist.isParameter("sps")) {
      std::stringstream ss;
      ss << prefix << " start period stop " << k;
      k++;
      if (prefix == "cycles") {
        outPL.set<Teuchos::Array<int> >(ss.str(), mlist.get<Teuchos::Array<int> >("sps"));
      } else {
        outPL.set<Teuchos::Array<double> >(ss.str(), mlist.get<Teuchos::Array<double> >("sps"));
      }
    } else if (mlist.isParameter("values")) {
      flag = true;
      if (prefix == "cycles") {
        cycles = mlist.get<Teuchos::Array<int> >("values");
        for (int n = 0; n < cycles.size(); n++) cm_list.push_back(cycles[n]);
      } else {
        times = mlist.get<Teuchos::Array<double> >("values");
        for (int n = 0; n < times.size(); n++) tm_list.push_back(times[n]);
      }
    }
  }

  if (flag) {
    if (prefix == "cycles") {
      cm_list.sort();
      cm_list.unique();

      cycles.clear();
      for (std::list<int>::iterator it = cm_list.begin(); it != cm_list.end(); ++it) 
          cycles.push_back(*it);

      outPL.set<Teuchos::Array<int> >("values", cycles);
    } else {
      tm_list.sort();
      tm_list.unique();

      times.clear();
      for (std::list<double>::iterator it = tm_list.begin(); it != tm_list.end(); ++it) 
          times.push_back(*it);

      outPL.set<Teuchos::Array<double> >("times", times);
    }
  }
}

}  // namespace AmanziInput
}  // namespace Amanzi
