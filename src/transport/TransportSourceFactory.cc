/*
  This is the transport component of the Amanzi code.

  Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided Reconstruction.cppin the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <string>
#include <vector>

#include "errors.hh"

#include "TransportSourceFactory.hh"

namespace Amanzi {
namespace Transport {

/* ******************************************************************
* Process source, step 1.
****************************************************************** */
TransportDomainFunction* TransportSourceFactory::CreateSource() 
{
  Errors::Message msg;
  TransportDomainFunction* src = new TransportDomainFunction(mesh_);

  if (plist_->isSublist("concentration")) {
    Teuchos::ParameterList& clist = plist_->get<Teuchos::ParameterList>("concentration");

    // Iterate through the source specification sublists in the clist.
    // All are expected to be sublists of identical structure.
    for (Teuchos::ParameterList::ConstIterator it = clist.begin(); it != clist.end(); ++it) {
      std::string name = it->first;
      if (clist.isSublist(name)) {
        Teuchos::ParameterList& srclist = clist.sublist(name);
	for (Teuchos::ParameterList::ConstIterator it1 = srclist.begin(); it1 != srclist.end(); ++it1) {
	  std::string specname = it1->first;

	  if (srclist.isSublist(specname)) {
	    Teuchos::ParameterList& spec = srclist.sublist(specname);
            try {
              ProcessSourceSpec(spec, name, src);
            } catch (Errors::Message& m) {
              msg << "in sublist \"" << specname.c_str() << "\": " << m.what();
              Exceptions::amanzi_throw(msg);
            }
          } else {
            msg << "parameter \"" << specname.c_str() << "\" is not a sublist";
            Exceptions::amanzi_throw(msg);
          }
        }
      } else {
        msg << "parameter \"" << name.c_str() << "\" is not a sublist";
        Exceptions::amanzi_throw(msg);
      }
    }
  } else {
    msg << "Transport PK: \"source terms\" has no sublist \"concentration\".\n";
    Exceptions::amanzi_throw(msg);  
  }

  return src;
}


/* ******************************************************************
* Process source, step 2.
****************************************************************** */
void TransportSourceFactory::ProcessSourceSpec(
  Teuchos::ParameterList& list, const std::string& name, TransportDomainFunction* src) const 
{
  Errors::Message m;
  std::vector<std::string> regions;

  if (list.isParameter("regions")) {
    if (list.isType<Teuchos::Array<std::string> >("regions")) {
      regions = list.get<Teuchos::Array<std::string> >("regions").toVector();
    } else {
      m << "parameter \"regions\" is not of type \"Array string\"";
      Exceptions::amanzi_throw(m);
    }
  } else {
    m << "parameter \"regions\" is missing";
    Exceptions::amanzi_throw(m);
  }

  Teuchos::ParameterList* f_list;
  if (list.isSublist("sink")) {
    f_list = &list.sublist("sink");
  } else {
    m << "parameter \"sink\" is not a sublist";
    Exceptions::amanzi_throw(m);
  }

  // Make the source function.
  Teuchos::RCP<MultiFunction> f;
  try {
    f = Teuchos::rcp(new Amanzi::MultiFunction(*f_list));
  } catch (Errors::Message& msg) {
    m << "error in source sublist : " << msg.what();
    Exceptions::amanzi_throw(m);
  }

  // Add this source specification to the domain function.
  int method;
  std::string action_name = list.get<std::string>("spatial distribution method", "none");
  ProcessStringActions(action_name, &method);

  src->Define(regions, f, method, name);
}


/* ****************************************************************
* Process string for a source specipic action.
**************************************************************** */
void TransportSourceFactory::ProcessStringActions(const std::string& name, int* method) const 
{
  Errors::Message msg;
  if (name == "none") {
    *method = TransportActions::DOMAIN_FUNCTION_ACTION_NONE;
  } else if (name == "volume") {
    *method = TransportActions::DOMAIN_FUNCTION_ACTION_DISTRIBUTE_VOLUME;
  } else if (name == "permeability") {
    *method = TransportActions::DOMAIN_FUNCTION_ACTION_DISTRIBUTE_PERMEABILITY;
  } else {
    msg << "Transport PK: unknown source distribution method has been specified.";
    Exceptions::amanzi_throw(msg);
  }
}

}  // namespace Transport
}  // namespace Amanzi
