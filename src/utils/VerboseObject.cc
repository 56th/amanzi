/*
  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Ethan Coon

  Basic VerboseObject for use by Amanzi code.  Trilinos's VerboseObject is
  templated with the class (for no reason) and then requests that the
  VerboseObject be inserted as a base class to the using class.  This is serious
  code smell (composition over inheritance, especially for code reuse).
*/

#include "Teuchos_VerboseObjectParameterListHelpers.hpp"
#include "Epetra_MpiComm.h"

#include "VerboseObject.hh"

namespace Amanzi {

VerboseObject::VerboseObject(std::string name, Teuchos::ParameterList& plist) :
    comm_(NULL)
{
  // Set up the default level.
  setDefaultVerbLevel(global_default_level);

  // Options from ParameterList

  // -- Set up the VerboseObject header.
  std::string headername = plist.sublist("VerboseObject").get<std::string>("Name",name);
  plist.sublist("VerboseObject").remove("Name");
  set_name(headername);

  // -- Show the line prefix
  bool no_pre = plist.sublist("VerboseObject").get<bool>("Hide Line Prefix", hide_line_prefix);
  plist.sublist("VerboseObject").remove("Hide Line Prefix");

  // Override from ParameterList.
  Teuchos::readVerboseObjectSublist(&plist,this);

  // out, tab
  out_ = getOStream();
  out_->setShowLinePrefix(!no_pre);
}


void VerboseObject::set_name(std::string name)
{
  std::string header(name);
  if (header.size() > line_prefix_size) {
    header.erase(line_prefix_size);
  } else if (header.size() < line_prefix_size) {
    header.append(line_prefix_size - header.size(), ' ');
  }
  setLinePrefix(header);
}

std::string VerboseObject::color(std::string name) const
{ 
  std::string output("");
#ifdef __linux
  if (name == "red") {
    output = std::string("\033[1;31m");
  } else if (name == "green") {
    output = std::string("\033[1;32m");
  } else if (name == "yellow") {
    output = std::string("\033[1;33m");
  }
#endif
  return output;
}


VerboseObject::VerboseObject(const Epetra_MpiComm* const comm, std::string name,
                             Teuchos::ParameterList& plist) :
    comm_(comm)
{
  int root = -1;
  // Check if we are in the mode of writing only a specific rank.
  if (plist.sublist("VerboseObject").isParameter("Write On Rank")) {
    root = plist.sublist("VerboseObject").get<int>("Write On Rank");
    plist.sublist("VerboseObject").remove("Write On Rank");
  }

  // Init the basics
  // Set up the default level.
  setDefaultVerbLevel(global_default_level);

  // Options from ParameterList

  // -- Set up the VerboseObject header.
  std::string headername = plist.sublist("VerboseObject").get<std::string>("name",name);
  plist.sublist("VerboseObject").remove("name");

  set_name(headername);

  // -- Show the line prefix
  bool no_pre = plist.sublist("VerboseObject").get<bool>("hide line prefix",
          hide_line_prefix);
  plist.sublist("VerboseObject").remove("hide line prefix");

  // Override from ParameterList.
  Teuchos::readVerboseObjectSublist(&plist,this);

  // out, tab
  out_ = getOStream();
  out_->setShowLinePrefix(!no_pre);

  // Set up a local FancyOStream
  if (root >= 0) {
    int size = comm_->NumProc();
    int pid = comm_->MyPID();
    Teuchos::RCP<Teuchos::FancyOStream> newout = Teuchos::rcp(new Teuchos::FancyOStream(out_->getOStream()));
    newout->setProcRankAndSize(pid,size);
    newout->setOutputToRootOnly(root);
    setOStream(newout);

    std::stringstream headerstream;
    headerstream << pid << ": " << getLinePrefix();
    std::string header = headerstream.str();
    if (header.size() > line_prefix_size) {
      header.erase(line_prefix_size);
    } else if (header.size() < line_prefix_size) {
      header.append(line_prefix_size - header.size(), ' ');
    }

    setLinePrefix(header);
    out_ = getOStream();
    out_->setShowLinePrefix(!no_pre);
  }
}

std::string VerboseObject::reset() const
{ 
  std::string output("");
#ifdef __linux
  output = std::string("\033[0m");
#endif
  return output;
}

} // namespace Amanzi
