/*
  This is the input component of the Amanzi code. 

  Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// TPLs
#include "ParmParse.H"
#include "UnitTest++.h"

// Amanzi
#include "InputConverterS.hh"


/* **************************************************************** */
TEST(CONVERTER_S) {
  using namespace Amanzi;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) std::cout << "Test: convert v2.x -> ParmParse test" << std::endl;

  // read parameter list
  std::string xmlFileName = "test/test_converter_s.xml";

  Amanzi::AmanziInput::InputConverterS converter;
  converter.Init(xmlFileName);
  try {
    // Translate the input. This produces a singleton instance of ParmParse that is 
    // populated with data.
    converter.Translate();

    // Dump the guy to stdout.
    ParmParse pp;
    std::ofstream dumpy("parmparse.dump");
    pp.dumpTable(dumpy);

    std::cout << "Successful translation. Validating the result...\n\n";

  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}
	
