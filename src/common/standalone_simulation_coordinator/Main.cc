#ifdef ENABLE_Unstructured
#include "AmanziUnstructuredGridSimulationDriver.hh"
#endif
#ifdef ENABLE_Structured
#include "amanzi_structured_grid_simulation_driver.H"
#endif

#include <iostream>

#include <Epetra_Comm.h>
#include <Epetra_MpiComm.h>
#include "Epetra_SerialComm.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/util/OutOfMemoryException.hpp>
#include "DOMTreeErrorReporter.hpp"
#include "ErrorHandler.hpp"
#include "InputTranslator.hh"
//#include "DOMPrintErrorHandler.hpp"
#include "XMLParameterListWriter.hh"

#include "amanzi_version.hh"
#include "dbc.hh"
#include "errors.hh"
#include "exceptions.hh"
#include "TimerManager.hh"
#include "VerboseObject_objs.hh"

// include fenv if it exists
#include "boost/version.hpp"
#if (BOOST_VERSION / 100 % 1000 >= 46)
#include "boost/config.hpp"
#ifndef BOOST_NO_FENV_H
#ifdef _GNU_SOURCE
#define AMANZI_USE_FENV
#include "boost/detail/fenv.hpp"
#endif
#endif
#endif


#ifdef ENABLE_Unstructured
#include "state_evaluators_registration.hh"
#endif

#include "tpl_versions.h"

#include <iostream>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;



struct RunLog
    : public std::ostream
{
  RunLog(std::ostream& _os);
 protected:
  int rank;
};

int main(int argc, char *argv[]) {

#ifdef AMANZI_USE_FENV
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  Teuchos::GlobalMPISession mpiSession(&argc,&argv,0);
  int rank = mpiSession.getRank();

  try {
    Teuchos::CommandLineProcessor CLP;

    CLP.setDocString("The Amanzi driver reads an XML input file and\n"
                     "runs a reactive flow and transport simulation.\n");

    std::string xmlInFileName = "";
    CLP.setOption("xml_file", &xmlInFileName, "XML options file");

    std::string xmlSchema = ""; 
    CLP.setOption("xml_schema", &xmlSchema, "XML Schema File"); 

    bool print_version(false);
    CLP.setOption("print_version", "no_print_version", &print_version, "Print version number and exit.");

    bool print_tpl_versions(false);
    CLP.setOption("print_tplversions", "no_print_tplversions", &print_tpl_versions, "Print version numbers of third party libraries and exit.");

    bool print_all(false);
    CLP.setOption("print_all", "no_print_all", &print_all, "Print all pre-run information.");    

    bool print_paths(false);
    CLP.setOption("print_paths", "no_print_paths", &print_paths, "Print paths of the xml input file and the xml schema file.");    
    
    CLP.throwExceptions(false);
    CLP.recogniseAllOptions(true);

    Teuchos::CommandLineProcessor::EParseCommandLineReturn
        parseReturn = CLP.parse(argc, argv);

    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
      throw std::string("amanzi not run");
    }    
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION) {
      throw std::string("amanzi not run");
    }
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_ERROR) {
      throw std::string("amanzi not run");
    }
    
    if (print_all) {
      print_paths = true;
      print_tpl_versions = true;
      print_version = true;
    }

    // strinigy magic
#define XSTR(s) STR(s)
#define STR(s) #s

    // check for verbose option
    if (print_version) {
      if (rank == 0) {
        std::cout << "Amanzi Version " << XSTR(AMANZI_VERSION) << std::endl;
	std::cout << "HG branch      " << XSTR(AMANZI_HG_BRANCH) << std::endl;
	std::cout << "HG global hash " << XSTR(AMANZI_HG_GLOBAL_HASH) << std::endl;
	std::cout << "HG local id    " << XSTR(AMANZI_HG_LOCAL_ID) << std::endl;
      }
    }

    if (print_tpl_versions) {
      if (rank == 0) {
#ifdef AMANZI_MAJOR
	std::cout << "Amanzi TPL collection version "<<  XSTR(AMANZI_MAJOR) << "." << XSTR(AMANZI_MINOR) << "." << XSTR(AMANZI_PATCH) << std::endl;
#endif
	std::cout << "Third party libraries that this amanzi binary is linked against:" << std::endl;
#ifdef ALQUIMIA_MAJOR
	std::cout << "  ALQUIMIA       " << XSTR(ALQUIMIA_MAJOR) << "." << XSTR(ALQUIMIA_MINOR) << "." << XSTR(ALQUIMIA_PATCH) << std::endl;
#endif
#ifdef ASCEMIO_MAJOR
	std::cout << "  ASCEMIO        " << XSTR(ASCEMIO_MAJOR) << "." << XSTR(ASCEMIO_MINOR) << "." << XSTR(ASCEMIO_PATCH) << std::endl;
#endif
#ifdef Boost_MAJOR
	std::cout << "  Boost          " << XSTR(Boost_MAJOR) << "." << XSTR(Boost_MINOR) << "." << XSTR(Boost_PATCH) << std::endl;
#endif
#ifdef CCSE_MAJOR
	std::cout << "  CCSE           " << XSTR(CCSE_MAJOR) << "." << XSTR(CCSE_MINOR) << "." << XSTR(CCSE_PATCH) << std::endl;
#endif
#ifdef CURL_MAJOR
	std::cout << "  CURL           " << XSTR(CURL_MAJOR) << "." << XSTR(CURL_MINOR) << "." << XSTR(CURL_PATCH) << std::endl;
#endif
#ifdef ExodusII_MAJOR
	std::cout << "  ExodusII       " << XSTR(ExodusII_MAJOR) << "." << XSTR(ExodusII_MINOR) << "." << XSTR(ExodusII_PATCH) << std::endl;
#endif
#ifdef HDF5_MAJOR
	std::cout << "  HDF5           " << XSTR(HDF5_MAJOR) << "." << XSTR(HDF5_MINOR) << "." << XSTR(HDF5_PATCH) << std::endl;
#endif
#ifdef HYPRE_MAJOR
	std::cout << "  HYPRE          " << XSTR(HYPRE_MAJOR) << "." << XSTR(HYPRE_MINOR) << "." << XSTR(HYPRE_PATCH) << std::endl;
#endif
#ifdef METIS_MAJOR
	std::cout << "  METIS          " << XSTR(METIS_MAJOR) << "." << XSTR(METIS_MINOR) << "." << XSTR(METIS_PATCH) << std::endl;	
#endif
#ifdef MOAB_MAJOR
	std::cout << "  MOAB           " << XSTR(MOAB_MAJOR) << "." << XSTR(MOAB_MINOR) << "." << XSTR(MOAB_PATCH) << std::endl;
#endif
#ifdef MSTK_MAJOR 
	std::cout << "  MSTK           " << XSTR(MSTK_MAJOR) << "." << XSTR(MSTK_MINOR) << "." << XSTR(MSTK_PATCH) << std::endl;
#endif
#ifdef NetCDF_MAJOR
	std::cout << "  NetCDF         " << XSTR(NetCDF_MAJOR) << "." << XSTR(NetCDF_MINOR) << "." << XSTR(NetCDF_PATCH) << std::endl;
#endif
#ifdef NetCDF_Fortran_MAJOR
	std::cout << "  NetCDF_Fortran " << XSTR(NetCDF_Fortran_MAJOR) << "." << XSTR(NetCDF_Fortran_MINOR) << "." << XSTR(NetCDF_Fortran_PATCH) << std::endl;
#endif
#ifdef ParMetis_MAJOR
	std::cout << "  ParMetis       " << XSTR(ParMetis_MAJOR) << "." << XSTR(ParMetis_MINOR) << "." << XSTR(ParMetis_PATCH) << std::endl;
#endif
#ifdef PETSc_MAJOR
	std::cout << "  PETSc          " << XSTR(PETSc_MAJOR) << "." << XSTR(PETSc_MINOR) << "." << XSTR(PETSc_PATCH) << std::endl;	
#endif
#ifdef PFLOTRAN_MAJOR
	std::cout << "  PFLOTRAN       " << XSTR(PFLOTRAN_MAJOR) << "." << XSTR(PFLOTRAN_MINOR) << "." << XSTR(PFLOTRAN_PATCH) << std::endl;
#endif
#ifdef SEACAS_MAJOR
	std::cout << "  SEACAS         " << XSTR(SEACAS_MAJOR) << "." << XSTR(SEACAS_MINOR) << "." << XSTR(SEACAS_PATCH) << std::endl;
#endif
#ifdef SuperLU_MAJOR
	std::cout << "  SuperLU        " << XSTR(SuperLU_MAJOR) << "." << XSTR(SuperLU_MINOR) << "." << XSTR(SuperLU_PATCH) << std::endl;
#endif
#ifdef SuperLUDist_MAJOR
	std::cout << "  SuperLUDist    " << XSTR(SuperLUDist_MAJOR) << "." << XSTR(SuperLUDist_MINOR) << "." << XSTR(SuperLUDist_PATCH) << std::endl;
#endif
#ifdef Trilinos_MAJOR
	std::cout << "  Trilinos       " << XSTR(Trilinos_MAJOR) << "." << XSTR(Trilinos_MINOR) << "." << XSTR(Trilinos_PATCH) << std::endl;
#endif
#ifdef UnitTest_MAJOR
	std::cout << "  UnitTest       " << XSTR(UnitTest_MAJOR) << "." << XSTR(UnitTest_MINOR) << "." << XSTR(UnitTest_PATCH) << std::endl;
#endif
#ifdef XERCES_MAJOR
	std::cout << "  XERCES         " << XSTR(XERCES_MAJOR) << "." << XSTR(XERCES_MINOR) << "." << XSTR(XERCES_PATCH) << std::endl;
#endif
#ifdef ZLIB_MAJOR
	std::cout << "  ZLIB           " << XSTR(ZLIB_MAJOR) << "." << XSTR(ZLIB_MINOR) << "." << XSTR(ZLIB_PATCH) << std::endl;
#endif
      }
    }

    if (print_paths) {
      if (rank == 0) {
	std::cout << "xml input file:  " << xmlInFileName << std::endl;
      }
    }

    if (xmlInFileName.size() == 0) {
      if (rank == 0) {
	std::cout << "ERROR: No xml input file was specified. Use the command line option --xml_file to specify one." << std::endl;
      }      
      throw std::string("amanzi not run");      
    }

    // check if the input file actually exists
    if (!exists(xmlInFileName)) {
      if (rank == 0) {
	std::cout << "ERROR: The xml input file " << xmlInFileName << " that was specified using the command line option --xml_file does not exist." << std::endl;
      }
      throw std::string("amanzi not run");
    }


    // EIB - this is the new piece which reads either the new or old input
    /***************************************/
    MPI_Comm mpi_comm(MPI_COMM_WORLD);
    Teuchos::ParameterList driver_parameter_list;
    try {
      xercesc::XMLPlatformUtils::Initialize();
      xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser;
      parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);
      bool errorsOccured = false;
      try{
        parser->parse(xmlInFileName.c_str());
      }
      catch (const xercesc::OutOfMemoryException&)
      {
	std::cerr << "OutOfMemoryException" << std::endl;
        errorsOccured = true;
      }
      xercesc::DOMDocument *doc = parser->getDocument();
      xercesc::DOMElement *root = doc->getDocumentElement();
      char* temp2 = xercesc::XMLString::transcode(root->getTagName());
      //DOMImplementation* impl =  DOMImplementationRegistry::getDOMImplementation(X("Core"));

      if (strcmp(temp2, "amanzi_input") == 0) {
	
	//amanzi_throw(Errors::Message("Translation for new input spec is not yet complete, please use old input spec"));
	driver_parameter_list = Amanzi::AmanziNewInput::translate(xmlInFileName);
	
	//driver_parameter_list.print(std::cout,true,false);
	const Teuchos::ParameterList& echo_list = driver_parameter_list.sublist("Echo Translated Input");
	if (echo_list.isParameter("Format")) {
	  if (echo_list.get<std::string>("Format") == "v1") {
	    std::string new_filename = echo_list.get<std::string>("File Name");
	    if (rank == 0) {
	      printf("Amanzi: writing the translated parameter list to file %s...\n", new_filename.c_str());
	      Teuchos::Amanzi_XMLParameterListWriter XMLWriter;
	      Teuchos::XMLObject XMLobj = XMLWriter.toXML(driver_parameter_list);
	      std::ofstream xmlfile;
	      xmlfile.open(new_filename.c_str());
	      xmlfile << XMLobj;
	    }
	  }
	}
      } else if(strcmp(temp2, "ParameterList") == 0) {
	// Teuchos::ParameterXMLFileReader xmlreader(xmlInFileName);
        // driver_parameter_list = xmlreader.getParameters();
        // new initialization verifies the XML input
        Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::getParametersFromXmlFile(xmlInFileName);
        driver_parameter_list = *plist;
      }
      else {
	amanzi_throw(Errors::Message("Unexpected Error reading input file"));
      }

      // check root tag 
      // if ParameterList - do old and pass thru
      // if amanzi_input  - validate, convert to old

      xercesc::XMLString::release(&temp2) ;
      delete parser;
      xercesc::XMLPlatformUtils::Terminate();
    }
    catch (std::exception& e)
    {
      if (rank == 0) {
        std::cout << e.what() << std::endl;
        std::cout << "Amanzi::XERCES-INPUT_FAILED\n";
      }
      amanzi_throw(Errors::Message("Amanzi::Input - reading and translation of input file failed. ABORTING!"));
    }
    /***************************************/
    // EIB - this is the old stuff I'm replacing
    // *************************************//
    //// read the main parameter list
    //Teuchos::ParameterList driver_parameter_list;
    //// DEPRECATED    Teuchos::updateParametersFromXmlFile(xmlInFileName,&driver_parameter_list);
    //Teuchos::ParameterXMLFileReader xmlreader(xmlInFileName);
    //driver_parameter_list = xmlreader.getParameters();
    // *************************************//

    const Teuchos::ParameterList& mesh_parameter_list = driver_parameter_list.sublist("Mesh");
    driver_parameter_list.set<std::string>("input file name", xmlInFileName);

    // The Mesh list contains a "Structured" sublist or a "Unstructured" sublist, and will
    // determine which simulation driver to call
    std::string framework;
    if (mesh_parameter_list.isSublist("Structured")) {
      framework = "Structured";
    } else if (mesh_parameter_list.isSublist("Unstructured")) {
      framework = "Unstructured";
    } else {
      amanzi_throw(Errors::Message("The Mesh parameter list must contain one sublist: \"Structured\" or \"Unstructured\""));
    }

    Amanzi::Simulator* simulator = NULL;

    Amanzi::timer_manager.add("Full Simulation", Amanzi::Timer::ONCE);
    Amanzi::timer_manager.start("Full Simulation");

    if (framework=="Structured") {
#ifdef ENABLE_Structured
      simulator = new AmanziStructuredGridSimulationDriver();
#else
      amanzi_throw(Errors::Message("Structured not supported in current build"));
#endif
    } else {
#ifdef ENABLE_Unstructured
      simulator = new AmanziUnstructuredGridSimulationDriver();
#else
      amanzi_throw(Errors::Message("Unstructured not supported in current build"));
#endif
    }

    //MPI_Comm mpi_comm(MPI_COMM_WORLD);
    Amanzi::ObservationData output_observations;
    Amanzi::Simulator::ReturnType ret = simulator->Run(mpi_comm, driver_parameter_list, output_observations);

    if (ret == Amanzi::Simulator::FAIL) {
      amanzi_throw(Errors::Message("The amanzi simulator returned an error code, this is most likely due to an error in the mesh creation."));
    }

    Amanzi::timer_manager.stop("Full Simulation");
    Amanzi::timer_manager.parSync(mpi_comm);

    if (rank == 0) {
      std::cout << "Amanzi::SIMULATION_SUCCESSFUL\n\n";
      std::cout << Amanzi::timer_manager << std::endl;
    }
    delete simulator;
  }
  catch (std::string& s) {
    if (rank == 0) {
      if (s == "amanzi not run") {
	std::cout << "Amanzi::SIMULATION_DID_NOT_RUN\n";
      } 
    }
  }
  catch (std::exception& e) {
    if (rank == 0) {
      if (! strcmp(e.what(), "amanzi not run")) {
	std::cout << "Amanzi::SIMULATION_DID_NOT_RUN\n";
      } else {
	std::cout << e.what() << std::endl;
	std::cout << "Amanzi::SIMULATION_FAILED\n";
      }
    }
  }
  catch (int& ierr) {
    if (rank == 0) {
      std::cout << "Catched unknown exception with code " << ierr 
                << ". Known sources: Epetra_MultiVector::AllocateForCopy" << std::endl;
      std::cout << "Amanzi::SIMULATION_FAILED\n";
    }
  }
  
  // catch all
  catch (...) {
    if (rank == 0) {
      std::cout << "Unknown exception" << std::endl;
      std::cout << "Amanzi::SIMULATION_FAILED\n";
    }
  }
}

