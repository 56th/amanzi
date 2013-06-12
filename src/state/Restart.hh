#ifndef _RESTART_HPP_
#define _RESTART_HPP_

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Epetra_Comm.h"
#include "State.hh"
#include "HDF5MPIMesh.hh"

namespace Amanzi {

  class Restart : public Teuchos::VerboseObject<Restart>
  {

  public:
    Restart(Teuchos::ParameterList& plist, Epetra_MpiComm *comm); 
    Restart(Epetra_MpiComm *comm); // this object will not create any output 
    ~Restart();
   
    void dump_state (State& S, bool force = false);
    void read_state (State& S, std::string& filename);
    void create_files ();
    void read_parameters(Teuchos::ParameterList& plist);
    bool dump_requested(int cycle);

  private:    
    std::string filebasename; 
    int filenamedigits;

    Teuchos::ParameterList plist;
    
    int restart_cycle;
    int number_of_cycle_intervals;

    int interval;
    int start;
    int end;
    
    Teuchos::Array<int> steps;

    Amanzi::HDF5_MPI *restart_output; 

    // disable restart dumps alltogether
    bool disabled;

    // write data for walkabout
    bool walkabout;

    // the Epetra communicator
    Epetra_MpiComm *comm;
  };

}
#endif
