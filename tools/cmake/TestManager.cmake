# -*- mode: cmake -*-

#
# Functions for managing tests.
#

include(CMakeParseArguments)
include(PrintVariable)

function(_APPEND_TEST_LABEL test_name label)
  get_test_property(${test_name} LABELS current_labels)
  if (current_labels)
    set_tests_properties(${test_name} PROPERTIES LABELS "${current_labels};${label}")
  else()  
    set_tests_properties(${test_name} PROPERTIES LABELS "${label}")
  endif()
endfunction(_APPEND_TEST_LABEL)


function(_ADD_TEST_KIND_LABEL test_name kind_in)
  set(kind_prefixes UNIT INT REG AMANZI)

  string(TOUPPER "${kind_in}" kind)

  foreach(kind_prefix ${kind_prefixes})
    string(REGEX MATCH "${kind_prefix}" match ${kind})
    if(match)
      break()
    endif()
  endforeach()

 if (match)
    _append_test_label(${test_name} ${match})
  else()
    message(FATAL_ERROR "Invalid test label ${kind_in} (Valid Labels:${kind_prefixes})")
  endif()
endfunction(_ADD_TEST_KIND_LABEL)


# Usage:
#
# ADD_AMANZI_TEST(<test_name> <test_executable>
#                  [arg1 ...]
#                  KIND [unit | int | reg | AMANZI ]
#                  [AMANZI_INPUT file.xml]
#                  [SOURCE file1 file2  ...]
#                  [LINK_LIBS lib1 lib2 ...]
#                  [DEPENDS test1 test2 ...]
#                  [PARALLEL] [EXPECTED_FAIL]
#                  [NPROCS procs1 ... ]
#                  [MPI_EXEC_ARGS arg1 ... ])

#
# Arguments:
#  test_name: the name given to the resulting test in test reports
#  test_executable: The test executable which performs the test. 
#                   Required if KIND is unit, int or reg
#  arg1 ...: Additional arguments for the test executable
#
# Keyword KIND is required and should be one of unit, int, reg or AMANZI.
#         AMANZI is a special case where the test executable is
#         set to the main Amanzi binary.
#
# KEYWORD AMANZI_INPUT is required if keyword KIND is set to AMANZI. This
#         keyword defines the Amanzi XML input file.
#
# Option PARALLEL signifies that this is a parallel job. This is also
# implied by an NPROCS value > 1
#
# Optional NPROCS keyword starts a list of the number of processors to
# run the test on. Defaults to 1.
#
# Optional MPI_EXEC_ARGS keyword denotes extra arguments to give to
# mpi. It is ignored for serial tests.
#
# Optional SOURCE_FILES keyword that defines a list of source files
# required to build test_executable. An add_executable call will be made
# if this option is active.
#
# Optional LINK_LIBS keyword defines a list of link libraries or link options
# to link test_executable. An target_link_libraries call will be made if
# this option is active.
#
# Optional DEPENDS keyword defines a list of tests that should finish before
# test_name

function(ADD_AMANZI_TEST test_name)
  # --- Initialize 

  # Check test_name
  if ( NOT test_name )
    message(FATAL_ERROR "Must define a test name.")
  endif()

  # Parse through the remaining options
  set(options PARALLEL EXPECTED_FAIL)
  set(oneValueArgs KIND AMANZI_INPUT)
  set(multiValueArgs NPROCS SOURCE LINK_LIBS DEPENDS MPI_EXEC_ARGS)
  cmake_parse_arguments(AMANZI_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # --- Check options

  # Require a KIND value
  if ( NOT AMANZI_TEST_KIND )
    message(FATAL_ERROR "A test type has not been specified for ${test_name}.")
  endif()

  # Force each test to parallel run if mpiexec is required
  if(TESTS_REQUIRE_MPIEXEC)
    set(AMANZI_TEST_PARALLEL TRUE)
  endif()

  # Force each PARALLEL TRUE if NPROCS set 
  if(AMANZI_TEST_NPROCS AND ( "${AMANZI_TEST_NPROCS}" GREATER 1 ) )
    set(AMANZI_TEST_PARALLEL TRUE)
  endif()  

  # Default to nprocs=1 when running parallel
  if ( AMANZI_TEST_PARALLEL AND (NOT AMANZI_TEST_NPROCS) )
    set(AMANZI_TEST_NPROCS 1)
  endif() 

  # Test the value of number of procs value
  if(AMANZI_TEST_NPROCS)
    if(NOT ("${AMANZI_TEST_NPROCS}" GREATER 0) )
      message(FATAL_ERROR "${AMANZI_TEST_NPROCS} is an invalid NPROCS value.")
    endif()

    if(MPI_EXEC_MAX_NUMPROCS AND AMANZI_TEST_PARALLEL)
      if ( "${MPI_EXEC_MAX_NUMPROCS}" LESS "${AMANZI_TEST_NPROCS}")
        message(WARNING "Test ${test_name} request too many nprocs (${AMANZI_TEST_NPROCS}). "
                        "Will skip this test.")
        return()
      endif()
    endif()
  endif()

  # --- Define the test executable

  if ( "${AMANZI_TEST_KIND}" MATCHES "AMANZI" )

    # In this case, we need the Amanzi target definition
    if (NOT TARGET amanzi )
      message(FATAL_ERROR "Can not define an Amanzi test before defining Amanzi binary")
    endif()  

    get_target_property(base amanzi OUTPUT_NAME)
    get_target_property(dir  amanzi OUTPUT_DIRECTORY)
    #set(test_exec "${dir}/${base}")
    set(test_exec "${SSC_BINARY_DIR}/${base}")
   
  else() 
   
    # Do not set if this variable is empty
    if ( AMANZI_TEST_UNPARSED_ARGUMENTS )
      list(GET AMANZI_TEST_UNPARSED_ARGUMENTS 0 test_exec)
      list(REMOVE_AT AMANZI_TEST_UNPARSED_ARGUMENTS 0)
    endif()  

    # Throw an error if test_exec is not defined
    if ( NOT test_exec )
      message(FATAL_ERROR "Must define a test executable for ${test_name}")
    endif()  

    # Create the executable if SOURCE is defined
    if(AMANZI_TEST_SOURCE)
      add_executable(${test_exec} ${AMANZI_TEST_SOURCE})
      set_target_properties(${test_exec} PROPERTIES COMPILE_FLAGS "-DCMAKE_BINARY_DIR=\\\"${CMAKE_BINARY_DIR}\\\" -DCMAKE_SOURCE_DIR=\\\"${CMAKE_SOURCE_DIR}\\\"")
    endif()

    # Add link libraries if needed
    if(AMANZI_TEST_LINK_LIBS)
      target_link_libraries(${test_exec} ${AMANZI_TEST_LINK_LIBS})
    endif()
  endif()  

  
  # --- Define the test arguments

  set(test_args "${AMANZI_TEST_UNPARSED_ARGUMENTS}")
  if ( "${AMANZI_TEST_KIND}" MATCHES "AMANZI" )
    
    # In this case, we need an Amanzi input file
    if ( NOT AMANZI_TEST_AMANZI_INPUT )
      message(FATAL_ERROR "Amanzi tests require an Amanzi input file")
    endif()

    set(test_args "--xml_file=${AMANZI_TEST_AMANZI_INPUT};${test_args}")
  endif()  

  # --- Add test

  # Adjust the execuable name if NOT fullpath AND TESTS_REQUIRE_FULLPATH is set
  if ( TESTS_REQUIRE_FULLPATH )
    if ( NOT ("${test_exec}" MATCHES "^/") )
      set(_tmp      "${CMAKE_CURRENT_BINARY_DIR}/${test_exec}")
      set(test_exec "${_tmp}")
    endif()  
  endif()

  # Construct the test execution command
  set(add_test_exec)
  set(add_test_args)
  if (AMANZI_TEST_PARALLEL)

    if ( MPI_EXEC_GLOBAL_ARGS )
      separate_arguments(global_mpi_args UNIX_COMMAND "${MPI_EXEC_GLOBAL_ARGS}")
    endif() 

    set(add_test_exec ${MPI_EXEC})
    set(add_test_args
                      ${MPI_EXEC_NUMPROCS_FLAG}
                      ${AMANZI_TEST_NPROCS}
                      ${global_mpi_args}
                      ${AMANZI_TEST_MPI_EXEC_ARGS}
                      ${MPI_EXEC_PREFLAGS}
                      ${test_exec}
                      ${MPI_EXEC_POSTFLAGS}
                      ${test_args})
  else()
    set(add_test_exec ${test_exec})
    set(add_test_args ${test_args})
  endif()

  # Call add_test
  add_test(NAME ${test_name} COMMAND ${add_test_exec} ${add_test_args})

  # --- Add test properties

  # Labels
  _add_test_kind_label(${test_name} ${AMANZI_TEST_KIND})
  if ( AMANZI_TEST_PARALLEL AND AMANZI_TEST_NPROCS )
    if ( ${AMANZI_TEST_NPROCS} GREATER 1 )
      _append_test_label(${test_name} PARALLEL)
    else()
      _append_test_label(${test_name} SERIAL)
    endif()  
  else()  
    _append_test_label(${test_name} SERIAL)
  endif()

  # Add test dependencies
  if ( AMANZI_TEST_DEPENDS )
    set_tests_properties(${test_name} PROPERTIES
                         DEPENDS "${AMANZI_TEST_DEPENDS}")
  endif()		       
  
  # Remaining properties are single valued. Building 
  # test_properties as a list should get past the CMake parser.

  # Timeout
  if ( TESTS_TIMEOUT_THRESHOLD )
    list(APPEND test_properties TIMEOUT ${TESTS_TIMEOUT_THRESHOLD})
  endif()

  # CTest needs to know how procs this test needs
  if ( AMANZI_TEST_PARALLEL )
    list(APPEND test_properties PROCESSORS ${AMANZI_TEST_NPROCS})
  endif()

  # Set expected failure flag
  if ( AMANZI_TEST_EXPECTED_FAIL )
     list(APPEND test_properties WILL_FAIL TRUE)
  endif() 

  if ( test_properties )
    set_tests_properties(${test_name} PROPERTIES ${test_properties})
  endif()

endfunction(ADD_AMANZI_TEST)

# Usage:
#
# ADD_AMANZI_SMOKE_TEST(<test_name> 
#                       INPUT file.xml
#                       [FILES file1;file2;...;fileN]
#                       [PARALLEL] 
#                       [NPROCS procs1 ... ]
#                       [MPI_EXEC_ARGS arg1 ... ])

#
# Arguments:
#  test_name: the name given to the comparison test 
#
# KEYWORD INPUT is required. This keyword defines the Amanzi XML input file or 
#         observation file.
#
# Option FILES lists any additional files that the test needs to run in its 
# directory/environment. These files will be copied from the source directory
# to the run directory.
#
# Option PARALLEL signifies that this is a parallel job. This is also
# implied by an NPROCS value > 1
#
# Optional NPROCS keyword starts a list of the number of processors to
# run the test on. Defaults to 1.
#
# Optional MPI_EXEC_ARGS keyword denotes extra arguments to give to
# mpi. It is ignored for serial tests.

function(ADD_AMANZI_SMOKE_TEST test_name)
  # Check test_name
  if ( NOT test_name )
    message(FATAL_ERROR "Must define a test name.")
  endif()

  # Parse through the remaining options
  set(options PARALLEL)
  set(oneValueArgs INPUT FILES)
  set(multiValueArgs NPROCS MPI_EXEC_ARGS)
  cmake_parse_arguments(AMANZI_COMPARISON_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Set up the Amanzi input.
  if ( NOT AMANZI_COMPARISON_TEST_INPUT )
    message(FATAL_ERROR "Smoke tests require an Amanzi input file")
  endif()

  # Assemble the arguments we will pass to add_amanzi_test.
  set(amanzi_test_args "")
  if (AMANZI_COMPARISON_TEST_PARALLEL)
    set(amanzi_test_args ${amanzi_test_args} PARALLEL)
  endif()
  if (AMANZI_COMPARISON_TEST_NPROCS)
    set(amanzi_test_args ${amanzi_test_args} NPROCS ${AMANZI_COMPARISON_TEST_NPROCS})
  endif()
  if (AMANZI_COMPARISON_TEST_MPI_EXEC_ARGS)
    set(amanzi_test_args ${amanzi_test_args} MPI_EXEC_ARGS ${AMANZI_COMPARISON_TEST_MPI_EXEC_ARGS})
  endif()

  # Copy input and files into place.
  file(COPY ${PROJECT_SOURCE_DIR}/${AMANZI_COMPARISON_TEST_INPUT} DESTINATION .)
  foreach(f ${AMANZI_COMPARISON_TEST_FILES})
    file(COPY ${PROJECT_SOURCE_DIR}/${f} DESTINATION .)
  endforeach()

  # Call add_test
  add_amanzi_test(run_${test_name}_smoke_test KIND AMANZI AMANZI_INPUT ${AMANZI_COMPARISON_TEST_INPUT} ${amanzi_test_args})
  _append_test_label(run_${test_name}_smoke_test REG)

endfunction(ADD_AMANZI_SMOKE_TEST)

# Usage:
#
# ADD_AMANZI_COMPARISON_TEST(<test_name> 
#                            INPUT file.xml
#                            NORM L1 | L2 | Linf  <--- not supported yet.
#                            REFERENCE reference
#                            TOLERANCE tolerance
#                            [FILES file1;file2;...;fileN]
#                            [FIELD field_name]
#                            [OBSERVATION obs_name] 
#                            [PARALLEL] 
#                            [NPROCS procs1 ... ]
#                            [MPI_EXEC_ARGS arg1 ... ])

#
# Arguments:
#  test_name: the name given to the comparison test 
#
# One of keyword FIELD or OBSERVATION is required and should refer, respectively, to 
#         a field or observation in the given INPUT.
#
# KEYWORD INPUT is required. This keyword defines the Amanzi XML input file or 
#         observation file.
#
# KEYWORD REFERENCE The name of the file containing reference data to which 
#         the simulation output will be compared.

# KEYWORD NORM The maximum L2 error norm that can be measured for a successful
#         testing outcome.

# Option FILES lists any additional files that the test needs to run in its 
# directory/environment. These files will be copied from the source directory
# to the run directory.
#
# Option PARALLEL signifies that this is a parallel job. This is also
# implied by an NPROCS value > 1
#
# Optional NPROCS keyword starts a list of the number of processors to
# run the test on. Defaults to 1.
#
# Optional MPI_EXEC_ARGS keyword denotes extra arguments to give to
# mpi. It is ignored for serial tests.

function(ADD_AMANZI_COMPARISON_TEST test_name)

  # Check test_name
  if ( NOT test_name )
    message(FATAL_ERROR "Must define a test name.")
  endif()

  # Parse through the remaining options
  set(options PARALLEL)
  set(oneValueArgs FIELD OBSERVATION INPUT REFERENCE TOLERANCE FILES)
  set(multiValueArgs NPROCS MPI_EXEC_ARGS)
  cmake_parse_arguments(AMANZI_COMPARISON_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Set up the Amanzi input.
  if ( NOT AMANZI_COMPARISON_TEST_INPUT )
    message(FATAL_ERROR "Comparison tests require an Amanzi input file")
  endif()

  if ( NOT AMANZI_COMPARISON_TEST_FIELD AND NOT AMANZI_COMPARISON_TEST_OBSERVATION )
    message(FATAL_ERROR "Either FIELD or OBSERVATION is required.")
  endif()
  if ( AMANZI_COMPARISON_TEST_FIELD AND AMANZI_COMPARISON_TEST_OBSERVATION )
    message(FATAL_ERROR "Only one of FIELD or OBSERVATION may be given.")
  endif()

  if ( NOT AMANZI_COMPARISON_TEST_REFERENCE)
    message(FATAL_ERROR "REFERENCE must be specified.")
  endif()

  if ( NOT AMANZI_COMPARISON_TEST_TOLERANCE)
    message(FATAL_ERROR "TOLERANCE must be specified.")
  endif()

  # Assemble the arguments we will pass to add_amanzi_test.
  set(amanzi_test_args "")
  if (AMANZI_COMPARISON_TEST_PARALLEL)
    set(amanzi_test_args ${amanzi_test_args} PARALLEL)
  endif()
  if (AMANZI_COMPARISON_TEST_NPROCS)
    set(amanzi_test_args ${amanzi_test_args} NPROCS ${AMANZI_COMPARISON_TEST_NPROCS})
  endif()
  if (AMANZI_COMPARISON_TEST_MPI_EXEC_ARGS)
    set(amanzi_test_args ${amanzi_test_args} MPI_EXEC_ARGS ${AMANZI_COMPARISON_TEST_MPI_EXEC_ARGS})
  endif()

  # Copy input and files into place.
  file(COPY ${PROJECT_SOURCE_DIR}/${AMANZI_COMPARISON_TEST_INPUT} DESTINATION .)
  foreach(f ${AMANZI_COMPARISON_TEST_FILES})
    file(COPY ${PROJECT_SOURCE_DIR}/${f} DESTINATION .)
  endforeach()

  # Call add_test
  add_amanzi_test(run_${test_name}_for_comparison KIND AMANZI AMANZI_INPUT ${AMANZI_COMPARISON_TEST_INPUT} ${amanzi_test_args})
  set_tests_properties(run_${test_name}_for_comparison PROPERTIES FIXTURES_SETUP ${test_name})
  _append_test_label(run_${test_name}_for_comparison REG)
  if (AMANZI_COMPARISON_TEST_FIELD)
    add_test(NAME compare_${test_name}_field COMMAND ${PYTHON_EXECUTABLE} ${AMANZI_SOURCE_DIR}/tools/testing/compare_field_results.py ${AMANZI_COMPARISON_TEST_FIELD} ${AMANZI_COMPARISON_TEST_INPUT} ${AMANZI_COMPARISON_TEST_REFERENCE} ${AMANZI_COMPARISON_TEST_TOLERANCE})
    set_tests_properties(compare_${test_name}_field PROPERTIES FIXTURES_REQUIRED ${test_name})
    set_tests_properties(run_${test_name}_for_comparison compare_${test_name}_field PROPERTIES RESOURCE_LOCK field_${test_name}_db)
    set_tests_properties( compare_${test_name}_field PROPERTIES PASS_REGULAR_EXPRESSION "Comparison Passed" )
    _append_test_label(compare_${test_name}_field REG)
  else()
    add_test(NAME compare_${test_name}_observation COMMAND ${PYTHON_EXECUTABLE} ${AMANZI_SOURCE_DIR}/tools/testing/compare_observation_results.py ${AMANZI_COMPARISON_TEST_OBSERVATION} ${AMANZI_COMPARISON_TEST_INPUT} ${AMANZI_COMPARISON_TEST_REFERENCE} ${AMANZI_COMPARISON_TEST_TOLERANCE})
    set_tests_properties(compare_${test_name}_observation PROPERTIES FIXTURES_REQUIRED ${test_name})
    set_tests_properties(run_${test_name}_for_comparison compare_${test_name}_observation PROPERTIES RESOURCE_LOCK observation_${test_name}_db)
    set_tests_properties( compare_${test_name}_observation PROPERTIES PASS_REGULAR_EXPRESSION "Comparison Passed" )
    _append_test_label(compare_${test_name}_observation REG)
  endif()

endfunction(ADD_AMANZI_COMPARISON_TEST)



