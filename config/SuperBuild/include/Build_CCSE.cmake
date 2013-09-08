#  -*- mode: cmake -*-

#
# Build TPL:  CCSE 
#  

# --- Define all the directories and common external project flags
define_external_project_args(CCSE TARGET ccse DEPENDS ${MPI_PROJECT})

# --- Define the CMake configure parameters
# Note:
#      CMAKE_CACHE_ARGS requires -DVAR:<TYPE>=VALUE syntax
#      CMAKE_ARGS -DVAR=VALUE OK
# NO WHITESPACE between -D and VAR. Parser blows up otherwise.
# 
message(STATUS "Build CCSE with space dimension ${CCSE_BL_SPACEDIM}")
set(CCSE_CMAKE_CACHE_ARGS
                       -DENABLE_Config_Report:BOOL=TRUE
                       -DENABLE_MPI:BOOL=TRUE
                       -DENABLE_OpenMP:BOOL=${ENABLE_OpenMP}
                       -DENABLE_TESTS:BOOL=FALSE
                       -DBL_PRECISION:STRING=DOUBLE
                       -DBL_SPACEDIM:INT=${CCSE_BL_SPACEDIM}
                       -DCMAKE_INSTALL_PREFIX:STRING=<INSTALL_DIR>
                       -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
                       -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS})


# --- Set the name of the patch
set(CCSE_patch_file ccse-1.1.7-cgsolver-using.patch)
# --- Configure the bash patch script
set(CCSE_sh_patch ${CCSE_prefix_dir}/ccse-patch-step.sh)
configure_file(${SuperBuild_TEMPLATE_FILES_DIR}/ccse-patch-step.sh.in
               ${CCSE_sh_patch}
               @ONLY)
# --- Configure the CMake command file
set(CCSE_cmake_patch ${CCSE_prefix_dir}/ccse-patch-step.cmake)
configure_file(${SuperBuild_TEMPLATE_FILES_DIR}/ccse-patch-step.cmake.in
               ${CCSE_cmake_patch}
               @ONLY)
# --- Set the patch command
set(CCSE_PATCH_COMMAND ${CMAKE_COMMAND} -P ${CCSE_cmake_patch})     

# --- Add external project build and tie to the CCSE build target
ExternalProject_Add(${CCSE_BUILD_TARGET}
                    DEPENDS   ${CCSE_PACKAGE_DEPENDS}             # Package dependency target
                    TMP_DIR   ${CCSE_tmp_dir}                     # Temporary files directory
                    STAMP_DIR ${CCSE_stamp_dir}                   # Timestamp and log directory
                    # -- Download and URL definitions
                    DOWNLOAD_DIR ${TPL_DOWNLOAD_DIR}              # Download directory
                    URL          ${CCSE_URL}                      # URL may be a web site OR a local file
                    URL_MD5      ${CCSE_MD5_SUM}                  # md5sum of the archive file
                    # -- Patch 
                    PATCH_COMMAND ${CCSE_PATCH_COMMAND}
                    # -- Configure
                    SOURCE_DIR       ${CCSE_source_dir}               # Source directory
                    CMAKE_CACHE_ARGS ${CCSE_CMAKE_CACHE_ARGS}         # CMAKE_CACHE_ARGS or CMAKE_ARGS => CMake configure
                                     -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER_USE}
                                     ${Amanzi_CMAKE_C_COMPILER_ARGS}    # Ensure uniform build
                                     -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER_USE}
                                     ${Amanzi_CMAKE_CXX_COMPILER_ARGS}  # Ensure uniform build
                                     -DCMAKE_Fortran_COMPILER:FILEPATH=${CMAKE_Fortran_COMPILER_USE}
                                     ${Amanzi_CMAKE_Fortran_COMPILER_ARGS}  # Ensure uniform build
                    # -- Build
                    BINARY_DIR        ${CCSE_build_dir}           # Build directory 
                    BUILD_COMMAND     $(MAKE)                     # $(MAKE) enables parallel builds through make
                    BUILD_IN_SOURCE   ${CCSE_BUILD_IN_SOURCE}     # Flag for in source builds
                    # -- Install
                    INSTALL_DIR      ${TPL_INSTALL_PREFIX}        # Install directory
                    # -- Output control
                    ${CCSE_logging_args}) 
