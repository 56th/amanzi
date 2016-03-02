#!/bin/bash

# ############################################################################ #
#                                                                              #
#   Amanzi Boost Script                                                        #
#                                                                              #
#      Script that builds Amanzi from scratch                                  #
#                                                                              #
# ############################################################################ #

# ---------------------------------------------------------------------------- #
# Initialize
# ---------------------------------------------------------------------------- #

# Logic parameters
TRUE=1
FALSE=0

# System information
system_name=`uname`
system_arch=`uname -m`

# Test script controls
print_exit=${FALSE}

# Known compiler lists
known_c_compilers="mpicc cc gcc icc"
known_cxx_compilers="mpicxx mpiCC CC g++ icpc"
known_fortran_compilers="mpif90 ftn gfortran ifort"

# Directory information
start_directory=$PWD
amanzi_source_dir=$(cd $(dirname "$0")/;pwd)

# ASCEM Web address
ascem_protocol=https
ascem_site='software.lanl.gov/ascem'
ascem_tpl_site="${ascem_site}/tpls"

# Default root build and install prefix
dflt_build_prefix=`pwd`
dflt_install_prefix=`pwd`

# Source and build directories
amanzi_build_dir="${dflt_build_prefix}/build/amanzi"
amanzi_install_prefix="${dflt_install_prefix}/install/amanzi"

# Mercurial
hg_binary=`which hg`

# CURL
curl_binary=`which curl`

# CMake
cmake_binary=`which cmake`
ctest_binary=`which ctest`
cmake_version=2.8.8
cmake_url=http://www.cmake.org/files/v2.8
cmake_archive_file=cmake-${cmake_version}.tar.gz

# Build configuration
parallel_jobs=2
build_type=Release

# Compiler definitions
build_c_compiler=
build_cxx_compiler=
build_fort_compiler=

# Compiler flags
build_c_flags=
build_cxx_flags=
build_fort_flags=
build_link_flags=

# MPI installation
mpi_root_dir=

# TPL (Third Party Libraries)

# Point to a configuration file
tpl_config_file=

# TPL build parameters
tpl_build_dir=${dflt_build_prefix}/build/tpls
tpl_download_dir=${tpl_build_dir}/Downloads
tpl_install_prefix=${dflt_install_prefix}/install/tpls

# Color output display
no_color=$FALSE

# Amanzi build configuration
structured=$TRUE
unstructured=$TRUE
ccse_tools=$FALSE
stk_mesh=$TRUE
mstk_mesh=$TRUE
moab_mesh=$FALSE
amanzi_branch=
test_suite=$FALSE
reg_tests=$FALSE
netcdf4=${TRUE}
petsc=${TRUE}
hypre=${TRUE}
alquimia=${FALSE}
pflotran=${FALSE}
shared=${FALSE}
spacedim=2
native=${FALSE}
nersc=${FALSE}
nersc_tpl_opts=
nersc_amanzi_opts=
#silo=${TRUE}


# ---------------------------------------------------------------------------- #
#
# Begin Functions

# Basic messages and exiting functions
function exit_now()
{
  exit $1
}
function status_message()
{

  local GREEN='32m'
  if [ "${no_color}" -eq "${TRUE}" ]; then
    echo "[`date`] $1"
  else
    echo -e "[`date`]\033[$GREEN $1\033[m"
  fi  

}
function error_message()
{
  local RED='31m'
  if [ "${no_color}" -eq "${TRUE}" ]; then
    echo "Amanzi Bootstrap ERROR: $1" 1>&2
  else  
    echo -e "Amanzi Bootstrap ERROR:\033[$RED $1\033[m" 1>&2 
  fi
}

function warn_message()
{
  local PINK='35m'
  if [ "${no_color}" -eq "${TRUE}" ]; then
    echo "Amanzi Bootstrap Warning: $1" 1>&2
  else  
    echo -e "Amanzi Bootstrap Warning:\033[$PINK $1\033[m" 1>&2
  fi
}

function fatal_message()
{
  error_message $1
  exit_now 10
}

# Utilities

function make_fullpath()
{
  if echo $1 | grep "^\/" > /dev/null 2> /dev/null; then
    echo $1
  else
    echo "$start_directory/$1"
  fi
}

function mkdir_now()
{
  echo "making $1"
  if [ ! -e $1 ]; then
    mkdir -p $1
  fi
}

function download_file()
{
  url=$1
  file=$2
  output=$3

  curl_opts=
  if [ ! -z "${output}" ]; then
    if [ -d "${output}" ]; then
      curl_opts="--output $output/$file"
    else
      curl_opts="--output $output"
    fi
  else
    curl_opts="--remote-name"
  fi
  cmd="${curl_binary} ${curl_opts} $url/$file"
  echo $cmd
  ${curl_binary} $curl_opts $url/$file
  status=$?

  if [ "${status}" -ne "0" ]; then
     error_message "${curl_binary} returned $status"
     error_message "Failed to download $file from $url"
     exit_now $status
  fi

}

# Command line functions
function set_feature()
{
  feature=$1
  action=$2

  if echo $action | grep "disable" > /dev/null 2>/dev/null; then
    eval "${feature}=$FALSE"
  fi
  if echo $action | grep "enable" > /dev/null 2>/dev/null; then
    eval "${feature}=$TRUE"
  fi
  echo ${!feature}

}
function parse_feature()
{
  feature_opt=$1

  if echo ${feature_opt} | grep "^--disable-" > /dev/null 2> /dev/null; then
    echo ${feature_opt} | sed "s/^--disable-//"
  fi
  if echo ${feature_opt} | grep "^--enable-" > /dev/null 2> /dev/null; then
    echo ${feature_opt} | sed "s/^--enable-//"
  fi
    
}

function parse_option_with_equal()
{
  a=$1
  opt_name=$2

  if echo $a | grep "^--${opt_name}=" > /dev/null 2> /dev/null; then
    echo $a | sed "s/^--${opt_name}=//"
  fi

}

function print_usage()
{
echo '
Usage: '"$0"' [options]
Options: [defaults in brackets after descriptions]
Configuration:

  --help                  print this message
  
  --parallel=n            build in parallel, where n is
                          number of maximum make jobs ['"${parallel_jobs}"']

  --no-color              deactivate color-coded status output ['"${no_color}"']                        

  --tpl-config-file=FILE  define a CMake TPL configuration file. If this
                          option is selected, '"$0"' will NOT build the TPLs.

  --opt                   build optimized TPLs and Amanzi binaries. This the default
                          configuration.

  --debug                 build debug TPLs and Amanzi binaries.

  --branch=BRANCH         build TPLs and Amanzi found in BRANCH ['"${amanzi_branch}"']
  
  --spacedim=DIM          dimension of structured build (DIM=2 or 3) ['"${spacedim}"']

  --nersc_build           use cmake options required on NERSC machines ['"${nersc}"']
  
Build features:
Each feature listed here can be enabled/disabled with --[enable|disable]-[feature]
Value in brackets indicates default setting.

  structured              build structured AMR mesh capability ['"${structured}"']
  ccse_tools              build structured AMR tools for post processing and tecplot ['"${ccse_tools}"']

  unstructured            build unstructured mesh capability ['"${unstructured}"']
  stk_mesh                build the STK Mesh Toolkit ['"${stk_mesh}"']
  mstk_mesh               build the MSTK Mesh Toolkit ['"${mstk_mesh}"']
  moab_mesh               build the MOAB Mesh Toolkit ['"${moab_mesh}"']

  hypre                   build the HYPRE solver APIs ['"${hypre}"']
  petsc                   build the PETSc solver APIs ['"${petsc}"']

  pflotran                build the PFlotran geochemistry backend ['"${pflotran}"']
  alquimia                build the Alquimia geochemistry solver APIs ['"${alquimia}"']

  test_suite              run Amanzi Test Suite before installing ['"${test_suite}"']
  reg_tests               build regression tests into Amanzi Test Suite ['"${reg_tests}"']
  shared                  build Amanzi and tpls using shared libraries ['"${shared}"']
  native                  build Amanzi with native xml output for debugging enabled ['"${native}"']

Tool definitions:

  --with-c-compiler=FILE     FILE is the C compiler
  --with-cxx-compiler=FILE   FILE is the C++ compiler
  --with-fort-compiler=FILE  FILE is the Fortran compiler

  --with-c-flags=STRING      STRING is additional C compiler flags
  --with-cxx-flags=STRING    STRING is additional C++ compiler flags
  --with-fort-flags=STRING   STRING is additional Fortran compiler flags

  --with-link-flags=STRING   STRING is additional linker flags

  --with-cmake[=FILE]        FILE is the CMake binary ['"${cmake_binary}"'] without FILE builds CMake
  --with-ctest=FILE          FILE is the CTest binary ['"${ctest_binary}"'], ignored if --with-cmake is set
  --with-hg=FILE             FILE is the Mercurial binary ['"${hg_binary}"']
  --with-curl=FILE           FILE is the CURL binary ['"${curl_binary}"']

  --with-mpi=DIR             use MPI installed in DIR. Will search for MPI 
                             compiler wrappers under this directory. ['"${mpi_root_dir}"']



Directory and file names: 

  --prefix=PREFIX                install ALL files in tree rooted at PREFIX
                                 ['"${dflt_install_prefix}"']

  --amanzi-install-prefix=DIR    install Amanzi in tree rooted at DIR
                                 ['"${amanzi_install_prefix}"']

  --amanzi-build-dir=DIR         build Amanzi in DIR
                                 ['"${amanzi_build_dir}"']

  --tpl-install-prefix=DIR       install Amanzi TPLs in tree rooted at DIR
                                 ['"${tpl_install_prefix}"']
  
  --tpl-build-dir=DIR            build Amanzi TPLs in DIR
                                 ['"${tpl_build_dir}"']

  --tpl-download-dir=DIR         direct downloads of Amanzi TPLs to DIR
                                 ['"${tpl_download_dir}"']
'
}

function print_variable_values
{
echo '

Tools:
    cmake_binary ='"${cmake_binary}"'
    ctest_binary ='"${ctest_binary}"'
    hg_binary    ='"${hg_binary}"'
    curl_binary  ='"${curl_binary}"'
    mpi_root_dir ='"${mpi_root_dir}"'

Compilers:
    build_c_compiler    ='"${build_c_compiler}"'
    build_cxx_compiler  ='"${build_cxx_compiler}"'
    build_fort_compiler ='"${build_fort_compiler}"'

Compile Flags    
    build_c_flags    ='"${build_c_flags}"'
    build_cxx_flags  ='"${build_cxx_flags}"'
    build_fort_flags ='"${build_fort_flags}"'

Link Flags
    build_link_flags ='"${build_link_flags}"'

Build configuration:
    build_type          ='"${build_type}"'
    tpl_config_file     ='"${tpl_config_file}"'
    parallel            ='"${parallel_jobs}"'
    shared              ='"${shared}"'
    spacedim            ='"${spacedim}"'
    nersc_build         ='"${nersc}"'

Build Features:   
    structured          ='"${structured}"'
    unstructured        ='"${unstructured}"'
    ccse_tools          ='"${ccse_tools}"'
    stk_mesh            ='"${stk_mesh}"'
    mstk_mesh           ='"${mstk_mesh}"'
    moab_mesh           ='"${moab_mesh}"'
    test_suite          ='"${test_suite}"'
    reg_tests           ='"${reg_tests}"'
    netcdf4             ='"${netcdf4}"'
    hypre               ='"${hypre}"'
    petsc               ='"${petsc}"'
    alquimia            ='"${alquimia}"'
    pflotran            ='"${pflotran}"'
    native              ='"${native}"'

Directories:
    prefix                 ='"${prefix}"'
    amanzi_install_prefix  ='"${amanzi_install_prefix}"'
    amanzi_build_dir       ='"${amanzi_build_dir}"'
    tpl_install_prefix     ='"${tpl_install_prefix}"'
    tpl_build_dir          ='"${tpl_build_dir}"'
    tpl_download_dir       ='"${tpl_download_dir}"'
    
'    
}  
function parse_argv()
{
  argv=( "$@" )
  echo "${argv[1]}"
  last=$(( ${#argv[@]} - 1 ))
  i=0
  while [ $i -le ${last} ]
  do
    opt=${argv[$i]}
    echo "i: ${i} opt=$opt last: $last"
    case ${opt} in

      -h|--h|--help)
                print_usage
                exit_now 0
                ;;

      --prefix=*)
                 tmp=`parse_option_with_equal "${opt}" 'prefix'`
		 prefix=`make_fullpath ${tmp}`
                 ;;

      --parallel=[0-9]*)
                 parallel_jobs=`parse_option_with_equal "${opt}" 'parallel'`
                 ;;
      
      --opt)
                 build_type=Release
		 enable_native=${FALSE}
                 ;;

      --debug)
                 build_type=Debug
		 enable_native=${TRUE}
                 ;;

      --disable-*)
                 feature=`parse_feature "${opt}"`
                 set_feature ${feature} 'disable'
                 ;;

      --enable-*)
                 feature=`parse_feature "${opt}"`
                 set_feature ${feature} 'enable'
                 ;;

      --no-color)
                 no_color=${TRUE}
                 ;;

      --branch=*)
                 amanzi_branch=`parse_option_with_equal "${opt}" 'branch'`
                 ;;

      --spacedim=*)
                 spacedim=`parse_option_with_equal "${opt}" 'spacedim'`
                 ;;

      --nersc_build)
                 nersc=${TRUE}
                 ;;

      --with-c-compiler=*)
                 tmp=`parse_option_with_equal "${opt}" 'with-c-compiler'`
                 build_c_compiler=`make_fullpath $tmp`
                 ;;

      --with-c-flags=*)
                 build_c_flags=`parse_option_with_equal "${opt}" 'with-c-flags'`
                 ;;

      --with-cxx-compiler=*)
                 tmp=`parse_option_with_equal "${opt}" 'with-cxx-compiler'`
                 build_cxx_compiler=`make_fullpath $tmp`
                 ;;

      --with-cxx-flags=*)
                 build_cxx_flags=`parse_option_with_equal "${opt}" 'with-cxx-flags'`
                 ;;

      --with-fort-compiler=*)
                 tmp=`parse_option_with_equal "${opt}" 'with-fort-compiler'`
                 build_fort_compiler=`make_fullpath $tmp`
                 ;;

      --with-fort-flags=*)
                 build_fort_flags=`parse_option_with_equal "${opt}" 'with-fort-flags'`
                 ;;

      --with-link-flags=*)
                 build_link_flags=`parse_option_with_equal "${opt}" 'with-link-flags'`
                 ;;

      --with-cmake=*)
                 tmp=`parse_option_with_equal "${opt}" 'with-cmake'`
                 cmake_binary=`make_fullpath $tmp`
                 ;;

      --with-ctest=*)
                 tmp=`parse_option_with_equal "${opt}" 'with-ctest'`
                 ctest_binary=`make_fullpath $tmp`
                 ;;

      --with-cmake)
                 cmake_binary=
                 ;;

      --with-hg=*)
                 tmp=`parse_option_with_equal "${opt}" 'with-hg'`
                 hg_binary=`make_fullpath $tmp`
                 ;;

      --with-curl=*)
                 tmp=`parse_option_with_equal "${opt}" 'with-curl'`
                 curl_binary=`make_fullpath $tmp`
                 ;;

      --with-mpi=*)
                 tmp=`parse_option_with_equal "${opt}" 'with-mpi'`
                 mpi_root_dir=`make_fullpath $tmp`
                 ;;

      --amanzi-build-dir=*)
                 tmp=`parse_option_with_equal "${opt}" 'amanzi-build-dir'`
                 amanzi_build_dir=`make_fullpath $tmp`
                  ;;

      --amanzi-install-prefix=*)
                 tmp=`parse_option_with_equal "${opt}" 'amanzi-install-prefix'`
                 amanzi_install_prefix=`make_fullpath $tmp`
                 ;;

      --tpl-install-prefix=*)
                 tmp=`parse_option_with_equal "${opt}" 'tpl-install-prefix'`
                 tpl_install_prefix=`make_fullpath $tmp`
                 ;;

      --tpl-build-dir=*)
                 tmp=`parse_option_with_equal "${opt}" 'tpl-build-dir'`
                 tpl_build_dir=`make_fullpath $tmp`
                 ;;

      --tpl-download-dir=*)
                 tmp=`parse_option_with_equal "${opt}" 'tpl-download-dir'`
                 tpl_download_dir=`make_fullpath $tmp`
                 ;;
      
      --tpl-config-file=*)
                 tmp=`parse_option_with_equal "${opt}" 'tpl-config-file'`
                 tpl_config_file=`make_fullpath $tmp`
                 ;;

      --print)
                 print_exit=${TRUE}
		 ;;

       *)
                 error_message "'${opt}' is an unknown option or an option missing a value"
                 exit_now 20
                 ;;
      esac

      i=$[$i+1]
  done

}

# CURL functions

curl_version()
{
  $curl_binary --version
}

curl_protocols()
{
  vers_string=`$curl_binary --version | grep Protocols: | sed 's/Protocols://'`

  echo ${vers_string}
}

curl_test_protocol()
{
  local protocols tmp type=$1
  protocols=`curl_protocols`
  tmp=`echo $protocols | sed "s/.*${type}.*/${type}-TRUE/g"`
  if [ "${tmp}" = "${type}-TRUE" ]; then
    echo ${TRUE}
  else
    echo ${FALSE}
  fi
}  

# CMake functions

function cmake_version
{
  ver_string=`${cmake_binary} --version | sed 's/cmake version[ ]*//'`
  echo ${ver_string}
}

function download_cmake
{
   target_dir=$1
   target=
   pwd_save=`pwd`
   if [ ! -z "${target_dir}" ]; then
     mkdir_now ${target_dir}
     cd ${target_dir}
     target=${target_dir}/${cmake_archive_file}
   fi

   download_file ${cmake_url} ${cmake_archive_file} ${target}

   cd ${pwd_save}

   status_message "CMake download complete"


}

function unpack_cmake
{
  archive=$1
  final_source_dir=$2

  pwd_save=`pwd`

  # Create the final source destination
  mkdir_now ${final_source_dir}
  cd ${final_source_dir}

  # Untar the archive file removing the leading path component  
  tar -z -x -f ${archive} --strip-components 1

  if [ $? -ne 0 ]; then
    error_message "Failed to untar CMake archive file ${archive}"
    exit_now 30
  fi

  cd ${pwd_save}
  status_message "Unpack CMake complete"
}

function configure_cmake
{
  source_dir=$1
  build_dir=$2
  install_dir=$3

  pwd_save=`pwd`

  # Create and cd to the build directory
  mkdir_now ${build_dir}
  cd ${build_dir}

  # Build the bootstrap args
  boot_args="--prefix=${install_dir}"
  echo "boot_args=$boot_args"

  curl_supports_https=`curl_test_protocol https`
  if [ "${curl_supports_https}" ]; then
    boot_args="${boot_args} --system-curl"
  fi

  boot_args="${boot_args} --parallel=${parallel_jobs}"

  # Run the bootstrap script
  bootstrap_script=${source_dir}/bootstrap
  if [ ! -e ${bootstrap_script} ]; then
    error_message "CMake bootstrap script ${bootstrap_script} does not exist"
    exit_now 30
  fi

  ${bootstrap_script} ${boot_args}

  if [ $? -ne 0 ]; then
    error_message "CMake bootstrap script failed"
    exit_now 30
  fi

  cd ${pwd_save}

  status_message "Configure CMake complete"


}

function build_cmake
{
  root_build_dir=$1
  root_install_dir=$2
  root_download_dir=$3

  cmake_root_build_dir=${root_build_dir}/cmake
  cmake_source_dir=${cmake_root_build_dir}/cmake-${cmake_version}-source
  cmake_build_dir=${cmake_root_build_dir}/cmake-${cmake_version}-build

  # Download the file
  download_file ${cmake_url} ${cmake_archive_file} ${root_download_dir}

  # Define the full path name of the archive file
  if [ ! -z "${root_download_dir}" ]; then
    archive=${root_download_dir}/${cmake_archive_file}
  else
    archive=${cmake_archive_file}
  fi

  # Unpack the archive file
  unpack_cmake ${archive} ${cmake_source_dir}

  # Configure (bootstrap) CMake
  configure_cmake ${cmake_source_dir} ${cmake_build_dir} ${root_install_dir}

  # Build CMake 
  cd ${cmake_build_dir}
  make -j ${parallel_jobs}
  if [ $? -ne 0 ]; then
    error_message "Failed to build CMake"
    exit_now 30
  fi

  # Install CMake 
  make install
  if [ $? -ne 0 ]; then
    error_message "Failed to install CMake"
    exit_now 30
  fi

  cmake_binary=${root_install_dir}/bin/cmake
  ctest_binary=${root_install_dir}/bin/ctest

  status_message "CMake build complete"

}

# Mercury functions
function ascem_hg_clone
{
  repo=$1
  ${hg_binary} clone ${ascem_protocol}://${ascem_site}/hg/$repo
  if [ $? -ne 0 ]; then
    error_message "Failed to clone ${repo} from ${ascem_site}"
    exit_now 30
  fi
}

function hg_change_branch()
{
  branch=$1
  save_dir=`pwd`
  cd ${amanzi_source_dir}
  status_message "Updating ${amanzi_source_dir} to branch ${branch}"
  ${hg_binary} update ${branch}
  if [ $? -ne 0 ]; then
    error_message "Failed to update ${amanzi_source_dir} to branch ${branch}"
    exit_now 30
  fi
  cd ${save_dir}
 
}

# MPI Check
function check_mpi_root
{
  if [ -z "${mpi_root_dir}" ]; then

    mpi_root_env="${MPIROOT} ${MPI_ROOT} ${MPIHOME} ${MPI_HOME} ${MPI_PREFIX}"
    for env_try in ${mpi_root_env}; do
      if [ -e "${env_try}" ]; then
        mpi_root_dir="${env_try}"
        status_message "Located MPI installation in ${mpi_root_dir}"
        break
      fi
    done

  else

    if [ ! -e "${mpi_root_dir}" ] ; then
      error_message "MPI root directory ${mpi_root_dir} does not exist"
      exit_now 30
    fi

  fi
}

# Compiler functions
function define_c_compiler
{
  if [ -z "${build_c_compiler}" ]; then
 
    status_message "Searching for a C compiler"

    # build a list to search
    c_search_list=
    if [ -n "${mpi_root_dir}" ]; then
      c_search_list="${mpi_root_dir}/bin/mpicc"
    fi
    
    if [ -n "${CC}" ]; then 
      c_search_list="${c_search_list} ${CC}"
    fi

    c_search_list="${c_search_list} ${known_c_compilers}"
    for c_try in ${c_search_list}; do
      status_message "Searching for ${c_try}"
      full_c_try=`which ${c_try} 2>/dev/null`
      if [ -e "${full_c_try}" ]; then
        build_c_compiler="${full_c_try}"
        break
      fi
    done

    if [ -z "${build_c_compiler}" ]; then
      error_message "Failed to locate a C compiler. Please use the --with-c-compiler option."
      exit_now 30
    fi

  fi

  status_message "Build with C compiler: ${build_c_compiler}"
  export CC=${build_c_compiler}

}

function define_cxx_compiler
{
  if [ -z "${build_cxx_compiler}" ]; then
 
    status_message "Searching for a C++ compiler"

    # build a list to search
    cxx_search_list=
    if [ -n "${mpi_root_dir}" ]; then
      cxx_search_list="${mpi_root_dir}/bin/mpiCC"
      cxx_search_list="${mpi_root_dir}/bin/mpicxx ${cxx_search_list}"
    fi
    
    if [ -n "${CXX}" ]; then 
      cxx_search_list="${cxx_search_list} ${CXX}"
    fi

    cxx_search_list="${cxx_search_list} ${known_cxx_compilers}"
    for cxx_try in ${cxx_search_list}; do
      status_message "Searching for ${cxx_try}"
      full_cxx_try=`which ${cxx_try} 2>/dev/null`
      if [ -e "${full_cxx_try}" ]; then
        build_cxx_compiler="${full_cxx_try}"
        break
      fi
    done

    if [ -z "${build_cxx_compiler}" ]; then
      error_message "Failed to locate a C++ compiler. Please use the --with-cxx-compiler option."
      exit_now 30
    fi

  fi

  status_message "Build with C++ compiler: ${build_cxx_compiler}"
  export CXX=${build_cxx_compiler}

}

function define_fort_compiler
{
   if [ -z "${build_fort_compiler}" ]; then
 
    status_message "Searching for a Fortran compiler"

    # build a list to search
    fort_search_list=
    if [ -n "${mpi_root_dir}" ]; then
      fort_search_list="${mpi_root_dir}/bin/mpif90"
    fi
    
    if [ -n "${FC}" ]; then 
      fort_search_list="${fort_search_list} ${FC}"
    fi

    fort_search_list="${fort_search_list} ${known_fortran_compilers}"
    for fort_try in ${fort_search_list}; do
      status_message "Searching for ${fort_try}"
      full_fort_try=`which ${fort_try} 2>/dev/null`
      if [ -e "${full_fort_try}" ]; then
        build_fort_compiler="${full_fort_try}"
        break
      fi
    done

    if [ -z "${build_fort_compiler}" ]; then
      error_message "Failed to locate a Fortran compiler. Please use the --with-fort-compiler option."
      exit_now 30
    fi

  fi

  status_message "Build with Fortran compiler: ${build_fort_compiler}"
  export FC=${build_fort_compiler}

}

function check_compilers
{

  define_c_compiler
  define_cxx_compiler
  define_fort_compiler

  status_message "Compiler Check complete"

}

version_compare_element()
{
  (( $1 == $2 )) && return 0 #equal
  (( $1 > $2 )) && return 1  #greater than
  (( $1 < $2 )) && return 2  #less than
  exit 1
}

version_compare()
{
  A=(${1//./ })
  B=(${2//./ })
  i=0
  while (( i < ${#A[@]} )) && (( i < ${#B[@]})); 
  do
    version_compare_element "${A[i]}" "${B[i]}"
    result=$?
    [[ $result =~ [12] ]] && return $result
    let i++
  done
  version_compare_element "${#A[i]}" "${#B[i]}"
  return $?
}

# Tool Checks
function check_tools
{

  # Check Mercurial
  if [ ! -e "${hg_binary}" ]; then
    error_message "Mercurial (hg) binary does not exist"
    exit_now 10
  fi
  status_message "Mercury binary: ${hg_binary}"

  # Check CURL 
  if [ ! -e "${curl_binary}" ]; then
    error_message "CURL binary does not exist"
    exit_now 10
  fi
  status_message "CURL binary: ${curl_binary}"

  # Check CMake and CTest
  if [ -z "${cmake_binary}" ]; then 
    status_message "CMake not defined. Will build"
    build_cmake ${tpl_build_dir} ${tpl_install_prefix} ${tpl_download_dir} 
  fi
  if [ ! -e "${cmake_binary}" ]; then
    error_message "CMake binary does not exist. Will build."
    build_cmake ${tpl_build_dir} ${tpl_install_prefix} ${tpl_download_dir} 
  else
    # CMake binary does exist, not check the version number
    ver_string=`${cmake_binary} --version | sed 's/cmake version[ ]*//'`
    status_message "Found CMake version:  ${ver_string}"
    ( version_compare "$ver_string" "$cmake_version" )
    result=$?
    if [ ${result} -eq 2 ]; then
      status_message "CMake version is less than required version. Will build CMake version 2.8.8"
      build_cmake ${tpl_build_dir} ${tpl_install_prefix} ${tpl_download_dir} 
    fi
  fi
  if [ ! -e "${ctest_binary}" ]; then
    error_message "CTest binary does not exist. Will deactivate the test suite."
    test_suite=${FALSE}
    reg_tests=${FALSE}
  fi
  status_message "CMake binary: ${cmake_binary}"
  status_message "CTest binary: ${ctest_binary}"

  status_message "Tool check complete"

}

# Directory functions
function define_build_directories
{
  # The amanzi build directory
  if [ ! -e "${amanzi_build_dir}" ]; then
    mkdir_now "${amanzi_build_dir}"
  fi

  # The TPL build directory
  if [ ! -e "${tpl_build_dir}" ]; then
    mkdir_now "${tpl_build_dir}"
  fi

  # The TPL download directory
  if [ ! -e "${tpl_download_dir}" ]; then
    mkdir_now "${tpl_download_dir}"
  fi

  status_message "Build directories ready"

}

function define_install_directories
{
  # The prefix option overrides the other install choices
  if [ ! -z  "${prefix}" ] ; then
    status_message "Global prefix (${prefix}) set override Amanzi and TPL installations"
    amanzi_install_prefix=${prefix}
    tpl_install_prefix=${prefix}/tpls
  fi

  status_message "Amanzi installation: ${amanzi_install_prefix}"
  status_message "TPL installation: ${tpl_install_prefix}"

}    

function define_unstructured_dependencies
{
  if [ "${unstructured}" -eq "${FALSE}" ]; then
    eval "stk_mesh=$FALSE"
    eval "mstk_mesh=$FALSE"
    eval "moab_mesh=$FALSE"
  fi
}

function define_structured_dependencies
{
  if [ "${structured}" -eq "${TRUE}" ]; then
    eval "petsc=$TRUE"
  fi
}

function define_nersc_options
{
  if [ "${nersc}" -eq "${TRUE}" ]; then
    nersc_tpl_opts="-DPREFER_STATIC_LIBRARIES:BOOL=TRUE"
    nersc_amanzi_opts="-DTESTS_REQUIRE_MPIEXEC:BOOL=TRUE -DTESTS_REQUIRE_FULLPATH:BOOL=TRUE -DPREFER_STATIC_LIBRARIES:BOOL=TRUE -DBUILD_STATIC_EXECUTABLES:BOOL=TRUE"
    status_message "NERSC TPL OPTS = ${nersc_tpl_opts}"
    status_message "NERSC AMANZI OPTS = ${nersc_amanzi_opts}"
  fi
}

#
# End functions
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#  Main 
# ---------------------------------------------------------------------------- #

# Parse the command line arguments
array=( "$@" )
parse_argv "${array[@]}"

# Set extra options for building on nersc
define_nersc_options

# Set packages that depend on unstructured
define_unstructured_dependencies

# Set packages that depend on structured
define_structured_dependencies

# Define the install directories 
define_install_directories

# Create the build directories
define_build_directories

# Check the MPI root value 
check_mpi_root

# Define the compilers
check_compilers

# Check the cmake, hg and curl tools
check_tools

# Print and exit if --print is set
if [ "${print_exit}" -eq "${TRUE}" ]; then
  print_variable_values
  exit_now 0
fi

# Change the branch
if [ ! -z "${amanzi_branch}" ]; then
  hg_change_branch ${amanzi_branch}
fi

# Now build the TPLs if the config file is not defined
if [ -z "${tpl_config_file}" ]; then

  # Return to this directory once the TPL build is complete
  pwd_save=`pwd`


  # Define the TPL build source directory
  tpl_build_src_dir=${amanzi_source_dir}/config/SuperBuild

  # Configure the TPL build
  cd ${tpl_build_dir}
  ${cmake_binary} \
      	        -DCMAKE_C_FLAGS:STRING="${build_c_flags}" \
                -DCMAKE_CXX_FLAGS:STRING="${build_cxx_flags}" \
                -DCMAKE_Fortran_FLAGS:STRING="${build_fort_flags}" \
                -DCMAKE_EXE_LINKER_FLAGS:STRING="${build_link_flags}" \
                -DCMAKE_BUILD_TYPE:STRING=${build_type} \
                -DCMAKE_C_COMPILER:STRING=${build_c_compiler} \
                -DCMAKE_CXX_COMPILER:STRING=${build_cxx_compiler} \
                -DCMAKE_Fortran_COMPILER:STRING=${build_fort_compiler} \
                -DTPL_INSTALL_PREFIX:STRING=${tpl_install_prefix} \
                -DENABLE_Structured:BOOL=${structured} \
                -DENABLE_Unstructured:BOOL=${unstructured} \
                -DENABLE_CCSE_TOOLS:BOOL=${ccse_tools} \
                -DENABLE_STK_Mesh:BOOL=${stk_mesh} \
                -DENABLE_MOAB_Mesh:BOOL=${moab_mesh} \
                -DENABLE_MSTK_Mesh:BOOL=${mstk_mesh} \
                -DENABLE_NetCDF4:BOOL=${netcdf4} \
                -DENABLE_HYPRE:BOOL=${hypre} \
                -DENABLE_PETSC:BOOL=${petsc} \
                -DENABLE_ALQUIMIA:BOOL=${alquimia} \
                -DENABLE_PFLOTRAN:BOOL=${plotran} \
                -DENABLE_Silo:BOOL=${silo} \
                -DBUILD_SHARED_LIBS:BOOL=${shared} \
                -DCCSE_BL_SPACEDIM:INT=${spacedim} \
                -DPREFER_STATIC_LIBRARIES:BOOL=${static} \
                ${nersc_tpl_opts} \
                ${tpl_build_src_dir}

  if [ $? -ne 0 ]; then
    error_message "Failed to configure TPL build"
    exit_now 30
  fi
  status_message "TPL configure complete"

  # TPL make 
  cd ${tpl_build_dir}
  make -j ${parallel_jobs}
  if [ $? -ne 0 ]; then
    error_message "Failed to build TPLs"
    exit_now 30
  fi

  # TPL Install
  cd ${tpl_build_dir}
  make install
  if [ $? -ne 0 ]; then
    error_message "Failed to install configure script"
    exit_now 30
  fi

  tpl_config_file=${tpl_install_prefix}/share/cmake/amanzi-tpl-config.cmake

  cd ${pwd_save}

  status_message "TPL build complete"
  status_message "For future Amanzi builds use ${tpl_config_file}"

else 

  status_message "Checking configuration file ${tpl_config_file}"

  if [ ! -e "${tpl_config_file}" ]; then
    error_message "Configure file ${amanzi_config_file} does not exist"
    exit_now 30
  fi

  if [ ! -r "${tpl_config_file}" ]; then
    error_message "Configure file ${tpl_config_file} is not readable"
    exit_now 30
  fi

  if [ ! -f "${tpl_config_file}" ]; then
    error_message "Configure file ${tpl_config_file} is not a regular file"
    exit_now 30
  fi


fi


status_message "Build Amanzi with configure file ${tpl_config_file}"

# Amanzi Configure
cd ${amanzi_build_dir}

# -- Amanzi options

${cmake_binary} \
              -C${tpl_config_file} \
      	      -DCMAKE_C_FLAGS:STRING="${build_c_flags}" \
      	      -DCMAKE_CXX_FLAGS:STRING="${build_cxx_flags}" \
      	      -DCMAKE_Fortran_FLAGS:STRING="${build_fort_flags}" \
      	      -DCMAKE_EXE_LINKER_FLAGS:STRING="${build_link_flags}" \
              -DCMAKE_INSTALL_PREFIX:STRING=${amanzi_install_prefix} \
              -DCMAKE_BUILD_TYPE:STRING=${build_type} \
              -DCMAKE_C_COMPILER:STRING=${build_c_compiler} \
              -DCMAKE_CXX_COMPILER:STRING=${build_cxx_compiler} \
              -DCMAKE_Fortran_COMPILER:STRING=${build_fort_compiler} \
              -DENABLE_Structured:BOOL=${structured} \
              -DENABLE_Unstructured:BOOL=${unstructured} \
              -DENABLE_STK_Mesh:BOOL=${stk_mesh} \
              -DENABLE_MOAB_Mesh:BOOL=${moab_mesh} \
              -DENABLE_MSTK_Mesh:BOOL=${mstk_mesh} \
              -DENABLE_HYPRE:BOOL=${hypre} \
              -DENABLE_PETSC:BOOL=${petsc} \
              -DENABLE_ALQUIMIA:BOOL=${alquimia} \
              -DENABLE_PFLOTRAN:BOOL=${plotran} \
              -DBUILD_SHARED_LIBS:BOOL=${shared} \
              -DCCSE_BL_SPACEDIM:INT=${spacedim} \
	      -DENABLE_Regression_Tests:BOOL=${reg_tests} \
	      -DENABLE_NATIVE_XML_OUTPUT:BOOL=${native} \
              ${nersc_amanzi_opts} \
              ${amanzi_source_dir}


if [ $? -ne 0 ]; then
  error_message "Failed to configure Amanzi"
  exit_now 50
fi
status_message "Amanzi configure complete"

# Amanzi Build
make -j ${parallel_jobs}
if [ $? -ne 0 ]; then
  error_message "Failed to build Amanzi"
  exit_now 50
fi
status_message "Amanzi build complete"

# Amanzi Test Suite
if [ "${test_suite}" -eq "${TRUE}" ]; then
  status_message "Run Amanzi test suite"
  ${ctest_binary} --output-on-failure --output-log amanzi-test-results.log #-j ${parallel_jobs}
  status_message "Test results in amanzi-test-results.log"
  if [ $? -ne 0 ]; then
    error_message "Amanzi test suite failed"
    exit_now 30
  fi
fi

# Amanzi Install
make install
if [ $? -ne 0 ]; then
  error_message "Failed to install Amanzi"
  exit_now 50
fi
status_message "Amanzi install complete"

status_message "Bootstrap complete"

exit_now 0
