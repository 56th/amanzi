/*
  This is the operators component of the Amanzi code. 

  Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_OPERATORS_DEFS_HH_
#define AMANZI_OPERATORS_DEFS_HH_

namespace Amanzi {
namespace Operators {

// Constants in the next block must powers of 2.
const int OPERATOR_SCHEMA_DOFS_FACE = 1;
const int OPERATOR_SCHEMA_DOFS_CELL = 2;
const int OPERATOR_SCHEMA_DOFS_NODE = 4;
const int OPERATOR_SCHEMA_DOFS_EDGE = 8;

const int OPERATOR_SCHEMA_BASE_FACE = 16;
const int OPERATOR_SCHEMA_BASE_CELL = 32;
const int OPERATOR_SCHEMA_BASE_NODE = 64;
const int OPERATOR_SCHEMA_BASE_EDGE = 128;

const int OPERATOR_SCHEMA_RULE_EXACT = 1;
const int OPERATOR_SCHEMA_RULE_SUBSET = 2;

const int OPERATOR_BC_NONE = 0;
const int OPERATOR_BC_DIRICHLET = 1;
const int OPERATOR_BC_NEUMANN = 2;
const int OPERATOR_BC_MIXED = 4;

const int OPERATOR_BC_TYPE_NODE = 8;
const int OPERATOR_BC_TYPE_FACE = 16;
const int OPERATOR_BC_TYPE_EDGE = 32;

const int OPERATOR_HEX_FACES = 6;  // Hexahedron is the common element
const int OPERATOR_HEX_NODES = 8;
const int OPERATOR_HEX_EDGES = 12;

const int OPERATOR_QUAD_FACES = 4;  // Quadrilateral is the common element
const int OPERATOR_QUAD_NODES = 4;
const int OPERATOR_QUAD_EDGES = 4;

const int OPERATOR_MAX_FACES = 14;  // Kelvin's tetrakaidecahedron
const int OPERATOR_MAX_NODES = 47;  // More accurate bounds could be
const int OPERATOR_MAX_EDGES = 60;  // calculated during initialization.

// Newton-correction options
const int OPERATOR_DIFFUSION_JACOBIAN_NONE = 0;
const int OPERATOR_DIFFUSION_JACOBIAN_TRUE = 1;
const int OPERATOR_DIFFUSION_JACOBIAN_APPROXIMATE = 2;

// upwind options
const int OPERATOR_UPWIND_NONE = 0;
const int OPERATOR_UPWIND_CONSTANT_VECTOR = 3;
const int OPERATOR_UPWIND_FLUX = 4;
const int OPERATOR_UPWIND_ARITHMETIC_AVERAGE = 5;
const int OPERATOR_UPWIND_AMANZI_DIVK = 7;
const int OPERATOR_UPWIND_AMANZI_SECOND_ORDER = 8;
const double OPERATOR_UPWIND_RELATIVE_TOLERANCE = 1e-12;

// method for nonlinear coefficient
const int OPERATOR_LITTLE_K_UPWIND = 1;
const int OPERATOR_LITTLE_K_DIVK = 2;
const int OPERATOR_LITTLE_K_DIVK_TWIN = 3;
const int OPERATOR_LITTLE_K_DIVK_TWIN_GRAD = 4;
const int OPERATOR_LITTLE_K_STANDARD = 5;
const int OPERATOR_LITTLE_K_ARTIFICIAL_DIFFUSION = 6;

// method for gravity
const int OPERATOR_GRAVITY_HH = 1;
const int OPERATOR_GRAVITY_FV = 2;

// special properties of operators
const int OPERATOR_PROPERTY_DATA_READ_ONLY = 1;  // must power of 2
const int OPERATOR_PROPERTY_MAP_CONSTANT = 2;

// reconstruction options
const double OPERATOR_RECONSTRUCTION_MATRIX_CORRECTION = 1e-15;

// limiting options
const int OPERATOR_LIMITER_BARTH_JESPERSEN = 1; 
const int OPERATOR_LIMITER_TENSORIAL = 2;
const int OPERATOR_LIMITER_KUZMIN = 3;
const double OPERATOR_LIMITER_TOLERANCE = 1e-14;
const double OPERATOR_LIMITER_INFINITY = 1e+99;

}  // namespace Operators
}  // namespace Amanzi

#endif

