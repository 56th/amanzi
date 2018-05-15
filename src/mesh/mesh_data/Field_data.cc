#include "Field_data.hh"
#include "dbc.hh"


namespace Amanzi {
namespace AmanziMesh {
namespace Data {

Field::Field (std::string name, FIELD_TYPE type, Entity_kind location) : 
    name_ (name), type_ (type), location_ (location)
{
    AMANZI_ASSERT (ok_field_type (type));
    AMANZI_ASSERT (entity_valid_kind (location));
}

} // namespace Data
} // namespace Mesh
} // namespace Amanzi
