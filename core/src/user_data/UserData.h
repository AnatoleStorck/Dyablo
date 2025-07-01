#pragma once

#include <set>

#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

namespace dyablo {

namespace UserData_Impl{
  struct UserData_Fields_Pdata;
  class UserData_FieldAccessor;
  class UserData_FieldAccessor_fulltree;
  struct UserData_FieldAccessor_FieldInfo;

  struct UserData_Particles_Pdata;
  class UserData_ParticleAccessor;
  struct UserData_ParticleAccessor_AttributeInfo;
} //namespace UserData_Impl

class UserData
{
public:
  UserData( ConfigMap& configMap, ForeachCell& foreach_cell )
  : fields(configMap, foreach_cell),
    particles(configMap, foreach_cell)
  {}

  //########################
  // Fields Management 
  //########################

  using FieldView_t = ForeachCell::CellArray_global_ghosted;

  /***
   * @brief Return a CellArray_global_ghosted::Shape_t instance 
   * with the same size as all fields in current UserData
   * UserData must have at least one active field
   ***/
  const FieldView_t::Shape_t getShape() const;

  /***
   * @brief Add new fields with unique identifiers 
   * names should not be already present
   * WARNING : Invalidates all field accessors if reallocation happens
   ***/
  void new_fields( const std::set<std::string>& names);

  /***
   * @brief Allocate intermediate octants for existing field
   * WARNING : Invalidates all field accessors that have intermediates if reallocation happens
   ***/
  void new_intermediate_fields(const std::set<std::string>& names);

  /***
   * @brief Check if field exists
   ***/
  bool has_field(const std::string& name) const;

  /***
   * @brief Get identifier strings for all enabled fields
   ***/
  std::set<std::string> getEnabledFields() const;

  /***
   * @brief Get View associated with field name
   * Note : this creates a copy, you can't update fields that way
   ***/
  const FieldView_t getField(const std::string& name) const;

  /***
   * @brief Change name of a field from `src` to `dest`
   * NOTE : order of parameters is dest, src like in Kokkos deep_copy
   * WARNING : Invalidates all accessors containing source field
   ***/
  void move_field( const std::string& dest, const std::string& src );

  /***
   * @brief Delete a field
   * WARNING : Invalidates all accessors containing this field
   ***/
  void delete_field( const std::string& name );

  /***
   * @brief Delete an intermediate field
   * WARNING : Invalidates all accessors containing this field
   ***/
  void clear_intermediates();

  void exchange_loadbalance( const ViewCommunicator& ghost_comm );

  /***
   * @brief Get the number of active fields in UserData
   ***/
  int nbFields() const;

  using FieldAccessor_FieldInfo = UserData_Impl::UserData_FieldAccessor_FieldInfo;

  using FieldAccessor = UserData_Impl::UserData_FieldAccessor; /// FieldAccessor with only leaves
  using FieldAccessor_leaves = FieldAccessor; /// FieldAccessor with only leaves

  using FieldAccessor_fulltree = UserData_Impl::UserData_FieldAccessor_fulltree; /// FieldAccessor with leaves and intermediates  
  using FieldAccessor_intermediates [[deprecated]] = FieldAccessor_fulltree; /// Replaced by FieldAccessor_fulltree

  /***
   * @brief create a FieldAccessor to access fields listed in `fields_info`
   * NOTE : Accessors may be invalidated by some methods from UserData (e.g. deleting or moving a field, reallocating, ...)
   * Do not keep invalidated accessors since live accessors may prevent Kokkos::View deallocation and create memory leaks
   ***/
  FieldAccessor getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const;

  /***
   * @brief create a FieldAccessor to access fields listed in `fields_info` and allow access to intermediate cells
   ***/
  FieldAccessor_fulltree getAccessor_fulltree( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const;

  /***
   * @copydoc getAccessor_fulltree
   * @note Deprecated : replaced by getAccessor_fulltree
   ***/
   [[deprecated]] FieldAccessor_fulltree getAccessor_intermediates( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const;

  /***
   * @brief Reallocate Userdata to fit new AMRmesh size
   * @returns a FieldAccessor with old data
   * AMRmesh is contained in ForeachCell instance that was used to construct this UserData
   * Old data will be deallocated when the returned FieldAccessor is destroyed
   ***/
  FieldAccessor backup_and_realloc();

  void extend_fields();

  //########################
  // Particles Management 
  //########################

  /***
   * @brief Create a new particle array with `num_particles` particles
   ***/
  void new_ParticleArray( const std::string& name, uint32_t num_particles );

  /***
   * @brief Create a new attribute for particle array `array_name` (must exist)
   ***/
  void new_ParticleAttribute( const std::string& array_name, const std::string& attribute_name );

  /***
   * @brief Check if UserData contains a ParticleArray with this name
   ***/
  bool has_ParticleArray(const std::string& name) const;

  /***
   * @brief Check if ParticleArray `array_name` (must exist) has an attribute with this name
   ***/
  bool has_ParticleAttribute(const std::string& array_name, const std::string& attribute_name ) const;

  /***
   * @brief Get identifier strings for all enabled particle arrays
   ***/
  std::set<std::string> getEnabledParticleArrays() const;

  /***
   * @brief Get identifier strings for all enabled attributes of particle array `array_name`
   ***/
  std::set<std::string> getEnabledParticleAttributes( const std::string& array_name ) const;

  using ParticleArray_t = ParticleArray;
  /***
   * @brief Get particle array associated with name
   ***/
  ParticleArray_t getParticleArray( const std::string& array_name ) const;

  using ParticleAttribute_t = ParticleData;
  /***
   * @brief Get particle attribute `array_name`/`attribute_name`
   * Note : this will create a copy of the data, you can't modify attributes from there
   ***/
  ParticleAttribute_t getParticleAttribute( const std::string& array_name, const std::string& attribute_name ) const;
    
  /***
   * @brief Change name of particle attribute from `array_name`/`attr_src` to `array_name`/`attr_dest`
   * NOTE : order of parameters is dest, src like in Kokkos deep_copy
   * WARNING : Invalidates all accessors containing source attribute 
   ***/
  void move_ParticleAttribute( const std::string& array_name, const std::string& attr_dest, const std::string& attr_src );
   
  /***
   * @brief Delete attribte `array_name`/`attribute_name` from user data
   * WARNING : Invalidates all accessors containing this attribute
   ***/
  void delete_ParticleAttribute(const std::string& array_name, const std::string& attribute_name);
  
  using ParticleAccessor = UserData_Impl::UserData_ParticleAccessor; 
  using ParticleAccessor_AttributeInfo = UserData_Impl::UserData_ParticleAccessor_AttributeInfo;
  /***
   * @brief create a ParticleAccessor to access attributes listed in `attribute_info` from particle array `array_name`
   * NOTE : Accessors may be invalidated by some methods from UserData (e.g. deleting or moving an attribute or an array, reallocating, ...)
   * Do not keep invalidated accessors since live accessors may prevent Kokkos::View deallocation and create memory leaks
   ***/
  ParticleAccessor getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const;
    
  /***
   * @brief Distribute position array and attributes for particle array `array_name`
   * WARNING : Invalidates all accessors containing this particle array
   ***/
  void distributeParticles( const std::string& array_name );
    
  void distributeAllParticles();

  /***
   * @brief Delete particle array `name` from user data
   ***/
  void delete_ParticleArray( const std::string& name );

  /**
   * Merge particle arrays id_dest <- [id_dest,id_to_merge]
   * Particles are only merged if attribute id_to_merge/mask_field is not 0
   * Particle id_to_merge should have all attributes from id_dest, other attributes are ignored
   **/
  void merge_particles_if( const std::string& id_dest, const std::string& id_to_merge, const std::string& mask_field );

private:
  struct Fields
  {
    Fields(ConfigMap& configMap, ForeachCell& foreach_cell);
    ~Fields();
    std::unique_ptr<UserData_Impl::UserData_Fields_Pdata> pdata;
  };
  Fields fields;
  struct Particles
  {
    Particles(ConfigMap& configMap, ForeachCell& foreach_cell);
    ~Particles();
    std::unique_ptr<UserData_Impl::UserData_Particles_Pdata> pdata;
  };
  Particles particles;
};


} //namespace dyablo