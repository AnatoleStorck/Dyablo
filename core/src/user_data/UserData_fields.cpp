#include "FieldAccessor.h"

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

namespace dyablo {

namespace UserData_Impl{

struct UserData_Fields_Pdata
{
  struct field_index_t
  {
      int index;
  };
public:
    using FieldView_t = ForeachCell::CellArray_global_ghosted;

    UserData_Fields_Pdata( const UserData_Fields_Pdata& ) = default;
    UserData_Fields_Pdata( UserData_Fields_Pdata&& ) = default;

    UserData_Fields_Pdata( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_cell(foreach_cell)
    {}

    /***
     * @brief Return a CellArray_global_ghosted::Shape_t instance 
     * with the same size as all fields in current UserData_fields
     * UserData_fields must have at least one active field
     * WARNING : resulting shape doesn't account for intermediates
     ***/
    const FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_HOST_RELEASE( field_index.size() > 0, "Cannot getShape() of an empty UserData_fields" );
        return fields.getShape();
    }

    void extend_fields( )
    {
        int nbOcts = this->foreach_cell.get_amr_mesh().getNumOctants();
        int nbGhosts = this->foreach_cell.get_amr_mesh().getNumGhosts();
        extend_fields_aux(foreach_cell, fields, nbOcts, nbGhosts, max_field_count);
    }

    static void initialise_new_aux(FieldView_t& fields, const int index)
    {
        // Offsets are computed by hand in 64 bits : these arrays can hold more
        // than 2^32 elements, and Kokkos::View::operator() would wrap silently
        // (see CellArray_base::at_offset()).
        const auto& U = fields.U;
        const uint32_t extent_0 = U.extent(0);
        const uint32_t extent_2 = U.extent(2);
        real_t* const U_data = U.data();
        const size_t U_stride_1 = U.stride(1);
        const size_t U_stride_2 = U.stride(2);

        Kokkos::parallel_for( "zero_new_field", Kokkos::RangePolicy<>(0, extent_0*extent_2),
            KOKKOS_LAMBDA( const uint32_t i )
        {
            const uint32_t iCell = i%extent_0;
            const uint32_t iOct  = i/extent_0;
            U_data[ (size_t)iOct*U_stride_2 + index*U_stride_1 + iCell ] = 0;
        });

        const auto& Ughost = fields.Ughost;
        const uint32_t extent_0_ghost = Ughost.extent(0);
        const uint32_t extent_2_ghost = Ughost.extent(2);
        real_t* const Ughost_data = Ughost.data();
        const size_t Ughost_stride_1 = Ughost.stride(1);
        const size_t Ughost_stride_2 = Ughost.stride(2);
        Kokkos::parallel_for( "zero_new_field_ghost", Kokkos::RangePolicy<>(0, extent_0_ghost*extent_2_ghost),
            KOKKOS_LAMBDA( const uint32_t i )
        {
            const uint32_t iCell = i%extent_0_ghost;
            const uint32_t iOct  = i/extent_0_ghost;
            Ughost_data[ (size_t)iOct*Ughost_stride_2 + index*Ughost_stride_1 + iCell ] = 0;
        });
    }

    static void extend_fields_aux( ForeachCell& foreach_cell, FieldView_t& fields, uint32_t nbOcts, uint32_t nbGhosts, uint32_t max_field_count )
    {
        int allocated_field_count = fields.nbfields();
        FieldView_t::Shape_t shape{
          .bx = foreach_cell.blockSize()[IX],
          .by = foreach_cell.blockSize()[IY],
          .bz = foreach_cell.blockSize()[IZ],
          .nbFields = max_field_count,
          .nbOcts = nbOcts,
          .nbGhosts = nbGhosts,
        };
        FieldView_t fields_new( "UserData_fields", shape );
        
        if( allocated_field_count != 0 )
        {
            Kokkos::deep_copy( 
                Kokkos::subview(fields_new.U, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                fields.U
            );
            Kokkos::deep_copy( 
                Kokkos::subview(fields_new.Ughost, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                fields.Ughost
            );
        }
        fields = fields_new;
    }

    static void new_fields_aux( const std::set<std::string>& names,
                                const size_t nbOcts,
                                const size_t nbGhosts,
                                int& max_field_count,
                                std::map<std::string, field_index_t>& field_index,
                                FieldView_t& fields,
                                ForeachCell& foreach_cell)
    {
        if( field_index.size() != 0 )
        {
            DYABLO_ASSERT_HOST_RELEASE( fields.U.extent(2) == nbOcts, "UserData_fields internal error : mismatch between allocated size and octant count" );
            DYABLO_ASSERT_HOST_RELEASE( fields.Ughost.extent(2) == nbGhosts, "UserData_fields internal error : mismatch between allocated size and ghost octant count" );
        }
        
        int needed_field_count = field_index.size() + names.size();
        max_field_count = std::max( max_field_count, needed_field_count );
        int allocated_field_count = fields.nbfields();
        if( needed_field_count > allocated_field_count )
        {   // Not enough fields : resize to add fields
            std::cout << "Reallocate : add fields " << allocated_field_count << " -> " << max_field_count << std::endl;
            extend_fields_aux(foreach_cell, fields, nbOcts, nbGhosts, max_field_count);
        }

        for( const std::string& name : names )
        {
            if( 1 == field_index.count(name) )
                throw std::runtime_error(std::string("UserData_fields::new_fields() - field already exists : ") + name);
            /// Find first free ivar in `fields` view
            auto first_free = [&]() -> int
            {
                for(int i=0; i<fields.nbfields(); i++)
                {
                    bool free = true;
                    for( auto& p : field_index )
                    {
                        if(p.second.index == i)
                            free = false;
                    }
                    if( free ) return i;
                }
                DYABLO_ASSERT_HOST_RELEASE(false, "UserData_fields internal error : not enough fields allocated");
                return -1;
            };
            
            int index = first_free();
            field_index[name].index = index;
            initialise_new_aux(fields, index);
        }
    }

    /**
     * Add new fields with unique identifiers 
     * names should not be already present
     **/
    void new_fields( const std::set<std::string>& names)
    {
        size_t nbOcts = this->foreach_cell.get_amr_mesh().getNumOctants();
        size_t nbGhosts = this->foreach_cell.get_amr_mesh().getNumGhosts();
        int& max_field_count = this->max_field_count;
        std::map<std::string, field_index_t>& field_index = this->field_index;
        FieldView_t& fields = this->fields;

        new_fields_aux(names,nbOcts,nbGhosts,max_field_count,field_index,fields,this->foreach_cell);
    }

    void new_intermediate_fields( const std::set<std::string>& names)
    {
        size_t nbOcts = this->foreach_cell.get_amr_mesh().getNumIntermediates();
        size_t nbGhosts = this->foreach_cell.get_amr_mesh().getNumIntermediateGhosts();
        int& max_field_count = this->max_field_count_intermediate;
        std::map<std::string, field_index_t>& field_index = this->field_index_intermediate;
        FieldView_t& fields = this->fields_intermediate;

        new_fields_aux(names,nbOcts,nbGhosts,max_field_count,field_index,fields,this->foreach_cell);
    }

    /// Check if field exists
    bool has_field(const std::string& name) const
    {
        return field_index.end() != field_index.find(name);
    }

    /// Check if field exists
    bool has_intermediate_field(const std::string& name) const
    {
        return field_index_intermediate.end() != field_index_intermediate.find(name);
    }

    std::set<std::string> getEnabledFields() const
    {
        std::set<std::string> res;
        for( const auto& p : field_index )
        {
            res.insert( p.first );
        }
        return res;
    }   

    // Get View associated with field name
    const FieldView_t getField(const std::string& name) const
    {
        if( !this->has_field(name)  )
            throw std::runtime_error(std::string("UserData_fields::getField() - field doesn't exist : ") + name);
        
        int index = field_index.at(name).index;

        auto field = foreach_cell.allocate_ghosted_array( std::string("field_")+name, 1 );
        Kokkos::deep_copy( 
            field.U,
            Kokkos::subview(fields.U, Kokkos::ALL(), std::pair(index, index+1) , Kokkos::ALL() )
        );

        return field;
    }

    /// Change field name from src to dest. If dest already exist it is replaced
    void move_field( const std::string& dest, const std::string& src )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_field(src), "UserData_fields::move_field() - field doesn't exist : " << src);

        field_index[ dest ] = field_index.at( src );
        field_index.erase( src );
    }

    void delete_field( const std::string& name )
    {
        field_index.erase( name );
    }

    void clear_intermediates()
    {
        this->fields_intermediate = FieldView_t(); // Reset intermediate fields
        this->field_index_intermediate.clear(); // Clear the index
    }

    /// Get the number of active fields in UserData_fields
    int nbFields() const
    {
        return field_index.size();
    }

    int nbFields_intermediates()
    {
      return field_index_intermediate.size();
    }

    void exchange_loadbalance( const ViewCommunicator& ghost_comm )
    {
      DYABLO_ASSERT_HOST_RELEASE( 0 == nbFields_intermediates(), "UserData::exchange_loadbalance : Keeping intermediates between interations is not supported yet" );
      {  
        UserData::FieldAccessor fields_old = this->backup_and_realloc();
        int nb_fields = this->nbFields();
        auto old_fields_View = Kokkos::subview( fields_old.fields.U, 
                                                Kokkos::ALL(),
                                                std::make_pair(0, nb_fields),
                                                Kokkos::ALL()  );
        auto new_fields_View = Kokkos::subview(this->fields.U, 
                                                Kokkos::ALL(),
                                                std::make_pair(0, nb_fields),
                                                Kokkos::ALL()  );
        
        ghost_comm.exchange_ghosts<2>(old_fields_View, new_fields_View );
      }// This block is important to deallocate fields_old before extend_fields()

      this->extend_fields();
    }

    UserData::FieldAccessor getAccessor( const std::vector<UserData::FieldAccessor_FieldInfo>& fields_info ) const
    {
      return UserData::FieldAccessor(*this, fields_info);
    }

    UserData::FieldAccessor_fulltree getAccessor_fulltree( const std::vector<UserData::FieldAccessor_FieldInfo>& fields_info ) const
    {
      return UserData::FieldAccessor_fulltree(*this, fields_info);
    }

    UserData::FieldAccessor backup_and_realloc()
    {
      DYABLO_ASSERT_HOST_RELEASE( 0 == nbFields_intermediates(), "UserData::backup_and_realloc : Keeping intermediates between interations is not supported yet" );

      using FieldAccessor = UserData::FieldAccessor;
      std::vector<FieldAccessor::FieldInfo> all_fields;
      int i=0;
      for( const std::string& field : this->getEnabledFields() )
        all_fields.push_back({field, i++});
      FieldAccessor fields_old = this->getAccessor( all_fields );

      this->fields = foreach_cell.allocate_ghosted_array( "UserData_fields", this->field_index.size() );
      // Reorder fields to reduce fragmentation
      std::map<std::string, field_index_t> field_index_new;
      {
        int new_index = 0;
        for( const auto& [field_name, old_index] : this->field_index )
        {
          field_index_new[field_name].index = new_index;
          new_index++;
        }
      }
      this->field_index = field_index_new;

      return fields_old;
    }

//private:
    ForeachCell& foreach_cell;
    FieldView_t fields;
    std::map<std::string, field_index_t> field_index;
    int max_field_count = 0;

    FieldView_t fields_intermediate;
    std::map<std::string, field_index_t> field_index_intermediate;
    int max_field_count_intermediate = 0;
};

} //namespace UserData_Impl

namespace {

using Pdata = UserData_Impl::UserData_Fields_Pdata;
using FieldView_t = UserData::FieldView_t;

}

UserData::Fields::Fields(ConfigMap& configMap, ForeachCell& foreach_cell)
  : pdata( std::make_unique<Pdata>(configMap, foreach_cell) )
{}

UserData::Fields::~Fields()
{}

const FieldView_t::Shape_t UserData::getShape() const
{
  return this->fields.pdata->getShape();
}

void UserData::new_fields( const std::set<std::string>& names)
{
  this->fields.pdata->new_fields(names);
}

void UserData::new_intermediate_fields( const std::set<std::string>& names)
{
  this->fields.pdata->new_intermediate_fields(names);
}

bool UserData::has_field(const std::string& name) const
{
  return this->fields.pdata->has_field(name);
}

std::set<std::string> UserData::getEnabledFields() const
{
  return this->fields.pdata->getEnabledFields();
}

const FieldView_t UserData::getField(const std::string& name) const
{
  return this->fields.pdata->getField(name);
}

void UserData::move_field( const std::string& dest, const std::string& src )
{
  this->fields.pdata->move_field(dest, src);
}

void UserData::delete_field( const std::string& name )
{
  this->fields.pdata->delete_field(name);
}

void UserData::clear_intermediates()
{
  this->fields.pdata->clear_intermediates();
}

void UserData::exchange_loadbalance( const ViewCommunicator& ghost_comm )
{
  this->fields.pdata->exchange_loadbalance(ghost_comm);
}

int UserData::nbFields() const
{
  return this->fields.pdata->nbFields();
}

UserData::FieldAccessor UserData::backup_and_realloc()
{
  return this->fields.pdata->backup_and_realloc();
}

void UserData::extend_fields()
{
  this->fields.pdata->extend_fields();
}

[[deprecated]] UserData::FieldAccessor_fulltree UserData::getAccessor_intermediates( const std::vector<UserData::FieldAccessor_FieldInfo>& fields_info ) const
{
  return getAccessor_fulltree(fields_info);
}

namespace UserData_Impl {

using FieldInfo = UserData_FieldAccessor_FieldInfo;

void FieldAccessor_init( const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        FieldView_t& fields,
                        FieldView_t& fields_intermediates
                      )
{
  fields = user_data.fields;
  if( has_intermediates )
  {
    fields_intermediates = user_data.fields_intermediate;
  }
}


void FieldAccessor_FieldManager_init_static( const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        int& _nbFields,
                        int* var_to_arrayindex,
                        int* ivar_to_arrayindex
                      )
{
    DYABLO_ASSERT_HOST_RELEASE( fields_info.size() > 0, "fields_info cannot be empty" );

    auto unknown_field_error = [&](std::string field_name)
    {
        std::stringstream s;
        s << "Could not find field '" << field_name << "' in UserData" << std::endl;
        s << "Available fields are :" << std::endl;
        for( auto& p : user_data.field_index )
        {
            s << " - '" << p.first << "'" << std::endl;
        }
        return s.str();
    };

    _nbFields = fields_info.size();

    DYABLO_ASSERT_HOST_RELEASE( _nbFields <= max_field_count, "Too many fields for FieldAccessor, use getAccessor<_MAX_FIELD_COUNT>() to increase number of fields." );
    for( [[maybe_unused]] const FieldInfo& info : fields_info )
    {
      DYABLO_ASSERT_HOST_RELEASE( info.id >= 0 && info.id < max_field_count, "VarIndex must be included in [0,MAX_FIELD_COUNT[. "
                                                                             "VarIndex is " << info.id << ", MAX_FIELD_COUNT is " << max_field_count  );
    }

    for(size_t i=0; i<max_field_count; i++)
    {
      ivar_to_arrayindex[i] = 0;
      var_to_arrayindex[i] = -1;
    }

    int i=0; 
    for( const FieldInfo& info : fields_info )
    {
        DYABLO_ASSERT_HOST_RELEASE( has_intermediates ? user_data.has_intermediate_field(info.name) : user_data.has_field(info.name), 
                                    unknown_field_error(info.name) );
        int index = has_intermediates ? 
                        user_data.field_index_intermediate.at(info.name).index 
                      : user_data.field_index.at(info.name).index;
        var_to_arrayindex[info.id] = index;
        ivar_to_arrayindex[i] = index;
        i++;
    }
}

FieldAccessor_FieldManager<-1>::FieldAccessor_FieldManager( const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info, bool has_intermediates )
  : var_to_arrayindex("var_to_arrayindex", fields_info.size()),
    ivar_to_arrayindex_device("ivar_to_arrayindex", fields_info.size()),
    ivar_to_arrayindex_host( Kokkos::create_mirror_view(ivar_to_arrayindex_device) )
{
  auto var_to_arrayindex_host = Kokkos::create_mirror_view(var_to_arrayindex);

  int dummy;
  FieldAccessor_FieldManager_init_static( user_data, fields_info,
    fields_info.size(), has_intermediates,
    dummy, ivar_to_arrayindex_host.data(), var_to_arrayindex_host.data() );

  Kokkos::deep_copy( var_to_arrayindex, var_to_arrayindex_host );
  Kokkos::deep_copy( ivar_to_arrayindex_device, ivar_to_arrayindex_host );
}


} // namespace UserData_Impl
} // namespace dyablo

