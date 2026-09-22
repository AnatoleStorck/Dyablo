#pragma once

#include "user_data/UserData.h"

namespace dyablo {

class GhostCommunicator_full_blocks;

namespace UserData_Impl{

struct UserData_FieldAccessor_FieldInfo
{
  std::string name; /// Name as in VarIndex.h
  VarIndex id; /// id to use to access with at()
};

void FieldAccessor_init(const UserData_Fields_Pdata& user_data, const std::vector<UserData_FieldAccessor_FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        UserData::FieldView_t& fields,
                        UserData::FieldView_t& fields_intermediates
                      );

void FieldAccessor_FieldManager_init_static( const UserData_Fields_Pdata& user_data, const std::vector<UserData_FieldAccessor_FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        int& _nbFields,
                        int* var_to_arrayindex,
                        int* ivar_to_arrayindex
                      );

template< int _MAX_FIELD_COUNT >
class FieldAccessor_FieldManager
{
private:
    static constexpr int MAX_FIELD_COUNT = _MAX_FIELD_COUNT;
    using FieldInfo = UserData_FieldAccessor_FieldInfo;

    int _nbFields;
    Kokkos::Array<int, MAX_FIELD_COUNT> var_to_arrayindex; // Index conversion for at()
    Kokkos::Array<int, MAX_FIELD_COUNT> ivar_to_arrayindex; // Index conversion for at_ivar()

public:
    FieldAccessor_FieldManager() = default;
    FieldAccessor_FieldManager(const FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager(FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager& operator=(const FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager& operator=(FieldAccessor_FieldManager& ) = default;

    FieldAccessor_FieldManager(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info, bool has_intermediates)
    {
        FieldAccessor_FieldManager_init_static( user_data, fields_info, 
                                                MAX_FIELD_COUNT, has_intermediates,
                                                this->_nbFields,
                                                this->var_to_arrayindex.data(),
                                                this->ivar_to_arrayindex.data() );
    }

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return _nbFields;
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return var_to_arrayindex[var];
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return ivar_to_arrayindex[ivar];
    }

    int get_index_from_ivar_host(int ivar) const
    {
        return ivar_to_arrayindex[ivar];
    }
};

template<>
class FieldAccessor_FieldManager<-1>
{
private:
    static constexpr int MAX_FIELD_COUNT = 3;
    using FieldInfo = UserData_FieldAccessor_FieldInfo;

    Kokkos::View<int*> var_to_arrayindex; // Index conversion for at()
    Kokkos::View<int*> ivar_to_arrayindex_device; // Index conversion for at_ivar()
    Kokkos::View<int*>::host_mirror_type ivar_to_arrayindex_host;

public:
    FieldAccessor_FieldManager() = default;
    FieldAccessor_FieldManager(const FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager(FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager& operator=(const FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager& operator=(FieldAccessor_FieldManager& ) = default;

    FieldAccessor_FieldManager(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info, bool has_intermediates);

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return var_to_arrayindex.size();
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return var_to_arrayindex(var);
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return ivar_to_arrayindex_device(ivar);
    }

    int get_index_from_ivar_host(int ivar) const
    {
        return ivar_to_arrayindex_host(ivar);
    }
};

template< bool has_intermediates, int _MAX_FIELD_COUNT >
class UserData_FieldAccessor_impl
{
friend GhostCommunicator_full_blocks;
friend UserData_Fields_Pdata;
public:
    static constexpr int MAX_FIELD_COUNT = _MAX_FIELD_COUNT;
    using FieldInfo = UserData_FieldAccessor_FieldInfo;
    using FieldView_t = UserData::FieldView_t;
    using FieldManager = FieldAccessor_FieldManager<MAX_FIELD_COUNT>;

    UserData_FieldAccessor_impl() = default;
    UserData_FieldAccessor_impl(const UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl(UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl& operator=(const UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl& operator=(UserData_FieldAccessor_impl& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return fm.nbFields();
    }

    UserData_FieldAccessor_impl(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info)
        : fm( user_data, fields_info, false )
    {
        if constexpr (has_intermediates)
            fm_intermediates = FieldManager( user_data, fields_info, true );

        UserData_Impl::FieldAccessor_init(
            user_data, fields_info,
            MAX_FIELD_COUNT, has_intermediates,
            this->fields, this->fields_intermediates
        );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ForeachCell::CellIndex& iCell, const VarIndex& varindex ) const
    {
        if constexpr ( has_intermediates )
        {
            
            if( iCell.iOct.isIntermediate )
            {
                ForeachCell::CellIndex iCell_intermediate = iCell;
                iCell_intermediate.iOct.isIntermediate=false;
                return fields_intermediates.at_ivar( iCell_intermediate, get_index_from_varindex_intermediates(varindex) );
            }
        }
        DYABLO_ASSERT_KOKKOS_DEBUG( !iCell.iOct.isIntermediate, "Accessing intermediates in array without intermediates" );
        return fields.at_ivar( iCell, get_index_from_varindex(varindex) );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachCell::CellIndex& iCell, int ivar ) const
    {
        if constexpr ( has_intermediates )
        {
            if( iCell.iOct.isIntermediate )
                return fields_intermediates.at_ivar( iCell, get_index_from_ivar_device_intermediates(ivar) );
        }
        DYABLO_ASSERT_KOKKOS_DEBUG( !iCell.iOct.isIntermediate, "Accessing intermediates in array without intermediates" );
        return fields.at_ivar( iCell, get_index_from_ivar_device(ivar) );
    }

    KOKKOS_INLINE_FUNCTION
    FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG(nbFields() > 0, "Cannot getShape() of an empty UserData_fields" );
        auto iter_space = fields.getShape();
        auto iter_space_inter = fields_intermediates.getShape();
        return FieldView_t::Shape_t{
            .bx = iter_space.bx,
            .by = iter_space.by,
            .bz = iter_space.bz,
            .nbFields = (uint32_t)nbFields(),
            .nbOcts = iter_space.nbOcts,
            .nbGhosts = iter_space.nbGhosts,
            .nbIntermediateOcts = iter_space_inter.nbOcts,
            .nbIntermediateGhosts = iter_space_inter.nbGhosts,
        };
    }

private:    
    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return fm.get_index_from_varindex(var);
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex_intermediates(VarIndex var) const
    {
        if constexpr ( has_intermediates )
            return fm_intermediates.get_index_from_varindex(var);
        else
            return 0; // no intermediates : callers guard on a runtime flag, see fm_intermediates
    }
protected:
    /// FieldManager holds 2 Kokkos::Array<int,MAX_FIELD_COUNT>, so it is by far the
    /// biggest part of an accessor (1444B at MAX_FIELD_COUNT=180). Accessors are
    /// captured by value in every kernel, and a functor above
    /// CudaTraits::KernelArgumentLimit (32kB) falls back to the GlobalMemory launch
    /// mechanism, which nvcc fails to compile. Only keep the intermediates manager
    /// at full size when it can actually be used : it is left default-constructed
    /// and never read when has_intermediates is false.
    using FieldManager_intermediates = FieldAccessor_FieldManager<has_intermediates ? _MAX_FIELD_COUNT : 1>;
    FieldAccessor_FieldManager<_MAX_FIELD_COUNT> fm;
    FieldManager_intermediates fm_intermediates;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return fm.get_index_from_ivar_device(ivar);
    }
    int get_index_from_ivar_host(int ivar) const
    {
        return fm.get_index_from_ivar_host(ivar);
    }
    FieldView_t fields;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device_intermediates(int ivar) const
    {
        if constexpr ( has_intermediates )
            return fm_intermediates.get_index_from_ivar_device(ivar);
        else
            return 0; // no intermediates : callers guard on a runtime flag, see fm_intermediates
    }
    int get_index_from_ivar_host_intermediates(int ivar) const
    {
        if constexpr ( has_intermediates )
            return fm_intermediates.get_index_from_ivar_host(ivar);
        else
            return 0; // no intermediates : callers guard on a runtime flag, see fm_intermediates
    }
    FieldView_t fields_intermediates;
};

} //namespace UserData_Impl
} //namespace dyablo