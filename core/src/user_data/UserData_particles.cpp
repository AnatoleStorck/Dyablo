#include "ParticleAccessor.h"

namespace dyablo {

namespace UserData_Impl{
namespace{

class ParticleContainer
{
public:
    ParticleContainer( const ParticleContainer& ) = default;
    ParticleContainer( ParticleContainer&& ) = default;
    ParticleContainer( const std::string& name, const ForeachParticle& foreach_particle, uint32_t num_particles )
      : name(name), foreach_particle(foreach_particle), particles(name, num_particles, 0)
    {}
    int nbAttributes() const
    {
        return attribute_index.size();
    }
    void new_ParticleAttribute( const std::string& attribute_name )
    {
        int nb_new_attributes = 1;
        int needed_attr_count = nbAttributes() + nb_new_attributes;
        this->max_particle_count = std::max( this->max_particle_count, needed_attr_count );
        int allocated_attr_count = nbAttributes();
        if( needed_attr_count > allocated_attr_count )
        {
            ParticleData particles_new( particles, needed_attr_count );
            if( allocated_attr_count != 0 )
            {
                Kokkos::deep_copy( 
                    Kokkos::subview( particles_new.particle_data, Kokkos::ALL(), std::pair(0,allocated_attr_count) ),
                    particles.particle_data
                );
            }
            this->particles = particles_new;
        }

        for( const std::string& name : {attribute_name} )
        {
            DYABLO_ASSERT_HOST_RELEASE( !has_ParticleAttribute(name), "new_ParticleAttribute() - attribute already exist : " << name );
        
            auto first_free = [&]() -> int
            {
                for(int i=0; i<particles.nbAttributes(); i++)
                {
                    bool free = true;
                    for( auto& p : attribute_index )
                    {
                        if(p.second == i)
                            free = false;
                    }
                    if( free ) return i;
                }
                DYABLO_ASSERT_HOST_RELEASE(false, "new_ParticleAttribute internal error : not enough fields allocated");
                return -1;
            };

            int index = first_free();
            attribute_index[name] = index;
            
            const auto& particles = this->particles;

            foreach_particle.foreach_particle( "zero_attribute", particles,
                KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
            {
                particles.at_ivar(iPart, index) = 0;
            });
        }
        
    }
    bool has_ParticleAttribute( const std::string& name ) const
    {
        return attribute_index.end() != attribute_index.find(name); 
    }
    std::set<std::string> getEnabledParticleAttributes() const
    {
        std::set<std::string> res;
        for( const auto& p : attribute_index )
        {
            res.insert( p.first );
        }
        return res;
    }
    ParticleArray getParticleArray() const
    {
        return particles;
    }
    UserData::ParticleAttribute_t getParticleAttribute( const std::string& attribute_name ) const
    {
      ParticleData res( particles, 1 );

      int index = attribute_index.at(attribute_name);

      const auto& particles = this->particles;
      foreach_particle.foreach_particle( "copy_attr", particles,
          KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
      {
          res.at_ivar(iPart, 0) = particles.at_ivar( iPart, index );
      });

      return res;
    }
    void move_ParticleAttribute( const std::string& attr_dest, const std::string& attr_src )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleAttribute(attr_src), "move_ParticleAttribute() - source attribute doesn't exist : " << attr_src);

        attribute_index[ attr_dest ] = attribute_index.at(attr_src);
        attribute_index.erase( attr_src );
    }
    void delete_ParticleAttribute( const std::string& attribute_name)
    {
        attribute_index.erase( attribute_name );
    }
    void resize( uint32_t new_num_particles )
    {
        uint32_t old_size = particles.getNumParticles();
        if (new_num_particles == old_size) return;

        size_t Nprops = particles.particle_data.extent(1);
        Kokkos::resize(particles.particle_data, new_num_particles, Nprops);
        size_t Ndim = particles.particle_position.extent(1);
        Kokkos::resize(particles.particle_position, new_num_particles, Ndim);
    }
    void distributeParticles()
    {
        ViewCommunicator part_comm = foreach_particle.get_distribute_communicator( particles );
        uint32_t nbParticles_new = part_comm.getNumGhosts();

        ParticleData particles_new( this->name, nbParticles_new, particles.nbAttributes() );

        part_comm.exchange_ghosts<0>( particles.particle_data, particles_new.particle_data );
        part_comm.exchange_ghosts<0>( particles.particle_position, particles_new.particle_position );

        this->particles = particles_new;
    }

//private:
    std::string name;
    ForeachParticle foreach_particle;
    ParticleData particles;
    std::map<std::string, int> attribute_index;
    int max_particle_count = 0;
};

} // namespace

struct UserData_Particles_Pdata
{
public: 
    UserData_Particles_Pdata( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_particle( foreach_cell.get_amr_mesh(), configMap )
    {}    

    bool has_ParticleArray( const std::string& array_name )
    {
      return particle_containers.count(array_name) == 1;
    }

    ParticleContainer& getParticleContainer( const std::string& array_name )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(array_name), "Particle array does not exist : " << array_name );
        return particle_containers.at(array_name);
    }

    const ParticleContainer& getParticleContainer( const std::string& array_name ) const
    {
      return const_cast<UserData_Particles_Pdata*>(this)->getParticleContainer(array_name);
    }

    std::set<std::string> getEnabledParticleArrays() const
    {
        std::set<std::string> res;
        for( const auto& p : particle_containers )
        {
            res.insert( p.first );
        }
        return res;
    }

    void delete_ParticleArray( const std::string& name )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(name), "UserData_particles::delete_ParticleArray() - particle array does not exist : " << name );
        particle_containers.erase( name );
    }

//private:
    ForeachParticle foreach_particle;
    std::map<std::string, ParticleContainer> particle_containers;
};

} //namespace UserData_Impl

namespace {

using Pdata = UserData_Impl::UserData_Particles_Pdata;
using ParticleContainer = UserData_Impl::ParticleContainer;

}

UserData::Particles::Particles(ConfigMap& configMap, ForeachCell& foreach_cell)
  : pdata( std::make_unique<Pdata>(configMap, foreach_cell) )
{/*empty*/}

UserData::Particles::~Particles()
{/*empty*/}

void UserData::new_ParticleArray( const std::string& name, uint32_t num_particles )
{
  auto& pdata = *(this->particles.pdata);

  DYABLO_ASSERT_HOST_RELEASE( !this->has_ParticleArray(name), "new_ParticleArray() - particle array already exists : " << name );
  pdata.particle_containers.emplace( name, ParticleContainer(name, pdata.foreach_particle, num_particles) );
}

void UserData::new_ParticleAttribute( const std::string& array_name, const std::string& attribute_name )
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  DYABLO_ASSERT_HOST_RELEASE( !array.has_ParticleAttribute(attribute_name), "UserData_particles::new_ParticleAttribute() - particle attribute already exists : " << attribute_name );
  array.new_ParticleAttribute( attribute_name );
}

bool UserData::has_ParticleArray(const std::string& name) const
{
  return this->particles.pdata->has_ParticleArray(name);
}

std::set<std::string> UserData::getEnabledParticleArrays() const
{
  return this->particles.pdata->getEnabledParticleArrays();
}

bool UserData::has_ParticleAttribute(const std::string& array_name, const std::string& attribute_name ) const
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.has_ParticleAttribute( attribute_name );
}

UserData::ParticleArray_t UserData::getParticleArray( const std::string& array_name ) const
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.getParticleArray();
}

std::set<std::string> UserData::getEnabledParticleAttributes( const std::string& array_name ) const
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.getEnabledParticleAttributes();
}

UserData::ParticleAttribute_t UserData::getParticleAttribute( const std::string& array_name, const std::string& attribute_name ) const
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.getParticleAttribute(attribute_name);
}
  
void UserData::move_ParticleAttribute( const std::string& array_name, const std::string& attr_dest, const std::string& attr_src )
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.move_ParticleAttribute(attr_dest, attr_src); 
}
  
void UserData::delete_ParticleAttribute(const std::string& array_name, const std::string& attribute_name)
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.delete_ParticleAttribute(attribute_name);
}

void UserData::distributeParticles( const std::string& array_name )
{
  ParticleContainer& array = this->particles.pdata->getParticleContainer( array_name );
  return array.distributeParticles();
}
  
void UserData::distributeAllParticles()
{
  for(auto& pair : this->particles.pdata->particle_containers)
    pair.second.distributeParticles();
}

void UserData::delete_ParticleArray( const std::string& name )
{
  this->particles.pdata->delete_ParticleArray( name );
}

void UserData::merge_particles_if( const std::string& id_dest, const std::string& id_to_merge, const std::string& mask_field )
{
    auto& pdata = *this->particles.pdata;
    DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(id_dest), "merge_particles_if error : destination array '"<<id_dest<<"' does not exist" );
    DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(id_to_merge), "merge_particles_if error : array to merge '"<<id_to_merge<<"' does not exist" );
    DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleAttribute(id_to_merge, mask_field), "merge_particles_if error : mask field '"<<id_to_merge<<"/"<<mask_field<<"' does not exist" );

    std::vector<ParticleAccessor::AttributeInfo> all_attr;
    VarIndex Imask = -1;
    {
        for( const std::string& attr : this->getEnabledParticleAttributes( id_dest ) )
        {
            VarIndex ivar = all_attr.size();
            all_attr.push_back({attr, ivar});
            if( attr == mask_field )
            {
                Imask = ivar;
            }
        }
    }
    int nb_attr = all_attr.size();

    int n_to_merge = 0;
    {
        auto Pto_merge = this->getParticleArray(id_to_merge);
        auto Pmask = this->getParticleAccessor(id_to_merge, {{mask_field, 0}});
        pdata.foreach_particle.reduce_particle("count_to_merge", Pto_merge,
            KOKKOS_LAMBDA(const ForeachParticle::ParticleIndex& iPart, int& count)
        {
            if( Pmask.at(iPart, 0) != 0.0 )
                count++;
        }, n_to_merge);
    }

    auto& Pout_container = pdata.getParticleContainer( id_dest );
    uint32_t Pout_initial_size = Pout_container.getParticleArray().getNumParticles();
    Pout_container.resize( Pout_initial_size + n_to_merge );

    {
        auto Pin = this->getParticleArray(id_to_merge);
        auto Pin_data = this->getParticleAccessor( id_to_merge, all_attr );

        auto Pout = this->getParticleArray(id_dest);
        auto Pout_data = this->getParticleAccessor( id_dest, all_attr );

        Kokkos::View<uint32_t> count("count");
        pdata.foreach_particle.foreach_particle( "merge_copy", Pin,
            KOKKOS_LAMBDA(const ForeachParticle::ParticleIndex& iPart_in)
        {
            if( Pin_data.at_ivar(iPart_in, Imask) != 0 )
            {
                ForeachParticle::ParticleIndex iPart_out = Pout_initial_size + Kokkos::atomic_fetch_add(&count(), 1);

                Pout.pos( iPart_out, IX ) = Pin.pos( iPart_in, IX );
                Pout.pos( iPart_out, IY ) = Pin.pos( iPart_in, IY );
                Pout.pos( iPart_out, IZ ) = Pin.pos( iPart_in, IZ );

                for( int i=0; i<nb_attr; i++ )
                {
                    Pout_data.at_ivar(iPart_out, i) = Pin_data.at_ivar(iPart_in, i);
                }
            }
        });
    }

    this->delete_ParticleArray(id_to_merge);
}

void UserData::move_ParticleArray( const std::string& dest, const std::string& src )
{
    auto& pdata = *this->particles.pdata;
    DYABLO_ASSERT_HOST_RELEASE( pdata.has_ParticleArray(src), "move_ParticleArray() - source particle array does not exist : " << src );

    if( dest == src ) return;

    pdata.particle_containers.erase( dest );
    auto node = pdata.particle_containers.extract( src );
    node.key() = dest;
    // ParticleContainer::name labels the views reallocated by distributeParticles()
    node.mapped().name = dest;
    pdata.particle_containers.insert( std::move(node) );
}

UserData::ParticleAccessor UserData::getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const
{
    return ParticleAccessor( *this->particles.pdata, array_name, attribute_info );
}

namespace UserData_Impl {

UserData_ParticleAccessor::UserData_ParticleAccessor(const UserData_Particles_Pdata& user_data, const std::string& array_name, const std::vector<UserData::ParticleAccessor_AttributeInfo>& attr_info)
 : particles(user_data.getParticleContainer( array_name ).particles)
{
  auto& particles = user_data.getParticleContainer( array_name );

  DYABLO_ASSERT_HOST_RELEASE( attr_info.size() > 0, "fields_info cannot be empty" );

  auto unknown_attr_error = [&](std::string attr_name)
  {
      std::stringstream s;
      s << "Could not find particle attribute '" << attr_name << "' in ParticleArray" << std::endl;
      s << "Available attributes are :" << std::endl;
      for( const std::string& particle_attr : particles.getEnabledParticleAttributes() )
      {
          s << " - '" << particle_attr << "'" << std::endl;
      }
      return s.str();
  };

  int max_varindex = 0;
  for( const AttributeInfo& info : attr_info )
  {
      max_varindex = std::max( max_varindex, info.id );
  }

  this->var_to_arrayindex = Kokkos::View<int*>( "varindex_to_viewindex", max_varindex+1 );
  this->ivar_to_arrayindex = Kokkos::View<int*>( "ivar_to_viewindex", attr_info.size() );
  auto var_to_arrayindex_host = Kokkos::create_mirror_view( this->var_to_arrayindex );
  auto ivar_to_arrayindex_host = Kokkos::create_mirror_view( this->ivar_to_arrayindex );
  for(size_t i=0; i<var_to_arrayindex_host.size(); i++)
      var_to_arrayindex_host(i) = -1;

  int i=0; 
  for( const AttributeInfo& info : attr_info )
  {
      DYABLO_ASSERT_HOST_RELEASE( particles.has_ParticleAttribute(info.name),
                                  unknown_attr_error(info.name));
      int index = particles.attribute_index.at(info.name);
      var_to_arrayindex_host(info.id) = index;
      ivar_to_arrayindex_host(i) = index;
      i++;
  }
  Kokkos::deep_copy( this->var_to_arrayindex, var_to_arrayindex_host );
  Kokkos::deep_copy( this->ivar_to_arrayindex, ivar_to_arrayindex_host );
}

} // namespace UserData_Impl
} // namespace dyablo