#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

namespace dyablo{

class InitialConditions_simple_particles : public InitialConditions{ 
    //ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;
    real_t gamma0;
    std::string array_name;
    int npart;
    Kokkos::View<double*> px,py,pz;
    std::vector<std::string> attribute_names;
    Kokkos::View<double**, Kokkos::LayoutLeft> attribute_values;
public:
  InitialConditions_simple_particles(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : //foreach_cell(foreach_cell),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
    array_name(configMap.getValue<std::string>("simple_particles", "array_name", "particles")),
    npart(configMap.getValue<int>("simple_particles", "npart", 1)),
    px( "px", npart ), py( "py", npart ), pz( "pz", npart ), 
    attribute_names(configMap.getValue< std::vector<std::string> >("simple_particles", "attributes", {"vx","vy","vz","mass"})),
    attribute_values( "attributes", npart, attribute_names.size() )
  {    
    auto parse_array = [&](const Kokkos::View<double*>& a, const std::string& var)
    {
      std::vector<double> values = configMap.getValue< std::vector<double> >("simple_particles", var, {});
      int nb_values = std::max((int)values.size(),npart); // Select at most npart values from .ini

      // Create unmanaged view to copy vector
      using UnmanagedHostView = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
      UnmanagedHostView values_host( values.data(), nb_values ); 
      
      auto values_device = Kokkos::subview( a, std::make_pair(0,nb_values) );

      Kokkos::deep_copy(values_device, values_host);
    };

    parse_array(px, "px");
    parse_array(py, "py");
    parse_array(pz, "pz");

    for( size_t i=0; i<attribute_names.size(); i++ )
    {
      std::string& attr_name = attribute_names[i];

      parse_array( Kokkos::subview(attribute_values, Kokkos::ALL(), i ), attr_name );
    }
  }

  void init( UserData& U )
  {
    // Setting up particles
    int rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();

    int npart_local = (rank==0) ? npart : 0;

    U.new_ParticleArray(array_name, npart_local);
    for( std::string& attr : attribute_names )
    {
      if( U.has_ParticleAttribute( array_name, attr ) )
        std::cout << "WARNING : attribute '" << array_name << "/" << attr << "' exists but will be overwritten by InitialConditions_simple_particles" << std::endl;
      else
        U.new_ParticleAttribute(array_name, attr);
    }

    if (rank == 0) { 

      const ForeachParticle::ParticleArray& P = U.getParticleArray(array_name); 

      std::vector<UserData::ParticleAccessor::AttributeInfo> attr_info;
      size_t nbAttr = attribute_names.size();
      for( size_t i=0; i<nbAttr; i++ )
      {
        attr_info.push_back( {attribute_names[i], (VarIndex)i} );
      }
      const UserData::ParticleAccessor Pdata = U.getParticleAccessor(array_name, attr_info);

      const Kokkos::View<double*>& px = this->px;
      const Kokkos::View<double*>& py = this->py;
      const Kokkos::View<double*>& pz = this->pz;
      Kokkos::View<double**, Kokkos::LayoutLeft>& attribute_values = this->attribute_values;

      foreach_particle.foreach_particle("InitialConditions_simple_particles", P,
        KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) 
      {      
        P.pos(iPart, IX) = px(iPart);
        P.pos(iPart, IY) = py(iPart);
        P.pos(iPart, IZ) = pz(iPart);

        for( size_t ivar = 0; ivar<nbAttr; ivar++ )
        {
          Pdata.at_ivar(iPart, ivar) = attribute_values(iPart, ivar);
        }
      });
    }

    U.distributeParticles(array_name);
  }  
}; 

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_simple_particles, 
                 "simple_particles");

