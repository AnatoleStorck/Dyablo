#pragma once

#include "InitialConditions_base.h"

#include "particles/ForeachParticle.h"

namespace dyablo{

class InitialConditions_particles_uniform : public InitialConditions
{
protected:
    ForeachParticle foreach_particle;
    std::vector<std::string> attributes;
    std::vector<real_t> values;

    /// Simple Constructor to build deriverd InitialConditions with static values with static fields and values
    InitialConditions_particles_uniform(ConfigMap& configMap, 
                                        ForeachCell& foreach_cell)
    : foreach_particle(foreach_cell.get_amr_mesh(), configMap)
    {}

public:
    InitialConditions_particles_uniform(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
        :  InitialConditions_particles_uniform(configMap, foreach_cell)
    {
        this->attributes = configMap.getValue<std::vector<std::string>>( "InitialConditions_particles_uniform", "attributes" );
        this->values = configMap.getValue<std::vector<real_t>>( "InitialConditions_particles_uniform", "values" );
    }


    void init( UserData& U )
    {
        DYABLO_ASSERT_HOST_RELEASE( values.size() >= attributes.size(), "InitialConditions_particles_uniform : too many fields, not enough values" );

        std::vector<UserData::ParticleAccessor::AttributeInfo> attr_info;

        // Sort array/attributes per array in attribute_map
        std::map< std::string, std::map<std::string, int> > attribute_map; // attribute_map[array_name] contains unique attribute infos 
        for( int ivar=0; ivar < (int)attributes.size(); ivar++ )
        {
            const std::string& full_attr = attributes[ivar];
            // Parse array and attribute name from "<array>/<attribute>" string
            size_t slashPos = full_attr.find_last_of("/");            
            auto trim = [](std::string& str)
            {
                str.erase(std::remove(str.begin(),str.end(),' '),str.end());
            };
            std::string array_name = full_attr.substr(0, slashPos);
            trim(array_name);
            std::string attr_name = full_attr.substr(slashPos + 1);
            trim(attr_name);

            if( !U.has_ParticleArray( array_name ) )
            {
                std::cout << "WARNING : particle array " << array_name << " does not exist : skipping InitialConditions_particles_uniform for '" << full_attr << "'" << std::endl;
                continue;
            }
            if( U.has_ParticleAttribute(array_name, attr_name) )
            {
                std::cout << "WARNING : attribute '" << full_attr << "' exists but will be overwritten by InitialConditions_particles_uniform" << std::endl;
            }
            attribute_map[ array_name ].insert( {attr_name, ivar} );
        }        

        // For each involved ParticleArrays
        for( const auto& pair : attribute_map )
        {   
            const std::string& array_name = pair.first;
            const std::map<std::string, int>& attributes = pair.second;

            // Allocate Attributes, Construct AttributeInfos and array_values
            int nb_attr = attributes.size();
            // AttributeInfo to create Accessor
            std::vector<UserData::ParticleAccessor::AttributeInfo> attr_info; 
            // Attributes Values for this Array
            Kokkos::View<real_t*> array_values("particle_values", nb_attr); 
            auto array_values_host = Kokkos::create_mirror_view(array_values); 
            
            for( const auto& pair : attributes )
            {
                const std::string& attr_name = pair.first;
                int ini_ivar = pair.second;

                if( !U.has_ParticleAttribute(array_name, attr_name) )
                    U.new_ParticleAttribute(array_name, attr_name);
                
                VarIndex ivar = attr_info.size();
                attr_info.push_back( {attr_name, ivar} ); 
                
                array_values_host( ivar ) = values[ ini_ivar ];
            }

            Kokkos::deep_copy( array_values, array_values_host );

            auto P = U.getParticleAccessor( array_name, attr_info );

            foreach_particle.foreach_particle("InitialConditions_particles_uniform::init", U.getParticleArray(array_name),
                KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) 
            { 
                for( int i=0; i<nb_attr; i++ )
                {
                    P.at( iPart, i ) = array_values( i );
                }
            });
        }
    }
};

} // namespace dyablo
