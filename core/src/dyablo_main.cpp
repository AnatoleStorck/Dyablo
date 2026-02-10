#include <string>

#include "DyabloSession.hpp"
#include "utils/config/ConfigMap.h"
#include "DyabloTimeLoop.h"

int main(int argc, char *argv[])
{
  using namespace dyablo;
  DyabloSession mpi_session(argc, argv);

  if( argc < 2 )
  {
    std::cout << "Error : no input file" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  ./dyablo [--kokkos-***=*] input_file.ini" << std::endl;
    return EXIT_FAILURE;
  }

  /*
   * read parameter file and initialize a ConfigMap object
   */
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap = ConfigMap::broadcast_parameters(input_file);
  if( configMap.hasValue("units","time") )
  { // Set code units
    auto unit_time = configMap.getValue<Units::Time>("units", "time");
    auto unit_length = configMap.getValue<Units::Length>("units", "length");
    Units::Mass unit_mass = Units::kg();
    DYABLO_ASSERT_HOST_RELEASE( !(configMap.hasValue("units","mass") && configMap.hasValue("units","density")), "Parsing units in .ini : cannot set code density and mass et the same time" );
    if( configMap.hasValue("units","density") )
    {
      auto unit_density = configMap.getValue<Units::Density>("units", "density");
      unit_mass = unit_density * unit_length.pow<3>();
    }
    else
    {
      unit_mass = configMap.getValue<Units::Mass>("units", "mass");
    }
    
    auto unit_mol = Units::mol();
    if( configMap.hasValue("units","particle_density") )
    {
      using ParticleDensity = decltype( Units::atom()/Units::m3() );
      auto unit_particle_density = configMap.getValue<ParticleDensity>("units", "particle_density");
      unit_mol = unit_particle_density * unit_length.pow<3>();
    }
    else if(configMap.hasValue("units","mol"))
    {
      unit_mol = configMap.getValue<Units::Mol>("units", "mol");
    }
    

    auto unit_current = Units::Ampere();
    auto unit_temp = Units::Kelvin();
    auto unit_luminousIntensity = Units::candela();


    Units::code_units_init( Units::UnitSystem(
        unit_time,
        unit_length,
        unit_mass,
        unit_current,
        unit_temp,
        unit_mol,
        unit_luminousIntensity
      ));
  }
  DyabloTimeLoop simulation( configMap );

  simulation.run();

  return EXIT_SUCCESS;

} // end main
