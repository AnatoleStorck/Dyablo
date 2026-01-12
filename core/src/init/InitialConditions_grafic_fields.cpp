#include "InitialConditions_base.h"
#include "utils/io/FortranBinaryReader.h"
#include "utils/units/Units.h"

namespace dyablo {

class InitialConditions_grafic_fields : public InitialConditions
{
  struct grafic_header
  {
    int32_t nx,ny,nz;
    float dx;
    float xo,yo,zo;
    float astart;
    float om,ov,H0;
  };

public:
  InitialConditions_grafic_fields(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : 
    foreach_cell(foreach_cell),
    n_tiles(configMap.getValue<std::vector<uint32_t>>("cosmology", "n_tiles", {1,1,1})),
    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro", "smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro", "smallc", 1e-10)),
    smallp(configMap.getValue<real_t>("hydro", "smallp", 1e-10))
  {
    {
      const AMRmesh& pmesh = foreach_cell.get_amr_mesh();
      int bx = foreach_cell.blockSize()[IX];
      int cell_level = pmesh.get_level_min() + std::log2(bx);
      this->graficDir = configMap.getValue<std::string>("grafic", "inputDir", std::string("data/IC_")+std::to_string(cell_level));
    }

    // Read header from grafic file
    grafic_header header;
    std::string grafic_filename = this->graficDir + "/ic_deltab";
    std::ifstream grafic_file(grafic_filename , std::ios::in|std::ios::binary );
    DYABLO_ASSERT_HOST_RELEASE(grafic_file, "Could not open grafic file : `" + grafic_filename + "`");
    FortranBinaryReader::read_record( grafic_file, &header, 1 );

    printHeader(header, graficDir);

    // Check if values are the same as .ini (or create new entries)
    auto set_or_check = [&]( std::string section, std::string param, auto val )
    {
      using T = decltype(val);
      if( configMap.hasValue(section, param) )
      {
        DYABLO_ASSERT_HOST_RELEASE( configMap.getValue< T >(section, param) == val, 
          ".ini parameter does not match grafic file : \n"
          << ".ini " << section << "/" << param << " : " << configMap.getValue< T >(section, param) << "\n"
          << grafic_filename << " : `" << Impl::to_string( val ) << "`"
           )
      }
      else
        configMap.getValue< T >(section, param, val);
    };

    auto code_time = Units::code_units().getUnit(Units::s());
    auto code_length = Units::code_units().getUnit(Units::m());
    int ndim = foreach_cell.getDim();

    DYABLO_ASSERT_HOST_RELEASE(int(n_tiles.size()) == ndim, "n_tiles should be of length " << ndim << ", got instead " << n_tiles.size());
    DYABLO_ASSERT_HOST_RELEASE( 0 == (n_tiles[0]*header.nx)%foreach_cell.blockSize()[IX], "Block size (x) is not compatible with grafic file" );
    DYABLO_ASSERT_HOST_RELEASE( 0 == (n_tiles[1]*header.ny)%foreach_cell.blockSize()[IY], "Block size (y) is not compatible with grafic file" );
    DYABLO_ASSERT_HOST_RELEASE( 0 == (n_tiles[2]*header.nz)%foreach_cell.blockSize()[IZ], "Block size (z) is not compatible with grafic file" );
    set_or_check( "amr", "coarse_oct_resolution_x", n_tiles[0]*header.nx/foreach_cell.blockSize()[IX] );
    set_or_check( "amr", "coarse_oct_resolution_y", n_tiles[1]*header.ny/foreach_cell.blockSize()[IY] );
    set_or_check( "amr", "coarse_oct_resolution_z", n_tiles[2]*header.nz/foreach_cell.blockSize()[IZ] );
    this->xmin = header.xo * Units::Mpc().convert_to( code_length );
    set_or_check( "mesh", "xmin", this->xmin );
    this->ymin = header.yo * Units::Mpc().convert_to( code_length );
    set_or_check( "mesh", "ymin", this->ymin );
    this->zmin = header.zo * Units::Mpc().convert_to( code_length );
    set_or_check( "mesh", "zmin", this->zmin );

    this->astart = header.astart;
    set_or_check( "cosmology", "astart", header.astart );
    this->omegam = header.om;
    set_or_check( "cosmology", "omegam", omegam );
    set_or_check( "cosmology", "omegav", header.ov );
    this->omegab = configMap.getValue<real_t>("cosmology", "omegab", 0.049);

    auto H0_u = header.H0 * Units::km() / Units::s() / Units::Mpc(); // Hubble constant from grafic
    this->H0 = H0_u.convert_to( 1/code_time );

    {
      using Inv_Time = decltype( 1/Units::s() );
      real_t H0_ini = configMap.getValue_in_code_unit<Inv_Time>("cosmology", "H0");
      DYABLO_ASSERT_HOST_RELEASE( H0 == H0_ini, 
          ".ini parameter does not match grafic file : \n"
          << ".ini cosmology/H0 : " << H0_ini << "\n"
          << "grafic file : `" << this->H0 << "`"
           );
    }

    real_t dx = (header.dx * Units::Mpc()).convert_to(code_length);
    set_or_check( "cosmology", "dx", dx );
    this->xmax = xmin + n_tiles[0]*header.nx * dx;
    set_or_check( "mesh", "xmax", xmax  );
    this->ymax = ymin + n_tiles[1]*header.ny * dx;
    set_or_check( "mesh", "ymax", ymax  );
    this->zmax = zmin + n_tiles[2]*header.nz * dx;
    set_or_check( "mesh", "zmax", zmax  );

    real_t four_pi_G = configMap.getValue<real_t>("gravity", "4_Pi_G", 4 * M_PI * Units::constant_to_code_units(Units::NEWTON_G()));
    this->rhoc = (3. * H0 * H0 / (2 * four_pi_G)); // comoving critical density 

    {
      // Mean mass for a raw cell used for refinement
      double mass0 = rhoc * omegam * (dx * dx * dx) / (n_tiles[0] * n_tiles[1] * n_tiles[2]);
      std::cout << "mean mass per cell (code Units)=" << mass0 << std::endl;

      real_t mass_coarsen_factor = configMap.getValue<real_t>("cosmology", "mass_coarsen_factor", 0.1);
      real_t mass_refine_factor = configMap.getValue<real_t>("cosmology", "mass_refine_factor", 1.0);

      set_or_check("amr", "mass_coarsen", mass0*mass_coarsen_factor);
      set_or_check("amr", "mass_refine", mass0*mass_refine_factor);
    }

    real_t cosmo_z = 1. / astart - 1.;
    auto temp_u = 317.5 * Units::Kelvin() * (cosmo_z * cosmo_z) / (151.0 * 151.0);
    this->temp = temp_u.convert_to( Units::code_units().getUnit<Units::Temperature>() );
  }

  void init( UserData& U )
  {
    ForeachCell& foreach_cell = this->foreach_cell;

    // Read header from ic_deltab
    grafic_header header;
    {
      // Open grafic file ic_deltab
      std::string grafic_filename = this->graficDir + "/ic_deltab";
      std::ifstream grafic_file(grafic_filename , std::ios::in|std::ios::binary );
      DYABLO_ASSERT_HOST_RELEASE(grafic_file, "Could not open grafic file : `" + grafic_filename + "`");
      FortranBinaryReader::read_record( grafic_file, &header, 1 );
    }

    int32_t nx = header.nx;
    int32_t ny = header.ny;
    int32_t nz = header.nz;    

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    auto fill_field = [&](const std::string& grafic_file_name, const std::string& dyablo_field_name)
    {
      Kokkos::View< float***, Kokkos::LayoutLeft > grafic_field_device ( 
          grafic_file_name,
          nx, ny, nz
      );

      {
        // Open grafic file
        std::string grafic_filename = this->graficDir + "/" + grafic_file_name;
        std::ifstream grafic_file(grafic_filename , std::ios::in|std::ios::binary );
        DYABLO_ASSERT_HOST_RELEASE(grafic_file, "Could not open grafic file : `" + grafic_filename + "`");
        // Read header from grafic file
        grafic_header header;
        FortranBinaryReader::read_record( grafic_file, &header, 1 );
        // Read array from grafic file
        auto grafic_field_host = Kokkos::create_mirror_view( grafic_field_device );
        for(uint32_t z=0; z<(uint32_t)nz; z++)
          FortranBinaryReader::read_record( grafic_file, &grafic_field_host(0,0,z), nx*ny );
        Kokkos::deep_copy( grafic_field_device, grafic_field_host );
      }

      enum VarIndex { Ifield };
      UserData::FieldAccessor U_field = U.getAccessor({{dyablo_field_name, Ifield}});

      real_t xmin = this->xmin;
      real_t ymin = this->ymin;
      real_t zmin = this->zmin;

      foreach_cell.foreach_cell( "InitialConditions_grafic_fields::init_field", U_field.getShape(),
        KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
      {
        auto c = cellmetadata.getCellCenter( iCell );
        auto s = cellmetadata.getCellSize( iCell );
        uint32_t ix = (c[IX] - xmin) / s[IX];
        uint32_t iy = (c[IY] - ymin) / s[IY];
        uint32_t iz = (c[IZ] - zmin) / s[IZ];

        U_field.at( iCell, Ifield ) = grafic_field_device(ix % nx, iy % ny, iz % nz);
      });
    };

    // Sequentially read density and velocities from grafic file
    U.new_fields({"rho","e_tot","rho_vx","rho_vy","rho_vz"});
    fill_field( "ic_deltab", "rho" );
    fill_field( "ic_velbx", "rho_vx" );
    fill_field( "ic_velby", "rho_vy" );
    fill_field( "ic_velbz", "rho_vz" );

    enum VarIndex_hydro {ID, IE, IU, IV, IW};
    UserData::FieldAccessor Uinout = U.getAccessor({
        {"rho", ID},
        {"e_tot", IE},
        {"rho_vx", IU},
        {"rho_vy", IV},
        {"rho_vz", IW}
    });

    // Parameters
    real_t gamma0 = this->gamma0;
    real_t omegab = this->omegab;
    real_t astart = this->astart;
    real_t smallp = this->smallp;
    real_t rhoc = this->rhoc;
    real_t temp = this->temp;

    auto grafic_vel = Units::km() / Units::s();
    auto code_density = Units::code_units().getUnit(Units::kg() / Units::m3());
    auto code_vel = Units::code_units().getUnit(grafic_vel);
    auto code_pressure = Units::code_units().getUnit(Units::Pa());

    foreach_cell.foreach_cell( "InitialConditions_grafic_fields::compute_conservative", Uinout.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      real_t cosmo_contrast = Uinout.at( iCell, ID ); // contrast is (rho - rho_mean) / rho_mean
      real_t cosmo_velx = Uinout.at( iCell, IU );
      real_t cosmo_vely = Uinout.at( iCell, IV );
      real_t cosmo_velz = Uinout.at( iCell, IW );

      real_t u = (cosmo_velx * grafic_vel).convert_to(code_vel) * astart;
      real_t v = (cosmo_vely * grafic_vel).convert_to(code_vel) * astart;
      real_t w = (cosmo_velz * grafic_vel).convert_to(code_vel) * astart;
      real_t u2 = u * u + v * v + w * w;

      real_t rho_tot = (cosmo_contrast + 1.0) * rhoc;

      real_t rho = omegab * rho_tot;
      real_t rho_u = rho * u;
      real_t rho_v = rho * v;
      real_t rho_w = rho * w;

      auto cosmo_rhob = rho_tot * omegab / (astart * astart * astart) * code_density;
      auto cosmo_pressure_u = (gamma0 - 1.0) * 1.5 * (cosmo_rhob * (1. - Units::YHE()) / Units::PROTON_MASS() * (1. + Units::yHE())) * Units::KBOLTZ() * (temp * Units::Kelvin());
      real_t cosmo_pressure = cosmo_pressure_u.convert_to( code_pressure );
      real_t p = fmax( cosmo_pressure * (astart * astart * astart * astart * astart), smallp );
      real_t e_tot = rho*u2/2.0 + p/(gamma0-1.0);

      // Hydro
      Uinout.at( iCell, ID ) = rho;
      Uinout.at( iCell, IE ) = e_tot;
      Uinout.at( iCell, IU ) = rho_u;
      Uinout.at( iCell, IV ) = rho_v;
      Uinout.at( iCell, IW ) = rho_w;
    });

  }

/**
 * Print header
*/
void printHeader(grafic_header hdr, std::string graficDir){

 std::cout << "GRAFIC : reading initial Conditions from " << graficDir << std::endl
           << "nx = " << hdr.nx << std::endl
           << "ny = " << hdr.ny << std::endl
           << "nz = " << hdr.nz << std::endl
           << "x0 = " << hdr.xo << std::endl
           << "y0 = " << hdr.yo << std::endl
           << "z0 = " << hdr.zo << std::endl
           << "astart = " << hdr.astart << std::endl
           << "om = " << hdr.om << std::endl
           << "ov = " << hdr.ov << std::endl
           << "H0 = " << hdr.H0 << " [km/s/Mpc]" << std::endl
           << "dx = " << hdr.dx << " [Mpc]" << std::endl
           << "box size = " << hdr.nx * hdr.dx << " [Mpc]" << std::endl
           << "box size = " << hdr.nx * hdr.dx * hdr.H0/100.0 << " [Mpc/h]" << std::endl
      //     << "mass0 = " << mass0 << std::endl
           << "##########################" << std::endl;
}

private:
  std::string graficDir;
  ForeachCell& foreach_cell;
  
  // Cosmo params
  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;
  real_t omegab;
  real_t astart, omegam;
  real_t H0;
  real_t rhoc;

  std::vector<uint32_t> n_tiles;
  real_t gamma0;
  real_t smallr,smallc,smallp;

  real_t temp;

};

} //namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_grafic_fields, 
                 "grafic_fields");