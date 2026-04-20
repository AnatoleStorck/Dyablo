#include "InitialConditions_base.h"
#include "../hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "../utils/io/HDF5ViewReader.h"
#include "refine_condition/RefineCondition.h"
#include "particles/ParticleUpdate.h"
#include "mpi/GhostCommunicator.h"

namespace dyablo {

struct GadgetHeader
{
    double BoxSize;
    bool Flag_Cooling;
    bool Flag_DoublePrecision;
    bool Flag_Feedback;
    bool Flag_Metals;
    bool Flag_Sfr;
    bool Flag_StellarAge;
    double h0;
    std::array<double, 6> MassTable;
    int NumFilesPerSnapshot;
    std::array<uint32_t, 6> NumPart_ThisFile;
    std::array<uint64_t, 6> NumPart_Total;
    std::array<uint32_t, 6> NumPart_Total_HighWord;
    double Omega0;
    double OmegaLambda;
    double Redshift;
    double ExpansionFactor;
    double Time;

    GadgetHeader ( HDF5ViewReader& reader ) :
            BoxSize         ( reader.read_attr<double>     ("Header", "BoxSize") ),
            Flag_Cooling    ( reader.read_attr<bool>       ("Header", "Flag_Cooling") ),
            Flag_DoublePrecision
                            ( reader.read_attr<bool>       ("Header", "Flag_DoublePrecision") ),
            Flag_Feedback   ( reader.read_attr<bool>       ("Header", "Flag_Feedback") ),
            // Flag_IC_Info    ( reader.read_attr<int32_t>    ("Header", "Flag_IC_Info") ),
            Flag_Metals     ( reader.read_attr<int32_t>    ("Header", "Flag_Metals") ),
            Flag_Sfr        ( reader.read_attr<bool>       ("Header", "Flag_Sfr") ),
            Flag_StellarAge ( reader.read_attr<bool>       ("Header", "Flag_StellarAge") ),
            h0              ( reader.read_attr<double>     ("Header", "HubbleParam") ),
            MassTable       ( reader.read_attr<double, 6>  ("Header", "MassTable") ),
            NumFilesPerSnapshot
                            ( reader.read_attr<int>        ("Header", "NumFilesPerSnapshot") ),
            NumPart_ThisFile( reader.read_attr<uint32_t, 6>("Header", "NumPart_ThisFile") ),
            NumPart_Total   ( reader.read_attr<uint64_t, 6>("Header", "NumPart_Total") ),
            NumPart_Total_HighWord
                            ( reader.read_attr<uint32_t, 6>("Header", "NumPart_Total_HighWord") ),
            Omega0          ( reader.read_attr<double>     ("Header", "Omega0") ),
            OmegaLambda     ( reader.read_attr<double>     ("Header", "OmegaLambda") ),
            Redshift        ( reader.read_attr<double>     ("Header", "Redshift") ),
            ExpansionFactor ( 1 / (1 + Redshift) ),
            Time            ( reader.read_attr<double>     ("Header", "Time") )
        {};
    GadgetHeader () = default;
};

enum VarIndex_particle_gadget { IVX, IVY, IVZ, IMASS, IID, IRHO_PART, IMETAL_PART, IHSML, IINTERNAL_ENERGY };
enum VarIndex_hydro { IRho, IRho_vx, IRho_vy, IRho_vz, IE_tot };

class InitialConditions_gadget : public InitialConditions
{
    using Policy = HyperbolicPolicy_Hydro;
    using PrimState = Policy::PrimState;
    using ConsState = Policy::ConsState;

    using CellIndex        = typename ForeachCell::CellIndex;
    using CellIndexOffset  = typename ForeachCell::CellIndex::offset_t;

    using pos_t = Kokkos::Array<real_t, 3>;

    ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;
    Timers& timers;
    typename Policy::Params policy_params;
    std::string filename;
    std::unique_ptr<RefineCondition> refine_condition;

    struct Data{
        ForeachCell& foreach_cell;
        int level_min, level_max;
    } data;

    HDF5ViewReader hdf5_reader;
    GadgetHeader header;

    real_t smallr, smallp;

public:
  InitialConditions_gadget(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    policy_params(Policy::getParams(configMap)),
    filename(configMap.getValue<std::string>("gadget", "inputFile")),
    refine_condition(
        RefineConditionFactory::make_instance(
            configMap.getValue<std::string>("gadget", "marker_kernel", "RefineCondition_mass"),
            configMap,
            foreach_cell,
            timers)
    ),
    data({
      foreach_cell,
      foreach_cell.get_amr_mesh().get_level_min(),
      foreach_cell.get_amr_mesh().get_level_max()
    }),
    hdf5_reader( filename ),
    header ( hdf5_reader ),
    smallr(configMap.getValue<real_t>("hydro", "smallr", 1e-10)),
    smallp(configMap.getValue<real_t>("hydro", "smallp", 1e-10))
  {
    real_t h0 = header.h0;
    auto box_size = header.BoxSize * Units::kpc() / h0;
    std::cout << "Gadget snapshot info:" << std::endl;
    std::cout << "  Box size :         " << Units::constant_to_code_units(box_size) << " code units ("
            << box_size.convert_to(Units::Mpc()) << " Mpc = "
            << box_size.convert_to(Units::Mpc()) * header.h0 << " Mpc/h)" << std::endl;
    std::cout << "  Redshift :         " << header.Redshift << std::endl;
    std::cout << "  Expansion factor : " << header.ExpansionFactor << std::endl;
    std::cout << "  Omega0 :           " << header.Omega0 << std::endl;
    std::cout << "  OmegaLambda :      " << header.OmegaLambda << std::endl;
    std::cout << "  HubbleParam :      " << header.h0 << std::endl;
    std::cout << "  Time :             " << header.Time << " (code units)" << std::endl;
    std::cout << "  NumPart_Total :    ";
    for (int i = 0; i < 6; i++)
    std::cout << header.NumPart_Total[i] + ( (uint64_t)header.NumPart_Total_HighWord[i] << 32 ) << (i < 5 ? ", " : "\n");
    std::cout << "  MassTable :        ";
    for (int i = 0; i < 6; i++)
    std::cout << header.MassTable[i] << (i < 5 ? ", " : "\n");

    // Make sure we are compatible with .ini
    DYABLO_ASSERT_HOST_RELEASE( header.NumFilesPerSnapshot == 1, "Only single file gadget snapshots are supported" );
    auto set_or_check = [&]( const std::string& section, const std::string& key, auto val ) {
        using T = decltype(val);

        if (configMap.hasValue(section, key)) {
            DYABLO_ASSERT_HOST_RELEASE( fabs(configMap.getValue< T >(section, key) - val) < 1e-12 * std::max(1., fabs(val)),
              ".ini parameter does not match gadget file : \n"
              << ".ini " << section << "/" << key << " : " << configMap.getValue< T >(section, key) << "\n"
              << filename << " : `" << Impl::to_string( val ) << "`"
            )
        }
        else
            configMap.getValue<T>(section, key, val);
    };

    set_or_check("mesh", "xmin", 0.);
    set_or_check("mesh", "xmax", Units::constant_to_code_units(box_size));
    set_or_check("mesh", "ymin", 0.);
    set_or_check("mesh", "ymax", Units::constant_to_code_units(box_size));
    set_or_check("mesh", "zmin", 0.);
    set_or_check("mesh", "zmax", Units::constant_to_code_units(box_size));
    set_or_check("cosmology", "omegam", header.Omega0);
    set_or_check("cosmology", "omegal", header.OmegaLambda);
    set_or_check("cosmology", "H0", header.h0 * 100);
  }

  void read_particles ( UserData& U ) {
    auto _read = [&]( const int itype, const std::string& gadget_part_name, const std::string& part_name ) {
        if (header.Flag_DoublePrecision)
            read_particles_helper<double>(U, header, hdf5_reader, itype, gadget_part_name, part_name);
        else
            read_particles_helper<float>(U, header, hdf5_reader, itype, gadget_part_name, part_name);
    };
    _read(0, "PartType0", "gas");
    _read(1, "PartType1", "dark_matter");
    _read(2, "PartType2", "disk");
    _read(3, "PartType3", "bulge");
    _read(4, "PartType4", "star");
    _read(5, "PartType5", "sink");

    std::cout << "Merge all star particles into single array" << std::endl;
    if (U.has_ParticleArray("star")) {
        if (U.has_ParticleArray("bulge")) {
            U.merge_particles_if("star", "bulge", "mass");
        }
        if (U.has_ParticleArray("disk")) {
            U.merge_particles_if("star", "disk", "mass");
        }
    }
  }

  template< typename T>
  void read_particles_helper(
    UserData& U,
    const GadgetHeader& header, HDF5ViewReader& hdf5_reader,
    const size_t itype, const std::string& gadget_part_name, const std::string& part_name
) {
    size_t Npart = header.NumPart_ThisFile[itype];
    if (Npart == 0) {
        std::cout << "  No particles of type " << itype << " (" << part_name << ")" << std::endl;
        return;
    } else {
        std::cout << "  Reading " << Npart << " particles of type " << itype << " (" << part_name << ")" << std::endl;
    }

    // Create new particle array
    U.new_ParticleArray(part_name, Npart);

    // Initialize attributes
    U.new_ParticleAttribute(part_name, "vx");
    U.new_ParticleAttribute(part_name, "vy");
    U.new_ParticleAttribute(part_name, "vz");
    U.new_ParticleAttribute(part_name, "mass");
    U.new_ParticleAttribute(part_name, "id");

    // Read data from file
    using Darr2D = Kokkos::View<T**, Kokkos::LayoutRight>;

    // Those always exist
    const auto& xp = hdf5_reader.read_dataset<Darr2D>(gadget_part_name + "/Coordinates");
    const auto& vp = hdf5_reader.read_dataset<Darr2D>(gadget_part_name + "/Velocities");
    const auto& idp = hdf5_reader.read_dataset<Kokkos::View<uint64_t*> >(gadget_part_name + "/ParticleIDs");

    DYABLO_ASSERT_HOST_RELEASE(
        (xp.extent(0) == Npart) && (vp.extent(0) == Npart) && (idp.extent(0) == Npart),
        "Inconsistent number of particles in gadget file for PartType" << itype );

    // Copy data to particle array
    auto P = U.getParticleArray( part_name );
    auto Pout = U.getParticleAccessor(
        part_name,
        { {"vx",   IVX},
          {"vy",   IVY},
          {"vz",   IVZ},
          {"mass", IMASS},
          {"id",   IID} } );


    real_t h0 = header.h0;
    real_t aexp = header.ExpansionFactor;
    auto km_per_s = Units::km() / Units::s();
    real_t len_unit = Units::constant_to_code_units(Units::kpc() / h0);
    real_t mass_unit = Units::constant_to_code_units(Units::Msun() * 1e10 / h0);
    real_t vel_unit = Units::constant_to_code_units(km_per_s / sqrt(aexp));

    Kokkos::parallel_for("InitGadgetParticles", Npart, KOKKOS_LAMBDA(const int i) {
        // Positions are in kpc/h, comoving
        P.pos( i, IX ) = xp(i, 0) * len_unit;
        P.pos( i, IY ) = xp(i, 1) * len_unit;
        P.pos( i, IZ ) = xp(i, 2) * len_unit;

        // Velocities are in km/s/sqrt(a), physical
        Pout.at(i, IVX) = vp(i, 0) * vel_unit;
        Pout.at(i, IVY) = vp(i, 1) * vel_unit;
        Pout.at(i, IVZ) = vp(i, 2) * vel_unit;
        Pout.at(i, IID) = idp(i);

        if (header.MassTable[itype] > 0) {
            Pout.at(i, IMASS) = header.MassTable[itype] * mass_unit;
        }
    });

    auto _read_optional = [&](const std::string& gadget_attr_name, const std::string& attr_name, const auto& units) {
        read_optional_helper(hdf5_reader, U, part_name, attr_name, gadget_part_name, gadget_attr_name, units, Npart);
    };

    _read_optional("Masses", "mass", mass_unit);
    _read_optional("Density", "rho", mass_unit / (len_unit*len_unit*len_unit));

    _read_optional("SmoothingLength", "hsml", len_unit);
    _read_optional("InternalEnergy", "u", vel_unit * vel_unit);

  }

   void read_optional_helper (
       HDF5ViewReader& hdf5_reader,
       UserData& U,
       const std::string& part_name,
       const std::string& attr_name,
       const std::string& gadget_part_name,
       const std::string& gadget_attr_name,
       real_t units,
       size_t Npart
   ) {
        if ( !hdf5_reader.has_dataset(gadget_part_name + "/" + gadget_attr_name) )
            return;
        std::cout << "    gadget." << gadget_attr_name << " → dyablo." << attr_name << std::endl;

        // Create associated particle attribute
        if (!U.has_ParticleAttribute(part_name, attr_name))
            U.new_ParticleAttribute(part_name, attr_name);

        // Read data
        auto data = hdf5_reader.read_dataset<Kokkos::View<double*> >(gadget_part_name + "/" + gadget_attr_name);
        DYABLO_ASSERT_HOST_RELEASE( data.extent(0) == Npart, "Inconsistent number of particles in gadget file for " << gadget_part_name << " " << gadget_attr_name );

        // Copy data to particle array
        auto Pout = U.getParticleAccessor(part_name, { {attr_name, 0} } );

        Kokkos::parallel_for("InitGadgetParticles_optional", Npart, KOKKOS_LAMBDA(const int i) {
            Pout.at_ivar(i, 0) = data(i) * units;
        });
  };

  void init ( UserData& U ) {
    this->read_particles(U);

    // ----------------------------------
    //  Create the AMR structure
    // ----------------------------------
    std::cout << "  Create AMR structure" << std::endl;

    ForeachCell& foreach_cell = data.foreach_cell;
    AMRmesh& pmesh     = foreach_cell.get_amr_mesh();

    int level_min = data.level_min;
    int level_max = data.level_max;

    // auto& analytical_formula = this->analytical_formula;
    std::vector<dyablo::UserData_fields::FieldAccessor_FieldInfo> fields_info = {
        {"rho", 0 },
        {"rho_vx", 1},
        {"rho_vy", 2},
        {"rho_vz", 3},
        {"e_tot", 4}
    };
    {
        std::set<std::string> new_fields;
        for( const auto& fi : fields_info)
            if( !U.has_field(fi.name) )
                new_fields.insert(fi.name);
        U.new_fields( new_fields );
    }

    auto SPH_interpolation = [&](const bool density_only)
    {
        if( !U.has_ParticleArray("gas") )
            return;

        auto Uout = U.getAccessor( fields_info );
        auto Ppos = U.getParticleArray("gas");
        auto Pdata = U.getParticleAccessor( "gas", {
            {"vx", IVX},
            {"vy", IVY},
            {"vz", IVZ},
            {"mass", IMASS},
            {"rho", IRHO_PART},
            // {"metal", IMETAL_PART},
            {"hsml", IHSML},
            {"u", IINTERNAL_ENERGY} } );

        ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

        foreach_particle.foreach_particle("gadget::SPH_interpolation", Ppos,
            KOKKOS_LAMBDA ( ParticleData::ParticleIndex& iPart )
        {
            const real_t part_mass = Pdata.at( iPart, IMASS );
            const pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
            const real_t hsml = Pdata.at( iPart, IHSML );
            const pos_t part_vel = {Pdata.at(iPart, IVX), Pdata.at(iPart, IVY), Pdata.at(iPart, IVZ)};
            const real_t v2 = SQR(part_vel[IX]) + SQR(part_vel[IY]) + SQR(part_vel[IZ]);
            const real_t u = Pdata.at(iPart, IINTERNAL_ENERGY);

            if( hsml <= 0 ) return;

            const CellIndex iCell = cells.getCellFromPos( part_pos );
            const int lvl = cells.getCellLevel( iCell );
            const pos_t cell_pos = cells.getCellCenter( iCell );
            const pos_t cell_size = cells.getCellSize( iCell );

            const real_t pi_hsml3_inv = 1/(M_PI * hsml*hsml*hsml);

            auto W = [&]( const pos_t X, const real_t Vcell, real_t& W_V) {
                real_t dx = (part_pos[IX] - X[IX]) / hsml * 2;
                real_t dy = (part_pos[IY] - X[IY]) / hsml * 2;
                real_t dz = (part_pos[IZ] - X[IZ]) / hsml * 2;
                real_t r2 = dx*dx + dy*dy + dz*dz;

                // Cubic spline kernel [Monaghan 1992]
                real_t q = sqrt(r2);
                real_t W;
                if (q > 2)
                    W = 0;
                else if (q < 1)
                    W = pi_hsml3_inv * (1 - 1.5 * q * q * (1 - 0.5 * q));
                else
                    W = pi_hsml3_inv * 0.25 * pow(2 - q, 3);
                W_V += W * Vcell;
            };

            real_t W_V_tot = 0;
            // We limit interpolation to one block
            short int ixmax = (short int)std::ceil(hsml / cell_size[IX] + .5);
            short int iymax = (short int)std::ceil(hsml / cell_size[IY] + .5);
            short int izmax = (short int)std::ceil(hsml / cell_size[IZ] + .5);

            // Compute total contribution of kernel
            for (short int iz = -izmax; iz <= izmax; iz++)
            for (short int iy = -iymax; iy <= iymax; iy++)
            for (short int ix = -ixmax; ix <= ixmax; ix++)
            {
                // Position if the neighbor cell had same size
                pos_t pos_neigh = {
                    cell_pos[IX] + ix * cell_size[IX],
                    cell_pos[IY] + iy * cell_size[IY],
                    cell_pos[IZ] + iz * cell_size[IZ]
                };
                const CellIndex iCell_neigh = cells.getCellFromPos( pos_neigh );
                const pos_t cell_size_neigh = cells.getCellSize( iCell_neigh );
                real_t Vcell = cell_size_neigh[IX]*cell_size_neigh[IY]*cell_size_neigh[IZ];
                int lvl_neigh = cells.getCellLevel( iCell_neigh );

                short int Nsub = 1;
                real_t shift = 0;
                real_t Vfactor = 1;
                if (lvl_neigh > lvl) {
                    // Neighbour is smaller: subdivide
                    Nsub = (1 << (lvl_neigh - lvl));
                    shift = 0.5;
                } else if (lvl_neigh < lvl) {
                    // Neighbour is larger: reduce volume
                    // since we're going to hit it multiple times
                    Vfactor /= pow(1 << (lvl - lvl_neigh), 3);
                }

                for (short int iz2 = 0; iz2 < Nsub; iz2++)
                for (short int iy2 = 0; iy2 < Nsub; iy2++)
                for (short int ix2 = 0; ix2 < Nsub; ix2++)
                {
                    pos_t pos_neigh2 = {
                        pos_neigh[IX] + ((real_t)ix2 / Nsub - shift) * cell_size[IX],
                        pos_neigh[IY] + ((real_t)iy2 / Nsub - shift) * cell_size[IY],
                        pos_neigh[IZ] + ((real_t)iz2 / Nsub - shift) * cell_size[IZ]
                    };
                    W( pos_neigh2, Vcell * Vfactor, W_V_tot );
                }
            }

            // Special case : if hsml is too small, deposit everything in the cell
            if (W_V_tot == 0)
            {
                real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];
                real_t drho = part_mass / Vcell;
                Kokkos::atomic_add( &Uout.at(iCell, IRho), drho );
                if (!density_only) {
                    const real_t Ek = 0.5 * drho * v2;
                    Kokkos::atomic_add( &Uout.at(iCell, IRho_vx), drho * part_vel[IX] );
                    Kokkos::atomic_add( &Uout.at(iCell, IRho_vy), drho * part_vel[IY] );
                    Kokkos::atomic_add( &Uout.at(iCell, IRho_vz), drho * part_vel[IZ] );
                    Kokkos::atomic_add( &Uout.at(iCell, IE_tot), Ek + u * drho);
                }
                return;
            }

            // Compute total contribution of kernel
            for (short int iz = -izmax; iz <= izmax; iz++)
            for (short int iy = -iymax; iy <= iymax; iy++)
            for (short int ix = -ixmax; ix <= ixmax; ix++)
            {
                // Position if the neighbor cell had same size
                pos_t pos_neigh = {
                    cell_pos[IX] + ix * cell_size[IX],
                    cell_pos[IY] + iy * cell_size[IY],
                    cell_pos[IZ] + iz * cell_size[IZ]
                };
                const CellIndex iCell_neigh = cells.getCellFromPos( pos_neigh );
                const pos_t cell_size_neigh = cells.getCellSize( iCell_neigh );
                real_t Vcell = cell_size_neigh[IX]*cell_size_neigh[IY]*cell_size_neigh[IZ];
                int lvl_neigh = cells.getCellLevel( iCell_neigh );

                short int Nsub = 1;
                real_t shift = 0;
                real_t Vfactor = 1;
                if (lvl_neigh > lvl) {
                    // Neighbour is smaller: subdivide
                    Nsub = (1 << (lvl_neigh - lvl));
                    shift = 0.5;
                } else if (lvl_neigh < lvl) {
                    // Neighbour is larger: reduce volume
                    // since we're going to hit it multiple times
                    Vfactor /= pow(1 << (lvl - lvl_neigh), 3);
                }

                for (short int iz2 = 0; iz2 < Nsub; iz2++)
                for (short int iy2 = 0; iy2 < Nsub; iy2++)
                for (short int ix2 = 0; ix2 < Nsub; ix2++)
                {
                    pos_t pos_neigh2 = {
                        pos_neigh[IX] + ((real_t)ix2 / Nsub - shift) * cell_size[IX],
                        pos_neigh[IY] + ((real_t)iy2 / Nsub - shift) * cell_size[IY],
                        pos_neigh[IZ] + ((real_t)iz2 / Nsub - shift) * cell_size[IZ]
                    };
                    real_t W_V = 0;
                    W( pos_neigh2, Vcell * Vfactor, W_V );
                    if( W_V == 0 ) continue;

                    CellIndex iCell_neigh2 = cells.getCellFromPos( pos_neigh2 );

                    W_V /= (W_V_tot + 1e-30);

                    real_t drho = part_mass * W_V / Vcell;

                    Kokkos::atomic_add( &Uout.at(iCell_neigh2, IRho), drho );
                    if (!density_only) {
                        const real_t Ek = 0.5 * drho * v2;
                        Kokkos::atomic_add( &Uout.at(iCell_neigh2, IRho_vx), drho * part_vel[IX] );
                        Kokkos::atomic_add( &Uout.at(iCell_neigh2, IRho_vy), drho * part_vel[IY] );
                        Kokkos::atomic_add( &Uout.at(iCell_neigh2, IRho_vz), drho * part_vel[IZ] );
                        Kokkos::atomic_add( &Uout.at(iCell_neigh2, IE_tot), Ek + u * drho);
                    }
                }
            }
        });
    };

    auto reset_U = [&]()
    {
        auto Uout = U.getAccessor( fields_info );
        real_t smallr = this->smallr;
        real_t smallp = this->smallp;
        foreach_cell.foreach_cell("InitialConditions_gadget::reset_U", Uout.getShape(),
            KOKKOS_LAMBDA ( const CellIndex& iCell )
        {
            Uout.at(iCell, IRho) = smallr;
            Uout.at(iCell, IRho_vx) = 0;
            Uout.at(iCell, IRho_vy) = 0;
            Uout.at(iCell, IRho_vz) = 0;
            Uout.at(iCell, IE_tot) = smallp;
        });

    };

    // Reallocate and fill U
    auto fill_U = [&]()
    {
        U.backup_and_realloc();

        ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
        const UserData::FieldAccessor Uout = U.getAccessor( fields_info );

        // Reset U fields
        reset_U();
        // Do SPH interpolation
        SPH_interpolation(true);

        int ghost_count = std::min( {U.getShape().bx, U.getShape().by, (uint32_t)4} );
        GhostCommunicator ghost_comm(pmesh, U.getShape(), ghost_count);
        ghost_comm.exchange_ghosts( Uout );
    };

    auto debug = [&](const std::string& msg)
    {   
        auto Uout = U.getAccessor( fields_info );
        ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

        foreach_cell.foreach_cell("gadget::SPH_interpolation_normalize",
            Uout.getShape(),
            KOKKOS_LAMBDA ( const CellIndex& iCell )
        {
            real_t rho = Uout.at(iCell, IRho);
            pos_t cell_size = cells.getCellSize( iCell );
            real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];
        });
    };

    std::cout << "Building AMR mesh" << std::endl;

    if( this->refine_condition )
    {
      RefineCondition& refine_condition = *(this->refine_condition);
      ScalarSimulationData scalar_data; // Empty scalardata
      // Refine until level_max using a RefineConditions plugin
      for (uint8_t level=level_min; level<level_max; ++level)
      {
          std::cout << " … Refinement level " << (int)level << ", Noct = " << pmesh.getNumOctants() << std::endl;
          fill_U();
          refine_condition.mark_cells(U, scalar_data);
          // Refine the mesh according to markers
          pmesh.adapt();
          // Load balance at each level to avoid excessive inbalance
          pmesh.loadBalance();
      }
    }
    pmesh.loadBalance();

    // Reallocate and fill U fields
    fill_U();

    std::cout << "SPH interpolation of particle density onto mesh" << std::endl;
    reset_U();
    SPH_interpolation(false);

    // Remove "gas" particles
    U.delete_ParticleArray("gas");

  }

};

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory,
                 dyablo::InitialConditions_gadget,
                 "gadget");