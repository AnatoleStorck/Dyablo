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
    bool Flag_IC_Info;
    bool Flag_Metals;
    bool Flag_Sfr;
    bool Flag_StellarAge;
    double HubbleParam;
    std::array<double, 6> MassTable;
    uint32_t NumFilesPerSnapshot;
    std::array<uint32_t, 6> NumPart_ThisFile;
    std::array<uint32_t, 6> NumPart_Total;
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
            Flag_IC_Info    ( reader.read_attr<bool>       ("Header", "Flag_IC_Info") ),
            Flag_Metals     ( reader.read_attr<bool>       ("Header", "Flag_Metals") ),
            Flag_Sfr        ( reader.read_attr<bool>       ("Header", "Flag_Sfr") ),
            Flag_StellarAge ( reader.read_attr<bool>       ("Header", "Flag_StellarAge") ),
            HubbleParam     ( reader.read_attr<double>     ("Header", "HubbleParam") ),
            MassTable       ( reader.read_attr<double, 6>  ("Header", "MassTable") ),
            NumFilesPerSnapshot 
                            ( reader.read_attr<uint32_t>   ("Header", "NumFilesPerSnapshot") ),
            NumPart_ThisFile( reader.read_attr<uint32_t, 6>("Header", "NumPart_ThisFile") ),
            NumPart_Total   ( reader.read_attr<uint32_t, 6>("Header", "NumPart_Total") ),
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

enum VarIndex_particle_gadget { IVX, IVY, IVZ, IMASS, IID, IRho, IMetal, ISmoothingLength, IInternalEnergy };

class InitialConditions_gadget : public InitialConditions
{
    using Policy = HyperbolicPolicy_Hydro;
    using PrimState = Policy::PrimState;
    using ConsState = Policy::ConsState;

    ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;
    Timers& timers;
    typename Policy::Params policy_params;
    std::string filename;
    std::unique_ptr<RefineCondition> refine_condition; 
    std::unique_ptr<ParticleUpdate> particle_projection;

    struct Data{
        ForeachCell& foreach_cell;
        int level_min, level_max;
    } data;
        
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
    particle_projection(
        ParticleUpdateFactory::make_instance(
            configMap.getValue<std::string>("gadget", "projection_kernel", "ParticleUpdate_CIC_density"),
            configMap,
            foreach_cell,
            timers)
    ),
    data({
      foreach_cell,
      foreach_cell.get_amr_mesh().get_level_min(),
      foreach_cell.get_amr_mesh().get_level_max()
    })
  {}

  void init ( UserData& U ) {
    auto hdf5_reader = HDF5ViewReader(filename);

    GadgetHeader header = GadgetHeader(hdf5_reader);

    auto read_fields = [&](const size_t itype, const std::string& gadget_part_name, const std::string& part_name) -> void {
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
        using Darr2D = Kokkos::View<double**, Kokkos::LayoutRight>;

        // Those always exist
        const auto& xp = hdf5_reader.read_dataset<Darr2D>(gadget_part_name + "/Coordinates");
        const auto& vp = hdf5_reader.read_dataset<Darr2D>(gadget_part_name + "/Velocities");
        const auto& idp = hdf5_reader.read_dataset<Kokkos::View<uint32_t*> >(gadget_part_name + "/ParticleIDs");

        DYABLO_ASSERT_HOST_RELEASE(
            (xp.extent(0) == Npart) && (vp.extent(0) == Npart) && (idp.extent(0) == Npart),
            "Inconsistent number of particles in gadget file for PartType" << itype );

        // Copy data to particle array
        auto P = U.getParticleArray( part_name );
        auto Pout = U.getParticleAccessor(
            part_name,
            { {"vx", IVX}, 
              {"vy", IVY},
              {"vz", IVZ},
              {"mass", IMASS},
              {"id", IID} } );

        real_t boxsize = header.BoxSize;
        Kokkos::parallel_for("InitGadgetParticles", Npart, KOKKOS_LAMBDA(const int i) {
            P.pos( i, IX ) = xp(i, 0) / boxsize;
            P.pos( i, IY ) = xp(i, 1) / boxsize;
            P.pos( i, IZ ) = xp(i, 2) / boxsize;

            Pout.at(i, IVX) = vp(i, 0);
            Pout.at(i, IVY) = vp(i, 1);
            Pout.at(i, IVZ) = vp(i, 2);
            Pout.at(i, IID) = idp(i);
            // Note on mass: it will be overwritten later if Masses dataset exists
            Pout.at(i, IMASS) = header.MassTable[itype];
        });

        auto read_optional = [&]( const std::string& gadget_attr_name, const std::string& attr_name ) {
            if ( !hdf5_reader.has_dataset(gadget_part_name + "/" + gadget_attr_name) )
                return;
            std::cout << "    " << gadget_attr_name << " → " << attr_name << std::endl;

            // Create associated particle attribute
            if (!U.has_ParticleAttribute(part_name, attr_name))
                U.new_ParticleAttribute(part_name, attr_name);
        
            // Read data
            auto data = hdf5_reader.read_dataset<Kokkos::View<double*> >(gadget_part_name + "/" + gadget_attr_name);
            DYABLO_ASSERT_HOST_RELEASE( data.extent(0) == Npart, "Inconsistent number of particles in gadget file for PartType" << itype << " " << gadget_attr_name );

            // Copy data to particle array
            auto Pout = U.getParticleAccessor(part_name, { {attr_name, 0} } );

            Kokkos::parallel_for("InitGadgetParticles_optional", Npart, KOKKOS_LAMBDA(const int i) {
                Pout.at_ivar(i, 0) = data(i);
            });
        };

        read_optional("Masses", "mass");
        read_optional("Density", "rho");
        // read_optional("Metallicity", "metallicity"); <-- this is a 2D array
        read_optional("SmoothingLength", "hsml");
        read_optional("InternalEnergy", "u");

    };
    
    read_fields(0, "PartType0", "gas");
    read_fields(1, "PartType1", "dark_matter");
    read_fields(2, "PartType2", "disk");
    read_fields(3, "PartType3", "bulge");
    read_fields(4, "PartType4", "star");
    read_fields(5, "PartType5", "sink");

    std::cout << "Merge all star particles into single array" << std::endl;
    if (U.has_ParticleArray("star")) {
        if (U.has_ParticleArray("bulge")) {
            U.merge_particles_if("star", "bulge", "mass");
        }
        if (U.has_ParticleArray("disk")) {
            U.merge_particles_if("star", "disk", "mass");
        }
    }

    // ----------------------------------
    //  Create the AMR structure
    // ----------------------------------    
    std::cout << "  Create AMR structure" << std::endl;

    auto timer = timers.get("InitialConditions_gadget::init");
    timer.start();
    ForeachCell& foreach_cell = data.foreach_cell;
    AMRmesh& pmesh = foreach_cell.get_amr_mesh();

    int level_min = data.level_min;
    int level_max = data.level_max;

    ScalarSimulationData scalar_data;

    U.new_fields( { "rho", "rho_vx", "rho_vy", "rho_vz", "rho_gas" });
    auto P = U.getParticleArray("gas");
    auto Pdata = U.getParticleAccessor("gas", { {"mass", IMASS}, {"rho", IRho}, {"vx", IVX}, {"vy", IVY}, {"vz", IVZ} } );

    for (uint8_t level = level_min; level < level_max; ++level) {
        std::cout << "    Level " << (int)level << std::endl;
        U.backup_and_realloc();

        ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

        const UserData::FieldAccessor Uout = U.getAccessor( {
            {"rho", 0},
            {"rho_vx", 1},
            {"rho_vy", 2},
            {"rho_vz", 3},
        });

        // Reset density
        foreach_cell.foreach_cell("InitGadget::zero_rho", Uout.getShape(),
            KOKKOS_LAMBDA( const ForeachCell::CellIndex &iCell )
        {
            Uout.at(iCell, 0) = 0.0;
        });

        // Straight injection of particles
        // NOTE: this should be an SPH-like projection
        foreach_particle.foreach_particle("Project_gas", Pdata.getShape(),
            KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex &iPart )
        {
            real_t part_mass = Pdata.at(iPart, 0);

            ForeachCell::CellIndex iCell = cellmetadata.getCellFromPos( {
                P.pos(iPart, IX), P.pos(iPart, IY), P.pos(iPart, IZ)
            } );

            real_t vx = Pdata.at(iPart, IVX);
            real_t vy = Pdata.at(iPart, IVY);
            real_t vz = Pdata.at(iPart, IVZ);

            // Compute cell size
            auto cell_size = cellmetadata.getCellSize( iCell );
            real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];

            // Project density to grid
            real_t rho_contrib = part_mass / Vcell;

            Kokkos::atomic_add( &Uout.at(iCell, 0), rho_contrib );
            Kokkos::atomic_add( &Uout.at(iCell, 1), rho_contrib * vx);
            Kokkos::atomic_add( &Uout.at(iCell, 2), rho_contrib * vy);
            Kokkos::atomic_add( &Uout.at(iCell, 3), rho_contrib * vz);
        });

        int ghost_count = std::min( {U.getShape().bx, U.getShape().by, (uint32_t)4} );
        GhostCommunicator ghost_comm(pmesh, U.getShape(), ghost_count);
        ghost_comm.exchange_ghosts( Uout );

        // Refine: this takes care to call `particle_update_density`, which 
        // exchanges ghost zones for rho_g
        refine_condition->mark_cells(U, scalar_data);

        // Refine the mesh according to markers
        pmesh.adapt();

        // Load balance at each level to avoid excessive inbalance
        pmesh.loadBalance();

        std::cout << "Noct = " << pmesh.getNumOctants() << std::endl;
    }
    pmesh.loadBalance();

    // We are done with the AMR structure, now fill the density
    U.new_fields( { "rho_vx", "rho_vy", "rho_vz", "e_tot" } );

    // Remove "gas" particles
    U.delete_ParticleArray("gas");

    timer.stop();
  }

};

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory,
                 dyablo::InitialConditions_gadget, 
                 "gadget");