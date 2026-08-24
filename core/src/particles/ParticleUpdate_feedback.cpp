#include "ParticleUpdate_base.h"
#include "utils/units/Units.h"
#include "ForeachParticle.h"
#include "states/State_hydro.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace timedelay_SNII {

  // Raiteri et al. (1996)
  KOKKOS_INLINE_FUNCTION
  real_t raiteri_ms_mass( real_t t, real_t Z ) {
    
    // clip metallicity
    real_t Z_eff;
    if (Z < 7e-5)       Z_eff = 7e-5;
    else if (Z > 3e-2)  Z_eff = 3e-2;
    else                Z_eff = Z;

    real_t a0 =  10.13 + 0.07547 * std::log10(Z_eff) - 0.008084 * std::pow(std::log10(Z_eff), 2);
    real_t a1 = -4.424 - 0.7939  * std::log10(Z_eff) - 0.1187   * std::pow(std::log10(Z_eff), 2);
    real_t a2 =  1.262 + 0.3385  * std::log10(Z_eff) + 0.05417  * std::pow(std::log10(Z_eff), 2);

    real_t c = a0 - std::log10(t);
    real_t b = a1;
    real_t a = a2;

    real_t discriminant = b*b - 4*a*c;

    real_t mass;
    if (discriminant > 0) {
      mass = (-b - std::sqrt(discriminant)) / (2*a);
      mass = std::pow(10, mass);
    }
    else {
      mass = 120.0;
    }
    return mass;
  }

  // Kroupa (2001), 0.1-100 Msun
  KOKKOS_INLINE_FUNCTION
  real_t kroupa_imf( real_t m1, real_t m2) {
    real_t A = 0.2244557;
    real_t ind = -2.3;
    return (-A/(ind + 1)) * ( std::pow(m2, ind+1) - std::pow(m1, ind+1) );
  }
}



namespace dyablo {

class ParticleUpdate_feedback : public ParticleUpdate {
public:
  using pos_t = Kokkos::Array<real_t, 3>;

  ParticleUpdate_feedback(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers)
  : foreach_cell    ( foreach_cell ),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    timers          ( timers ),
    E_SNII_physical ( configMap.getValue_in_code_unit<Units::Energy>("star_feedback", "E_SNII", "1e51 erg") ),
    cosmology       ( configMap.getValue<bool>("cosmology", "active", false) ),
    star_family     ( configMap.getValue<std::string>("star_feedback", "star_family", "stars") ),
    n_passive_scalars( configMap.getValue<int>("passive_scalars", "n_passive_scalars",
        configMap.getValue<std::vector<std::string>>("passive_scalars", "passive_scalars_names", {}).size()) ),
    feedback_radius ( configMap.getValue<int>("star_feedback", "feedback_radius", 0) )
  {
    // The injection stencil can't be larger than the AMR block size
    const Kokkos::Array<uint32_t, 3> block_size = foreach_cell.blockSize();
    uint32_t block_size_min = std::min( block_size[IX], block_size[IY] );
    if( foreach_cell.getDim() == 3 )
      block_size_min = std::min( block_size_min, block_size[IZ] );

    DYABLO_ASSERT_HOST_RELEASE( feedback_radius >= 0
                             && (uint32_t)feedback_radius <= block_size_min,
      "star_feedback/feedback_radius (" << feedback_radius << ") must be between 0 and "
      "the AMR block size (" << block_size_min << ")" );
  }

  ~ParticleUpdate_feedback() {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {

    const real_t t = cosmology ? scalar_data.get<real_t>("time_physical") : scalar_data.get<real_t>("time");
    const real_t dt = scalar_data.get<real_t>("dt");

    const real_t t_phys_yr = (t * Units::code_units().getUnit<Units::Time>()).convert_to(Units::yr());

    enum VarIndex {
      IRho, IE_tot, IE_int, IRho_vx, IRho_vy, IRho_vz, IRho_Z, IRho_added,
    };
    enum VarIndex_particle {
      IMASS, IBIRTHMASS, IVX, IVY, IVZ, IBIRTH, IMETAL
    };

    timers.get("ParticleUpdate_feedback").start();

    if( !U.has_ParticleArray( star_family ) )
    {
      timers.get("ParticleUpdate_feedback").stop();
      return;
    }

    // Scratch field to replenish the passive scalars by accumulating the gas mass deposited
    // into each cell by the supernovae of this step.
    U.new_fields( {"feedback_rho_added"} );

    // See if dual energy is enabled
    const bool has_e_int = U.has_field( "e_int" );

    std::vector<UserData::FieldAccessor_FieldInfo>
      Uin_infos = {{"rho", IRho},    {"e_tot", IE_tot},    {"rho_vx", IRho_vx},    {"rho_vy", IRho_vy},    {"rho_vz", IRho_vz},    {"feedback_rho_added", IRho_added}};
    if( has_e_int )
      Uin_infos.push_back( {"e_int", IE_int} );
    std::vector<UserData::ParticleAccessor_AttributeInfo>
      pinfos = {{"mass", IMASS}, {"birth_mass", IBIRTHMASS}, {"vx", IVX}, {"vy", IVY}, {"vz", IVZ}, {"birth_time", IBIRTH}, {"metallicity", IMETAL}};

    std::vector<UserData::FieldAccessor_FieldInfo> passive_fields;
    for (int i = 0; i < n_passive_scalars; ++i)
      passive_fields.push_back( {"rho_scalar_" + std::to_string(i), i} );
    UserData::FieldAccessor Uin_passive;
    if (!passive_fields.empty())
      Uin_passive = U.getAccessor( passive_fields );

    // Get accessors
    auto Ppos = U.getParticleArray( star_family );
    auto Pdata = U.getParticleAccessor( star_family, pinfos );
    auto Uin = U.getAccessor( Uin_infos );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t aexp = scalar_data.get<real_t>("aexp");

    const auto code_time = Units::code_units().getUnit<Units::Time>();
    const auto code_mass = Units::code_units().getUnit<Units::Mass>();

    real_t dt_phys_yr = Units::supercomoving_to_physical<Units::Time>(
      (dt * code_time).convert_to(Units::yr()),
      aexp
    );

    // Gather SN feedback parameters
    const real_t E_SNII = Units::physical_to_supercomoving<Units::Energy>(E_SNII_physical, aexp);

    Kokkos::View<int> N_supernovae_view("N_supernovae");
    Kokkos::deep_copy(N_supernovae_view, 0);

    const int ndim = foreach_cell.getDim();
    const int R    = feedback_radius;
    const int R_k  = (ndim == 3) ? R : 0;

    // For MPI, we don't want to have the same seed. TODO: Generate a unique seed based on rank and/or clock time?

    // Random seed based on current time and timestep (good enough)
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345 + int(t * 1/dt));

    foreach_particle.foreach_particle( "particles_update_feedback", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      real_t part_birth_time_phys_yr = (Pdata.at(iPart, IBIRTH) * code_time).convert_to(Units::yr());
      real_t part_age_phys_yr = t_phys_yr - part_birth_time_phys_yr;
      if (part_age_phys_yr <= 0) // decide whether this is actually needed
        return;

      // get mass of stars that leave MS in current time step based on
      // Raiteri et al. 1996 and assuming a Salpeter IMF.
      // Check if M1-M2 mass range is within CCSN progenitor mass range (8-40 solar masses)
      real_t particle_metallicity = Pdata.at(iPart, IMETAL);

      // get upper and lower mass limits of stars that die in current timestep based on Raiteri et al. (1996)
      real_t M1 = timedelay_SNII::raiteri_ms_mass(part_age_phys_yr, particle_metallicity);
      real_t M2 = timedelay_SNII::raiteri_ms_mass(part_age_phys_yr + dt_phys_yr, particle_metallicity);

      // If the entire mass range is outside the SNII progenitor mass range, skip
      if (M1 < 8.0 || M2 > 40.0) return;
      M1 = std::min(M1, 40.0);
      M2 = std::max(M2, 8.0);

      real_t part_birth_mass_phys_Msun = Units::supercomoving_to_physical<Units::Mass>(
        (Pdata.at(iPart, IBIRTHMASS) * code_mass).convert_to(Units::solar_mass()),
        aexp
      );
                                                              // Note:    upper  lower
      real_t num = part_birth_mass_phys_Msun * timedelay_SNII::kroupa_imf(M1,    M2);
      real_t num_residual = num - int(num);
      num = int(num);

      // get random number between 0 and 1
      auto generator = random_pool.get_state();
      real_t rand_num = generator.frand();
      random_pool.free_state(generator);

      if (rand_num < num_residual)
        num = num + 1;

      // If we have a SNII explosion
      if ( num > 0 ) {

        real_t meanmass = Units::physical_to_supercomoving<Units::Mass>(
          (0.5 * (M1 + M2) * Units::solar_mass()).convert_to(code_mass),
          aexp
        );
        real_t Mloss = num * meanmass;

        pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
        pos_t part_vel = {Pdata.at(iPart, IVX), Pdata.at(iPart, IVY), Pdata.at(iPart, IVZ)};

        ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

        ForeachCell::SearchMode_neighbor search_neighbor(
          cells.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

        // Walk the injection sphere : every cell whose offset from the star's own
        // cell is within `R` cells. `apply(cell, w)` is called once per stencil slot
        // with the fraction `w` of a slot the cell covers. A neighbor coarser than
        // the star's cell is reached by several slots (weight 1 each), a refined one
        // splits its slot between its 2^ndim subcells : the deposit below is thus
        // uniform in density whatever the refinement inside the sphere.
        auto foreach_target_cell = [&]( auto&& apply )
        {
          for( int dk = -R_k ; dk <= R_k ; dk++ )
          for( int dj = -R   ; dj <= R   ; dj++ )
          for( int di = -R   ; di <= R   ; di++ )
          {
            if( di*di + dj*dj + dk*dk > R*R ) continue;

            ForeachCell::CellIndex iCell_n = iCell.getNeighbor(
              { (int16_t)di, (int16_t)dj, (int16_t)dk }, search_neighbor );

            // Outside the domain, or owned by another MPI rank (writing there would
            // land in a ghost block that is never sent back) : the slot is dropped
            // and its share picked up by the rest of the sphere.
            if( !iCell_n.is_valid() || iCell_n.iOct.isGhost ) continue;

            if( iCell_n.level_diff() >= 0 )
            {
              apply( iCell_n, (real_t)1 );
            }
            else
            { // Neighbor is refined : the slot is covered by its 2^ndim subcells
              const int sub_k_count = (ndim == 3) ? 2 : 1;
              const real_t w = 1.0 / ( 2 * 2 * sub_k_count );
              for( int16_t sk = 0 ; sk < sub_k_count ; sk++ )
              for( int16_t sj = 0 ; sj < 2           ; sj++ )
              for( int16_t si = 0 ; si < 2           ; si++ )
              {
                ForeachCell::CellIndex iCell_s = iCell_n.getNeighbor( {si,sj,sk}, search_neighbor );
                if( !iCell_s.is_valid() || iCell_s.iOct.isGhost ) continue;
                apply( iCell_s, w );
              }
            }
          }
        };

        // First pass : how much of the sphere is actually reachable. Normalizing by
        // this rather than by its nominal size is what keeps the injected mass and
        // energy equal to Mloss and num*E_SNII when part of the sphere was dropped.
        real_t weight_tot = 0;
        foreach_target_cell( [&]( const ForeachCell::CellIndex&, real_t w ) { weight_tot += w; } );
        if( weight_tot <= 0 ) return;

        const real_t Mloss_per_weight = Mloss / weight_tot;
        const real_t E_per_weight     = E_SNII * num / weight_tot;

        // Second pass : deposit. Each target is divided by its own volume, so the
        // deposit stays conservative across a refinement jump.
        foreach_target_cell( [&]( const ForeachCell::CellIndex& iCell_n, real_t w )
        {
          pos_t cell_size = cells.getCellSize( iCell_n );
          real_t inv_cell_volume = w / ( cell_size[IX] * cell_size[IY] * cell_size[IZ] );

          real_t rho_loss = Mloss_per_weight * inv_cell_volume;

          real_t ethermal = E_per_weight * inv_cell_volume;
          real_t ekin = 0.5 * rho_loss * (
            SQR(part_vel[IX]) + SQR(part_vel[IY]) + SQR(part_vel[IZ])
          );

          // Atomic are mandatory since multiple particles can explode in the same cell
          Kokkos::atomic_add(&Uin.at(iCell_n, IRho), rho_loss);
          // Track the mass added to this cell so the passive scalars can be
          // rescaled consistently once all deposits are summed (see cell pass below)
          Kokkos::atomic_add(&Uin.at(iCell_n, IRho_added), rho_loss);
          Kokkos::atomic_add(&Uin.at(iCell_n, IE_tot), ethermal + ekin);
          if (has_e_int)
            Kokkos::atomic_add(&Uin.at(iCell_n, IE_int), ethermal);
          Kokkos::atomic_add(&Uin.at(iCell_n, IRho_vx), rho_loss * part_vel[IX]);
          Kokkos::atomic_add(&Uin.at(iCell_n, IRho_vy), rho_loss * part_vel[IY]);
          Kokkos::atomic_add(&Uin.at(iCell_n, IRho_vz), rho_loss * part_vel[IZ]);
        });

        // Update particle properties
        Pdata.at(iPart, IMASS) -= Mloss;

        Kokkos::atomic_add(&N_supernovae_view(), int(num));
      }
    });

    // Inject the returned supernova mass back into the passive scalars.
    // Lacking per-element SNII yields, the ejecta is assumed to carry the ambient
    // cell composition, so every per-volume passive scalar - element number densities and ion
    // abundances - scales with the gas density: field_new = field_old * rho_new / rho_old.
    if (n_passive_scalars > 0) {
      foreach_cell.foreach_cell( "feedback_scale_passive_scalars", U.getShape(),
        CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
      {
        const real_t rho_added = Uin.at(iCell, IRho_added);
        if (rho_added > 0) {
          const real_t rho_new = Uin.at(iCell, IRho);
          const real_t rho_old = rho_new - rho_added;
          const real_t factor  = rho_new / rho_old;
          for (int ivar = 0; ivar < Uin_passive.nbFields(); ++ivar)
            Uin_passive.at_ivar(iCell, ivar) *= factor;
        }
      });
    }

    U.delete_field( "feedback_rho_added" );

    int N_supernovae = 0;
    Kokkos::deep_copy(N_supernovae, N_supernovae_view);

    if (N_supernovae > 0)
      std::cout << "Feedback: " << N_supernovae << " supernovae exploded" << std::endl;

    timers.get("ParticleUpdate_feedback").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;

  real_t E_SNII_physical;

  bool cosmology;

  std::string star_family;
  int n_passive_scalars;

  int feedback_radius;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_feedback,
                  "ParticleUpdate_feedback")
