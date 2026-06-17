#include "SourceUpdate_base.h"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;
using complex_t = Kokkos::complex<real_t>;

namespace dyablo{

namespace {
  // Helpers
  using ComplexView = Kokkos::View<complex_t***>;
  using RealView    = Kokkos::View<real_t***>;
  using GridRange   = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
}

/**
 * @brief Source Term adding Orstein-Uhlenbeck turbulent forcing to the velocity and energy fields
 * 
 * This implementation follows what is done in the following papers : 
 *  . Brucy, N. et al.     "Inefficient star formation in high Mach number environments: II. Numerical simulations 
 *                          and comparison with analytical models", Astronomy & Astrophysics, 2024
 *  . Federrath, C. et al  "Comparing the statistics of interstellar turbulence in simulations and observations: 
 *                          Solenoidal versus compressive turbulence forcing", Astronomy & Astrophysics, 2010
 * 
 * @note Important note : As of today, complex -> real inverse transform modify the input buffer. This is apparently
 *       something known in the litterature. Hence, we use complex -> complex transform and only take the real part
 *       of the result
 */
class SourceUpdate_Turbulent_Forcing : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;
  int ndim;

  // Size of the uniform grid for the FFT
  int grid_size;
  
  // Correlation time for the OU process
  real_t T_driv;

  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;

  // Turbulence normalisation factor
  real_t turbulence_rms;

  // Solenoidal to compressive factor
  real_t chi;

  // Which term to include
  bool include_decay, include_wiener;

  // Random number generator
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool;

  // Grid sizes
  int nkx, nky, nkz;

  // Norms
  real_t turbulence_norm;

  // Force field 
  RealView    force_field[3];
  ComplexView force_field_k[3];
  ComplexView wiener_process[3];
  GridRange   range_policy;

  // Debugging tools
  bool debug_force_field;
  int ff_iteration;

public:
  SourceUpdate_Turbulent_Forcing(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    timers(timers),
    ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    grid_size(configMap.getValue<size_t>("turbulent_forcing", "grid_size", 64)),
    T_driv(configMap.getValue<real_t>("turbulent_forcing", "T_driv", 1.0)),
    xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
    ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
    zmin(configMap.getValue<real_t>("mesh", "zmin", 0.0)),
    xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
    ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
    zmax(configMap.getValue<real_t>("mesh", "zmax", 1.0)),
    turbulence_rms(configMap.getValue<real_t>("turbulent_forcing", "turbulence_rms", 500.0)),
    chi(configMap.getValue<real_t>("turbulent_forcing", "chi", 0.5)),
    include_decay(configMap.getValue<bool>("turbulent_forcing", "include_decay", true)),
    include_wiener(configMap.getValue<bool>("turbulent_forcing", "include_wiener", true)),
    random_pool(configMap.getValue<uint64_t>("turbulent_forcing", "random_seed", 12345)),
    debug_force_field(configMap.getValue<bool>("turbulent_forcing", "debug_force_field", false))
  { 
    ff_iteration = 0;
    nkx = grid_size/2+1;
    nky = grid_size;
    nkz = (ndim == 2 ? 1 : grid_size);

    for (int idim=0; idim < ndim; ++idim) {
      force_field[idim]    = RealView("force_field",       (ndim == 2 ? 1 : grid_size), grid_size, grid_size);
      force_field_k[idim]  = ComplexView("force_field_k",  (ndim == 2 ? 1 : nkz),       nky,       nkx);
      wiener_process[idim] = ComplexView("wiener_process", (ndim == 2 ? 1 : nkz),       nky,       nkx);
    }
    range_policy = GridRange({0,0,0},
                             {nkx, nky, ndim == 2 ? 1 : nkz});

    init_forcing();
  }

  void init_forcing() {
    auto ndim           = this->ndim;
    auto force_field    = this->force_field;
    auto force_field_k  = this->force_field_k;
    auto wiener_process = this->wiener_process;

    auto grid_size = this->grid_size;

    auto nkx = this->nkx;
    auto nky = this->nky;
    auto nkz = this->nkz;

    // Initializing power spectrum and computing total power
    Kokkos::parallel_for("Forcing_Init", range_policy, 
      KOKKOS_LAMBDA(const int i, const int j, const int k) {

        // Setting stuff to 0 first
        for (int idim=0; idim < ndim; ++idim) {
          force_field_k[idim](k, j, i) = 0.0;
          wiener_process[idim]( k, j, i) = 0.0;
          if (i < nkx && j < nky && k < nkz)
            force_field[idim](k, j, i) = 0.0;
        }

        const int ki = (i <= grid_size / 2 ? i : grid_size - i);
        const int kj = (j <= grid_size / 2 ? j : grid_size - j);
        const int kk = (k <= grid_size / 2 ? k : grid_size - k);

        const real_t kmag = Kokkos::sqrt(ki*ki + kj*kj + kk*kk);
        if (kmag >= 1.0 and kmag < 3.0) {
          for (int idim=0; idim < ndim; ++idim)
            force_field_k[idim](k, j, i) = (1.0 - (kmag-2.0)*(kmag-2.0));
        }
      });
    
    // Normalization
    if (ndim==2)
      ifft<2>();
    else 
      ifft<3>();

    // Computing power
    real_t P2 = 0.0;
    Kokkos::MDRangePolicy range_real({0, 0, 0, 0}, {grid_size, grid_size, (ndim == 2 ? 1 : grid_size), ndim});
    Kokkos::parallel_reduce("Computing P", range_real,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int idim, real_t &P2tmp) {
        const real_t Pval = force_field[idim](k, j, i);
        P2tmp += Pval*Pval;
      }, Kokkos::Sum<real_t>(P2));

    const int N = ndim * grid_size * grid_size * (ndim == 2 ? 1 : grid_size);

    // Computing the normalization factors for the turbulence 
    const real_t P_norm = Kokkos::sqrt(P2/N);
    const real_t proj_norm = (ndim == 2 ? 
                              1.0 : 
                              (0.797*chi*chi) - (0.529*chi) + 0.568); // Fitted empirically ? Where does that come from
    const real_t OU_norm = Kokkos::sqrt(T_driv / 2.0);
    turbulence_norm = 1.0 / (P_norm * proj_norm * OU_norm);
  }

  template< int ndim >
  inline void ifft() {
    auto exec_space = execution_space();

    /**
     * Copying input force-field to avoid modification
     * This is because input is not guaranteed to be const
     * when applying complex to real transformations.
     * See this issue : https://github.com/kokkos/kokkos-fft/issues/148
     **/

    ComplexView copy_ff_k("copy_for_FFT", nkz, nky, nkx);
    if (ndim == 2) {
      for (int idim=0; idim < ndim; ++idim) {
        Kokkos::deep_copy(copy_ff_k, force_field_k[idim]);
        KokkosFFT::irfft2(exec_space, copy_ff_k, force_field[idim]);
      }
    }
    else {
      for (int idim=0; idim < ndim; ++idim) {
        Kokkos::deep_copy(copy_ff_k, force_field_k[idim]);
        KokkosFFT::irfftn(exec_space, copy_ff_k, force_field[idim], {-3, -2, -1});
      }
    }
  }

  void update_wiener_process( const real_t dt ) {

    auto& wiener_process = this->wiener_process;
    auto& random_pool = this->random_pool;
    auto ndim = this->ndim;

    Kokkos::parallel_for("update_wiener_process", range_policy, KOKKOS_LAMBDA
        (const int i, const int j, const int k) {
          for (int idim=0; idim < ndim; ++idim) {
            auto generator = random_pool.get_state();

            // Should we instead generate a gaussian magnitude and random uniform angle ?
            /*
            real_t z0 = generator.normal(0, Kokkos::sqrt(dt)); //dt or sqrt(dt) ???
            real_t z1 = generator.normal(0, Kokkos::sqrt(dt));
            */
            real_t mag = generator.normal(0.0, 1.0);
            real_t arg = generator.drand() * M_PI * 2.0;
            real_t z0 = mag * Kokkos::cos(arg);
            real_t z1 = mag * Kokkos::sin(arg);
            random_pool.free_state(generator);

            complex_t dw(z0, z1);

            dw *= Kokkos::sqrt(dt);

            // Eq. (5) in Federrath 2010
            wiener_process[idim](k, j, i) = dw;
            //wiener_process[idim](k, j, i) += dw;
          }
        });
  }

  template< int ndim >
  void update_aux( UserData &U, ScalarSimulationData& scalar_data)
  {
    timers.get("Turbulent forcing").start();

    real_t T_driv = this->T_driv;
    real_t dt = scalar_data.get<real_t>("dt");

    auto grid_size     = this->grid_size;
    auto force_field_k = this->force_field_k;
    auto dW            = this->wiener_process;

    auto include_decay  = this->include_decay;
    auto include_wiener = this->include_wiener;

    auto chi = this->chi;

    // Stochastic process for excitation
    update_wiener_process(dt);

    // Compute df_k = -f_k * dt / T_driv
    Kokkos::parallel_for("update_force_field", range_policy, 
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute k vector
        const real_t kx = ((real_t)((i <= grid_size / 2) ? i : i - grid_size));
        const real_t ky = ((real_t)((j <= grid_size / 2) ? j : j - grid_size));
        const real_t kz = ((real_t)((k <= grid_size / 2) ? k : k - grid_size));
        const real_t kk[3] = {kx, ky, kz};

        const real_t k_mag2 = kx*kx + ky*ky + kz*kz;
        const real_t k_mag = Kokkos::sqrt(k_mag2);

        // Eq. (2) in Brucy 2024 -> The 2pi have disappeared 
        const bool mask = (k_mag > 1.0) && (k_mag < 3.0);
        const real_t kfac = k_mag - 2;
        const real_t F0 = (1 - kfac*kfac) * mask;

        // Eq. (6) in Federrath 2010 (dotted with Wp)
        complex_t added_turbulence[3] = {0.0, 0.0, 0.0};

        if (mask) {
          for (auto idim = 0; idim < ndim; ++idim) {
            complex_t PdW = 0.0;
            for (auto idim2 = 0; idim2 < ndim; ++idim2) {
              PdW += (
                chi * (idim == idim2) + (1 - 2 * chi) * kk[idim] * kk[idim2] / k_mag2
              ) * dW[idim](k, j, i);
            }

            if (include_wiener)
              added_turbulence[idim] = F0 * PdW;
          }
        }

        // Eq. (1) in Brucy 2024
        // if (i==1 && j==1) {
        //   complex_t ff_x = force_field_k[IX](k, j, i);
        //   complex_t ff_y = force_field_k[IY](k, j, i);
        //   printf("TEST : %e+i%e(%e); %e+i%e(%e)\n", 
        //     ff_x.real(), ff_x.imag(), Kokkos::sqrt(ff_x.real()*ff_x.real() + ff_x.imag()*ff_x.imag()),
        //     ff_y.real(), ff_y.imag(), Kokkos::sqrt(ff_y.real()*ff_y.real() + ff_y.imag()*ff_y.imag()));
        // }
        for (auto idim = 0; idim < ndim; ++idim) {
          const complex_t decaying_turbulence = include_decay * (force_field_k[idim](k, j, i) * dt / T_driv);

          // if (i==1 && j==1) {
          //   complex_t ff = force_field_k[idim](k, j, i);
          //   complex_t ad = added_turbulence[idim];
          //   complex_t dc = decaying_turbulence;
          //   complex_t df = ad - dc;
          //   complex_t nf = ff + df;
          //   printf("For dir %s; initial_force_field = %e+i%e(%e); add=%e+i%e(%e); decay=%e+i%e(%e); df=%e+i%e(%e); next_force=%e+i%e(%e)\n", 
          //     (idim == IX ? "IX" : "IY"),
          //     ff.real(), ff.imag(), Kokkos::sqrt(ff.real()*ff.real() + ff.imag()*ff.imag()),
          //     ad.real(), ad.imag(), Kokkos::sqrt(ad.real()*ad.real() + ad.imag()*ad.imag()),
          //     dc.real(), dc.imag(), Kokkos::sqrt(dc.real()*dc.real() + dc.imag()*dc.imag()),
          //     df.real(), df.imag(), Kokkos::sqrt(df.real()*df.real() + df.imag()*df.imag()),
          //     nf.real(), nf.imag(), Kokkos::sqrt(nf.real()*nf.real() + nf.imag()*nf.imag()));
          // }

          force_field_k[idim](k, j, i) += added_turbulence[idim] - decaying_turbulence;
        }
     });

    // Use inverse FFT to obtain force field in real space
    ifft<ndim>();

    // Now, loop over all cells and add force to velocity field
    auto cells = foreach_cell.getCellMetaData();
    real_t xmin = this->xmin;
    real_t ymin = this->ymin;
    real_t zmin = this->zmin;
    real_t xmax = this->xmax;
    real_t ymax = this->ymax;
    real_t zmax = this->zmax;

    enum VarIndex {IE, IRHO_VX, IRHO_VY, IRHO_VZ, IRHO};
    std::vector<UserData::FieldAccessor_FieldInfo> fields_out = 
    {
      {"rho", IRHO},
      {"e_tot_next", IE},
      {"rho_vx_next", IRHO_VX},
      {"rho_vy_next", IRHO_VY}
    };

    std::vector<UserData::FieldAccessor_FieldInfo> fields_in = 
    {
      {"rho", IRHO},
      {"rho_vx", IRHO_VX},
      {"rho_vy", IRHO_VY}
    };

    if (ndim == 3) {
      fields_in.push_back({"rho_vz", IRHO_VZ});
      fields_out.push_back({"rho_vz_next", IRHO_VZ});
    }

    auto Uin  = U.getAccessor(fields_in);
    auto Uout = U.getAccessor(fields_out);
    auto force_field = this->force_field;

    const real_t norm_factor = turbulence_norm * turbulence_rms;

    foreach_cell.foreach_cell("add_turbulent_forcing", Uout.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
      auto cell_pos = cells.getCellCenter(iCell);
      // Interpolate force from 2^ndim grid points
      real_t force[3] = {0.0, 0.0, 0.0};

      // Compute normalized position in [0,1]
      const real_t x01 = (cell_pos[IX]-xmin)/(xmax-xmin);
      const real_t y01 = (cell_pos[IY]-ymin)/(ymax-ymin);
      const real_t z01 = (cell_pos[IZ]-zmin)/(zmax-zmin);
      // Compute position in grid
      const real_t xg = x01 * grid_size;
      const real_t yg = y01 * grid_size;
      const real_t zg = z01 * grid_size;

      // NGP-like scheme
      const int i0 = ((int)floor(xg)) % grid_size;
      const int j0 = ((int)floor(yg)) % grid_size;
      const int k0 = ((int)floor(zg)) % grid_size;
      const int i1 = (i0+1) % grid_size;
      const int j1 = (j0+1) % grid_size;
      const int k1 = (k0+1) % grid_size;

      const real_t xi = xg-i0;
      const real_t yi = yg-j0;
      const real_t zi = zg-k0; 

      // Bi/Tri-linear interpolation
      for (int idim=0; idim < ndim; ++idim) {
        if (ndim == 2) {
          const real_t fx0 = (1.0-xi) * force_field[idim](0, j0, i0) + xi * force_field[idim](0, j0, i1);
          const real_t fx1 = (1.0-xi) * force_field[idim](0, j1, i0) + xi * force_field[idim](0, j1, i1);
          force[idim] = (1.0-yi) * fx0 + yi * fx1;
        }
        else {
          const real_t fx00 = (1.0-xi) * force_field[idim](k0, j0, i0) + xi * force_field[idim](k0, j0, i1);
          const real_t fx01 = (1.0-xi) * force_field[idim](k0, j1, i0) + xi * force_field[idim](k0, j1, i1);
          const real_t fx10 = (1.0-xi) * force_field[idim](k1, j0, i0) + xi * force_field[idim](k1, j0, i1);
          const real_t fx11 = (1.0-xi) * force_field[idim](k1, j1, i0) + xi * force_field[idim](k1, j0, i1);
          
          const real_t fy0 = (1.0-yi) * fx00 + yi * fx01;
          const real_t fy1 = (1.0-yi) * fx10 + yi * fx11;

          force[idim] = (1.0-zi) * fy0 + zi * fy1;
        }

        // Normalizing turbulence
        force[idim] *= norm_factor;
      }
        
      // Now adding the force to the field
      const real_t rho = Uin.at(iCell, IRHO);

      // Calculating Ek before updating with turbulence
      const real_t rho_u_old = Uout.at(iCell, IRHO_VX);
      const real_t rho_v_old = Uout.at(iCell, IRHO_VY);
      const real_t rho_w_old = (ndim == 2 ? 0.0 : Uout.at(iCell, IRHO_VZ));
      const real_t rho_old   = Uout.at(iCell, IRHO); 
      const real_t Ek_old    = (rho_u_old*rho_u_old + rho_v_old*rho_v_old + rho_w_old*rho_w_old) / rho_old * 0.5;

      Uout.at(iCell, IRHO_VX) += dt * rho * force[IX];
      Uout.at(iCell, IRHO_VY) += dt * rho * force[IY];
      if (ndim == 3)
        Uout.at(iCell, IRHO_VZ) += dt * rho * force[IZ];

      const real_t rho_u_new = Uout.at(iCell, IRHO_VX);
      const real_t rho_v_new = Uout.at(iCell, IRHO_VY);
      const real_t rho_w_new = (ndim == 2 ? 0.0 : Uout.at(iCell, IRHO_VZ));
      const real_t Ek_new    = (rho_u_new*rho_u_new + rho_v_new*rho_v_new + rho_w_new*rho_w_new) / rho_old * 0.5;

      Uout.at(iCell, IE) += Ek_new - Ek_old;
    });

    if (debug_force_field) {
      auto force_field_host = Kokkos::create_mirror_view(force_field[IX]);
      Kokkos::deep_copy(force_field_host, force_field[IX]);

      // std::ofstream f_out;
      // std::ostringstream oss;
      // oss << "force_field_" << std::setw(5) << std::setfill('0') << ff_iteration << ".dat";
      // ff_iteration++;
      // f_out.open(oss.str());
      // int k = (ndim == 2 ? 0 : grid_size / 2);
      // for (int i=0; i < grid_size; ++i) {
      //   for (int j=0; j < grid_size; ++j)
      //     f_out << force_field_host(k, i, j) << " ";
      //   f_out << std::endl;
      // }
      // f_out.close();
    }

    timers.get("Turbulent forcing").stop();
  }

  void update( UserData &U, ScalarSimulationData& scalar_data) {
    if (ndim == 2) {
      update_aux<2>(U, scalar_data);
    } else if (ndim == 3) {
      update_aux<3>(U, scalar_data);
    }
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::SourceUpdate_Turbulent_Forcing,
                  "SourceUpdate_Turbulent_Forcing" );