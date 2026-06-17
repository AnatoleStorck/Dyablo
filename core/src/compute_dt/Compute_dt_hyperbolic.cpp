#include "Compute_dt_base.h"

#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

namespace dyablo {

/**
 * @brief Timestep limiter for (magneto)hydrodynamics
 * 
 * Limits the timestep according to the CFL condition.
 * The limitation is of the form dt = C * min_h(d_h / |lambda_h|)
 * with :
 *  . C a constant factor < 1,
 *  . d_h the cell_size along direction h,
 *  . lambda_h the maximum signal speed in that direction
 **/
template <typename Policy>
class Compute_dt_hyperbolic : public Compute_dt
{
public:
  Compute_dt_hyperbolic( ConfigMap& configMap,
                    ForeachCell& foreach_cell,
                    Timers& timers )
  : foreach_cell(foreach_cell),
    policy_params(Policy::getParams(configMap))
  {
    real_t default_cfl = 0.5;
    if (configMap.hasValue("hydro", "cfl")) {
      std::cout << "WARNING : hydro/cfl is deprecated in .ini, use dt/hydro_cfl instead !" << std::endl;
      default_cfl = configMap.getValue<real_t>("hydro", "cfl");
    }
    this->cfl = configMap.getValue<real_t>("dt", "hydro_cfl", default_cfl);

    // Floors used to keep the signal speed well-defined : without them a cell with a
    // (truncation/cancellation induced) negative pressure gives cs = sqrt(<0) = NaN,
    // which FMAX silently drops from the CFL reduction -> that cell gets no timestep
    // limit lol
    this->smallr = configMap.getValue<real_t>("hydro", "smallr", 1e-10);
    this->smallp = configMap.getValue<real_t>("hydro", "smallp", 1e-10);

    // Verify dt_mhd is enabled if hydro update uses MHD
    // ( "Compute_dt_hydro" used to support MHD and may still be used in outdated .inis )
    bool has_mhd = configMap.getValue<std::string>("hydro", "update", "HydroUpdate_euler").find("MHD") != std::string::npos;
    if (has_mhd && std::is_same_v<Policy, HyperbolicPolicy_Hydro>)
    {
      std::cout << "WARNING : dt/dt_kernel is compute_dt_hydro but MHD policy in use. Use compute_dt_GLMMHD instead !" << std::endl;
      if( ! configMap.getValue<bool>("compute_dt_hydro", "skip_MHD_check", false) )
      {
        DYABLO_ASSERT_HOST_RELEASE( !(has_mhd && std::is_same_v<Policy, HyperbolicPolicy_Hydro>), "dt/dt_kernel is compute_dt_hydro but MHD policy in use. Use compute_dt_GLMMHD instead ! If you think this is an error set compute_dt_hydro/skip_MHD_check = true" );
      }
    }
  }

  void compute_dt( UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t dt_local;
    dt_local = compute_dt_aux(U, scalar_data);
    
    DYABLO_ASSERT_HOST_RELEASE(dt_local>0, "invalid dt = " << dt_local);

    real_t dt;
    auto communicator = foreach_cell.get_amr_mesh().getMpiComm();
    communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);

    scalar_data.set<real_t>("dt", dt);
  }

  double compute_dt_aux( UserData& U, ScalarSimulationData& scalar_data )
  {
    using PrimState = typename Policy::PrimState;
    using ConsState = typename Policy::ConsState;

    Policy policy{ policy_params, scalar_data };

    int ndim = foreach_cell.getDim();
    real_t gamma0 = policy_params.policy_params.gamma0;
    const real_t smallr = this->smallr;
    const real_t smallp = this->smallp;

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    UserData::FieldAccessor Uin = policy.getUin(U);

    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];
      
      ConsState uLoc = policy.getConsState(Uin, iCell);
      PrimState qLoc = policy.consToPrim(uLoc);

      const real_t rho = FMAX(qLoc.rho, smallr);
      const real_t p   = FMAX(qLoc.p,   smallp);
      const real_t cs = sqrt(p * gamma0 / rho);

      real_t vx = cs + FABS(qLoc.u);
      real_t vy = cs + FABS(qLoc.v);
      real_t vz = (ndim==2)? 0 : cs + FABS(qLoc.w);

      inv_dt_update = FMAX( inv_dt_update, vx/dx + vy/dy + vz/dz );

      // TODO : Find a BETTER way to do this !
      if constexpr (std::is_same_v<PrimState, HyperbolicPolicy_PrimGLMMHDState>) {
        const real_t Bx = qLoc.Bx;
        const real_t By = qLoc.By;
        const real_t Bz = qLoc.Bz;
        const real_t gr = cs*cs*rho;
        const real_t Bt2 [] = {By*By+Bz*Bz,
                               Bx*Bx+Bz*Bz,
                               Bx*Bx+By*By};
        const real_t B2 = Bx*Bx + By*By + Bz*Bz;
        const real_t cf1 = gr-B2;
        const real_t V [] = {qLoc.u, qLoc.v, qLoc.w};
        const real_t D [] = {dx, dy, dz};

        real_t cmax = 0.0;
        for (int i=0; i < ndim; ++i) {
          const real_t cf2 = gr + B2 + sqrt(cf1*cf1 + 4.0*gr*Bt2[i]);
          const real_t cf = sqrt(0.5 * cf2 / rho);

          cmax += (cf + Kokkos::abs(V[i])) / D[i];
        }
        inv_dt_update = FMAX(inv_dt_update, cmax);
     }

    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt = cfl / inv_dt;
    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);
    return dt;
  }

private:
  ForeachCell& foreach_cell;
  typename Policy::Params policy_params;

  real_t cfl;
  real_t smallr, smallp;
};

class Compute_dt_hydro : public Compute_dt_hyperbolic<HyperbolicPolicy_Hydro> {
public:
  using Compute_dt_hyperbolic<HyperbolicPolicy_Hydro>::Compute_dt_hyperbolic;
};

class Compute_dt_GLMMHD : public Compute_dt_hyperbolic<HyperbolicPolicy_GLMMHD> {
public:
  using Compute_dt_hyperbolic<HyperbolicPolicy_GLMMHD>::Compute_dt_hyperbolic;
};

} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, 
                  dyablo::Compute_dt_hydro, 
                  "Compute_dt_hydro" );

FACTORY_REGISTER( dyablo::Compute_dtFactory, 
                  dyablo::Compute_dt_GLMMHD, 
                  "Compute_dt_GLMMHD" );
