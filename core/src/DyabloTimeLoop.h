#pragma once

#include <csignal>

#include "utils/config/ConfigMap.h"
#include "utils/monitoring/Timers.h"
#include "ScalarSimulationData.h"
#include "amr/AMRmesh.h"
#include "foreach_cell/ForeachCell.h"

#include "compute_dt/Compute_dt.h"
#include "refine_condition/RefineCondition.h"
#include "init/InitialConditions.h"
#include "io/IOManager.h"
#include "io/Output_Movie.h"
#include "gravity/GravitySolver.h"
#include "hyperbolic/HyperbolicUpdate.h"
#include "particles/ParticleUpdate.h"
#include "amr/MapUserData.h"
#include "parabolic/ParabolicUpdate.h"
#include "source_terms/SourceUpdate.h"
#include "passive_scalars/PassiveScalar_IC.h"
#include "UserData.h"
#include "Cosmo.h"
#include "mpi/GhostCommunicator.h"

namespace dyablo {


/***
 * Triggers when variable <var> has increased more than <interval> since last trigger
 * (last trigger is rounded to a multiple of <interval> to avoid drifting)
 ***/
template< typename T >
struct Interval_trigger
{
private:
  std::string var;
  bool enabled = false;
  T interval;
  T last_trigger;

public:
  Interval_trigger() = default;
  Interval_trigger(const Interval_trigger&) = default;
  Interval_trigger& operator=(const Interval_trigger&) = default;

  Interval_trigger( const std::string& var, T interval, const ScalarSimulationData& scalar_data  )
    : var(var),
      enabled( interval > 0 ),
      interval(interval),
      last_trigger( std::floor(scalar_data.get<T>(var)/interval) * interval )
  {}

  bool trigger( const ScalarSimulationData& scalar_data )
  {
    if( !enabled )
      return false;

    T val = scalar_data.get<T>(var);
    if( val - last_trigger >= interval )
    {
      last_trigger = std::floor(val/interval) * interval;
      return true;
    }
    else
      return false;
  }
};

/***
 * Contains information to determine is specific operations must be executed or not at each timestep
 * ( output, checkpoint, end, ... )
 ***/
class IterationHandler
{
public:
  Interval_trigger<int> output_frequency, checkpoint_frequency;
  std::string output_slice_var;
  Interval_trigger<real_t> output_timeslice, checkpoint_timeslice;
  Interval_trigger<int> loadbalance_frequency, amr_frequency;

  bool output_first_iter = true;

  int iter_end; // max iteration count (negative => ignored)
  std::string t_end_var; // scalar_data variable used to test for simulation end ("time" by default, but could be "aexp")
  real_t t_end; // End value for selected scalar_data variable t_end_var
  bool use_t_end; // enable/disable termination when end value is attaigned (default on t_end is positive, but can be overriden )
  real_t t_end_epsilon = 0;
  CosmoManager cosmo_manager;

  IterationHandler(ConfigMap& configMap, const ScalarSimulationData& scalar_data )
  : output_frequency     ( "iter", configMap.getValue<int>("run", "output_frequency",       -1), scalar_data ),
    checkpoint_frequency    ( "iter", configMap.getValue<int>("run", "checkpoint_frequency",   -1), scalar_data ),
    loadbalance_frequency( "iter", configMap.getValue<int>("amr", "load_balancing_frequency",1000), scalar_data ),
    amr_frequency        ( "iter", configMap.getValue<int>("amr", "cycle_frequency",         1), scalar_data),
    iter_end             ( configMap.getValue<int>("run", "nstepmax",                1000) ),
    t_end_var            ( configMap.getValue<std::string>("run", "t_end_var", "time") ),
    t_end                ( configMap.getValue<real_t>("run", "tEnd", 0.0) ),
    use_t_end            ( configMap.getValue<bool>("run", "use_tEnd", t_end > 0) ),
    cosmo_manager        ( configMap )
  {
    // Translate output/checkpoint_expslice into
    if( configMap.hasValue("run", "output_expslice") || configMap.hasValue("run", "checkpoint_expslice")  )
    {
      real_t output_expslice = configMap.getValue<real_t>("run", "output_expslice", -1);
      real_t checkpoint_expslice = configMap.getValue<real_t>("run", "checkpoint_expslice", -1);
      if( output_expslice > 0 || checkpoint_expslice > 0)
      {
        std::string slice_var = configMap.getValue<std::string>("run", "output_slice_var", "aexp");
        if( slice_var != "aexp" )
          std::cout << "WARNING : run/checkpoint_expslice is set but run/output_slice_var != aexp. output/checkpoint_expslice ignored" << std::endl;
        else
        {
          real_t output_timeslice = configMap.getValue<real_t>("run", "output_timeslice", output_expslice);
          if( output_timeslice != output_expslice )
            std::cout << "WARNING : output_expslice ("<<output_expslice<<") is set but run/output_timeslice was already set to a different value. Using run/output_timeslice = " << output_timeslice << std::endl;

          real_t checkpoint_timeslice = configMap.getValue<real_t>("run", "checkpoint_timeslice", checkpoint_expslice);
          if( checkpoint_timeslice != checkpoint_expslice )
            std::cout << "WARNING : checkpoint_expslice ("<<checkpoint_expslice<<") is set but run/checkpoint_timeslice was already set to a different value. Using run/checkpoint_timeslice = " << checkpoint_timeslice << std::endl;
        }
      }
    }
    this->output_slice_var = configMap.getValue<std::string>("run", "output_slice_var", "time");
    this->output_timeslice     = Interval_trigger(output_slice_var, configMap.getValue<real_t>("run", "output_timeslice", -1), scalar_data);
    this->checkpoint_timeslice = Interval_trigger(output_slice_var, configMap.getValue<real_t>("run", "checkpoint_timeslice", -1), scalar_data);

    if( t_end_var == "aexp" )
    {
      this->t_end_var = "time";
      real_t aexp_end = this->t_end;
      this->t_end = cosmo_manager.expansionToTime( aexp_end );
    }
    else if( t_end_var != "time" )
      std::cout << "WARNING : can't correct dt to match t_end, possible overshoot. var=" << t_end_var << std::endl;

  }

  /// return true if end of simulation
  bool stop_criterion( const ScalarSimulationData& scalar_data )
  {
    int iter = scalar_data.get<int>("iter");
    real_t time = scalar_data.get<real_t>(t_end_var); // time may not be time, can be some other variable user for end

    bool stop_iter = iter_end > 0 && iter >= iter_end;
    bool stop_time = use_t_end    && time >= t_end - t_end_epsilon;

    return stop_iter || stop_time;
  }

  /// return true if snapshot output should be executed at this timestep
  bool output_trigger( const ScalarSimulationData& scalar_data )
  {
    // Always output first iteration
    bool first_iter = output_first_iter;
    output_first_iter = false;
    // Output at fixed frequency set by 'm_output_frequency'
    bool frequency_trigger = output_frequency.trigger( scalar_data );
    bool timeslice_trigger = output_timeslice.trigger( scalar_data );

    return first_iter || frequency_trigger || timeslice_trigger;
  }

  /// return true if checkpoint output should be executed at this timestep
  bool checkpoint_trigger( const ScalarSimulationData& scalar_data )
  {
    // Output at fixed frequency set by 'm_output_frequency'
    bool frequency_trigger = checkpoint_frequency.trigger( scalar_data );
    bool timeslice_trigger = checkpoint_timeslice.trigger( scalar_data );

    return frequency_trigger || timeslice_trigger;
  }

  /// return true if amr cycle should be executed at this timestep
  bool amr_cycle_trigger( const ScalarSimulationData& scalar_data )
  {
    return amr_frequency.trigger(scalar_data);
  }

  /// return true if loadbalancing should be executed at this timestep
  bool loadbalance_trigger( const ScalarSimulationData& scalar_data )
  {
    return loadbalance_frequency.trigger(scalar_data);
  }

  /// Correct dt value to avoid overshooting t_end
  void correct_dt_end( ScalarSimulationData& scalar_data )
  {
    if( use_t_end )
    {
      if( t_end_var == "time" ) // Keep check in constructor updated when adding cases
      {
        real_t time = scalar_data.get<real_t>("time");
        if (use_t_end && time + scalar_data.get<real_t>("dt") > t_end)
            scalar_data.get<real_t>("dt") = t_end - time;
      }
    }
  }

  /// Apply defore going to next iteration
  void next_iter( ScalarSimulationData& scalar_data )
  {
    scalar_data.get<int>("iter")++;
    scalar_data.get<real_t>("time")+=scalar_data.get<real_t>("dt");
    if( cosmo_manager.cosmo_run )
    {
      scalar_data.get<real_t>("aexp") = cosmo_manager.timeToExpansion(scalar_data.get<real_t>("time"));
      scalar_data.set<real_t>("time_physical", cosmo_manager.compute_physical_t(scalar_data.get<real_t>("aexp")));
    }
  }
};

/**
 * Main class to run Dyablo simulation
 **/
class DyabloTimeLoop{
private:
  std::shared_ptr<AMRmesh> init_amr_mesh( ConfigMap& configMap )
  {
    AMRmesh::Parameters p = AMRmesh::parse_parameters(configMap);
    return std::make_shared<AMRmesh>( p.dim, p.periodic, p.level_min, p.level_max,
                                      p.coarse_grid_size );
  }
public:
  /**
   * Create ans initialize a simulation
   **/
  DyabloTimeLoop( ConfigMap& configMap )
  : m_communicator( GlobalMpiSession::get_comm_world() ),
    m_amr_mesh( init_amr_mesh( configMap ) ),
    m_foreach_cell( *m_amr_mesh, configMap ),
    U(configMap, m_foreach_cell)
  {
    timers.get("Init").start();

    std::vector<std::string> initial_conditions_ids = configMap.getValue<std::vector<std::string>>("run", "initial_conditions");
    bool has_restart_ic =
      initial_conditions_ids.end() != std::find(initial_conditions_ids.begin(), initial_conditions_ids.end(), "restart") ||
      initial_conditions_ids.end() != std::find(initial_conditions_ids.begin(), initial_conditions_ids.end(), "tiled_restart");

    // Passive scalars
    // Needs to be first so that n_passive_scalars is defined for the policies
    passive_scalars_names = configMap.getValue<std::vector<std::string>>("run", "passive_scalars_names", {});
    n_passive_scalars = configMap.getValue<int>("run", "n_passive_scalars", passive_scalars_names.size());
    if( !has_restart_ic && n_passive_scalars > 0 )
    {
      std::set<std::string> field_names;
      for( const auto& name : passive_scalars_names )
        field_names.insert(name);
      U.new_fields(field_names);
    }

    // .ini : report legacy hydro/problem to run/initial_conditions
    if( configMap.hasValue("hydro", "problem") )
    {
      std::string hydro_problem = configMap.getValue<std::string>("hydro", "problem", "undefined");
      std::cout << "WARNING : hydro/problem is deprecated in .ini, use run/initial_conditions instead" << std::endl;
      configMap.getValue<std::vector<std::string>>("run", "initial_conditions", {hydro_problem});
    }


    timers.get("initial_conditions").start();
    // Initialize cells
    {
      // Handle legacy run/restart_enabled option
      if( configMap.hasValue("run","restart_enabled") )
      {
        std::cout << "WARNING : run/restart_enabled is deprecated, use run/initial_conditions=restart,... instead" << std::endl;
        DYABLO_ASSERT_HOST_RELEASE( !configMap.getValue<bool>("run","restart_enabled") || initial_conditions_ids.end() != std::find( initial_conditions_ids.begin(), initial_conditions_ids.end(), "restart" ),
                                    "run/restart_enabled=ON but 'restart' is not in run/initial_conditions list" );
      }

      for( std::string init_name : initial_conditions_ids )
      {
        std::unique_ptr<InitialConditions> initial_conditions =
          InitialConditionsFactory::make_instance(init_name,
            configMap,
            m_foreach_cell,
            timers);
        initial_conditions->init( U );
      }
    }

    if( !U.has_field("rho") )
      U.new_fields({"rho"});
    timers.get("initial_conditions").stop();

    configMap.getValue<int>("mesh", "ndim", 3);

    this->m_nlog = configMap.getValue<int>("run", "nlog", 10);
    this->m_enable_output = configMap.getValue<bool>("run", "enable_output", true);
    this->m_enable_checkpoint = configMap.getValue<bool>("run", "enable_checkpoint", true);
    this->m_iter_start = configMap.getValue<int>("run", "iter_start", 0);

    this->cosmo_manager = std::make_unique<CosmoManager>( configMap );
    real_t t0_default = 0;
    if( cosmo_manager->cosmo_run )
    {
      t0_default = cosmo_manager->expansionToTime( cosmo_manager->a_start );
      m_scalar_data.set("aexp", cosmo_manager->a_start);
      m_scalar_data.set<real_t>("time_physical", cosmo_manager->compute_physical_t(m_scalar_data.get<real_t>("aexp")));
    }
    else
    {
      m_scalar_data.set("aexp", 1.0);
      real_t astart = configMap.getValue<real_t>("cosmology",  "aStart",  1.0);
      DYABLO_ASSERT_HOST_RELEASE( astart == 1.0, "cosmology is disabled but cosmology/astart = " << astart << " != 1" );
    }

    this->m_scalar_data.set("iter", m_iter_start);
    this->m_scalar_data.set<int>("n_groups", configMap.getValue<int>("rad", "n_groups", 4));
    {
      real_t t0 = configMap.getValue<real_t>("run", "tStart", t0_default);
      this->m_scalar_data.set("time", t0);
    }

    this->m_loadbalance_coherent_levels = configMap.getValue<int>("amr", "loadbalance_coherent_levels", 3);

    std::string gravity_solver_id = configMap.getValue<std::string>("gravity", "solver", "none");
    this->gravity_solver = GravitySolverFactory::make_instance( gravity_solver_id,
        configMap,
        m_foreach_cell,
        timers
      );
    if( this->gravity_solver )
      m_gravity_type = configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE);
    else
      m_gravity_type = GRAVITY_NONE;

    std::string gravity_update_id = configMap.getValue<std::string>("gravity", "hydro_update", m_gravity_type==GRAVITY_NONE?"none":"HydroUpdate_gravity");
    this->gravity_update = HyperbolicUpdateFactory::make_instance( gravity_update_id,
        configMap,
        m_foreach_cell,
        timers
      );

    this->m_iteration_handler = std::make_unique<IterationHandler>(configMap, m_scalar_data);

    std::string godunov_updater_id = configMap.getValue<std::string>("hydro", "update", "HydroUpdate_hancock");
    this->has_mhd = godunov_updater_id.find("MHD") != std::string::npos;
    this->is_glm  = godunov_updater_id.find("GLM") != std::string::npos;


    {
      int hydro_ghost_count;
      if( godunov_updater_id.find("oneneighbor") != std::string::npos )
        hydro_ghost_count = 2; // Could be 1 but other kernels may need 2
      else if( godunov_updater_id.find("hancock") != std::string::npos )
        hydro_ghost_count = 4;
      else
        hydro_ghost_count = 2;

      this->ghost_count = std::min( {U.getShape().bx, U.getShape().by, (uint32_t)hydro_ghost_count} );
    }



    this->godunov_updater = HyperbolicUpdateFactory::make_instance( godunov_updater_id,
      configMap,
      this->m_foreach_cell,
      timers
    );

    std::string particle_position_updater_id = configMap.getValue<std::string>("particles", "update_position", "none");
    this->particle_position_updater = ParticleUpdateFactory::make_instance( particle_position_updater_id,
      configMap,
      this->m_foreach_cell,
      timers
    );

    std::string particle_update_density_id = configMap.getValue<std::string>("particles", "update_density", "none");
    this->particle_update_density = ParticleUpdateFactory::make_instance( particle_update_density_id,
      configMap,
      this->m_foreach_cell,
      timers
    );

    std::string particle_spawn_id = configMap.getValue<std::string>("particles", "spawn", "none");
    this->particle_spawn = ParticleUpdateFactory::make_instance( particle_spawn_id,
      configMap,
      this->m_foreach_cell,
      timers
    );

    std::vector<std::string> particle_update_source_term_ids = configMap.getValue<std::vector<std::string>>("particles", "source_term", {});
    for (auto particle_update_source_term_id: particle_update_source_term_ids) {
      this->particle_source_updaters.push_back(ParticleUpdateFactory::make_instance( particle_update_source_term_id,
        configMap,
        m_foreach_cell,
        timers
      ));
    }

    std::string mapUserData_id = configMap.getValue<std::string>("amr", "remap", "MapUserData_mean");
    this->mapUserData = MapUserDataFactory::make_instance( mapUserData_id,
      configMap,
      this->m_foreach_cell,
      timers
    );

    std::string iomanager_id = configMap.getValue<std::string>("output", "backend", "IOManager_hdf5");
    if( m_enable_output )
    {
      this->io_manager = IOManagerFactory::make_instance( iomanager_id,
        configMap,
        m_foreach_cell,
        timers
      );
    }

    std::string iomanager_checkpoint_id = configMap.getValue<std::string>("output", "checkpoint", "IOManager_checkpoint");
    if( m_enable_checkpoint )
    {
      this->io_manager_checkpoint = IOManagerFactory::make_instance( iomanager_checkpoint_id,
        configMap,
        m_foreach_cell,
        timers
      );
    }

    this->output_movie = std::make_unique<Output_Movie>(configMap, m_foreach_cell);
    if( !this->output_movie->is_enabled() )
      this->output_movie.reset();

    std::vector<std::string> compute_dt_ids = configMap.getValue<std::vector<std::string>>("dt", "dt_kernel", {"Compute_dt_hydro"});
    DYABLO_ASSERT_HOST_RELEASE(compute_dt_ids.size() > 0, "dt_kernel should not be empty !");
    for (auto compute_dt_id: compute_dt_ids) {
      this->compute_dt.push_back(Compute_dtFactory::make_instance( compute_dt_id,
        configMap,
        m_foreach_cell,
        timers
      ));
    }
    m_scalar_data.set<real_t>("dt", 1.0); // Setting initial value for dt to avoid problems with MHD kernels

    std::string refine_condition_id = configMap.getValue<std::string>("amr", "markers_kernel", "RefineCondition_second_derivative_error");
    this->refine_condition = RefineConditionFactory::make_instance( refine_condition_id,
      configMap,
      m_foreach_cell,
      timers
    );

    std::string viscosity_updater_id = configMap.getValue<std::string>("viscosity", "update", "none");
    if (viscosity_updater_id != "none") {
      this->viscosity_updater = ParabolicUpdateFactory::make_instance( viscosity_updater_id,
        configMap,
        m_foreach_cell,
        timers,
        PARABOLIC_VISCOSITY
      );
    }
    std::string tc_updater_id = configMap.getValue<std::string>("thermal_conduction", "update", "none");
    if (tc_updater_id != "none") {
      this->thermal_conduction_updater = ParabolicUpdateFactory::make_instance( tc_updater_id,
        configMap,
        m_foreach_cell,
        timers,
        PARABOLIC_THERMAL_CONDUCTION
      );
    }

    std::vector<std::string> source_updater_ids = configMap.getValue<std::vector<std::string>>("source_terms", "updates", {});
    for (auto source_updater_id: source_updater_ids) {
      this->source_updaters.push_back(SourceUpdateFactory::make_instance( source_updater_id,
        configMap,
        m_foreach_cell,
        timers
      ));
    }

    std::string rad_updater_id = configMap.getValue<std::string>("rad", "update", "none");
    if (rad_updater_id != "none") {
      this->rad_updater = HyperbolicUpdateFactory::make_instance( rad_updater_id,
        configMap,
        m_foreach_cell,
        timers);
    }
    
    timers.get("passive_scalar_init").start();
    // Get initial conditions ids
    std::vector<std::string> passive_scalars_ids = configMap.getValue<std::vector<std::string>>("run", "passive_scalars_init", {});
    // Initialize cells
    {
      int passive_id = 0;
      for( std::string init_name : passive_scalars_ids )
      {
        if (init_name == "none")
          continue;
        std::unique_ptr<PassiveScalar_IC> passive_scalar_ic =
          PassiveScalar_IC_Factory::make_instance(init_name, 
            configMap,
            m_foreach_cell,
            timers,
            passive_scalars_names[passive_id]);
        passive_scalar_ic->init( U );
        passive_id++;
      }  

      auto fields = U.getEnabledFields();
      std::cout << "Enabled fields : " << std::endl;
      for (auto f: fields)
        std::cout << " " << f << std::endl;
      
      // Once all passive scalars have been initialized we transfer the arrays to the hydro passive scalars
      for (int i=0; i < std::min(n_passive_scalars, 4); ++i) {
        std::ostringstream oss;
        oss << "rho_scalar_" << i;
        U.move_field(oss.str(), passive_scalars_names[i]);
      }
    } 
    timers.get("passive_scalar_init").stop();

    // Sanity check : No sense in doing parabolic update without hydro
    DYABLO_ASSERT_HOST_RELEASE(godunov_updater || !viscosity_updater, "Cannot have viscosity without hydro !");
    DYABLO_ASSERT_HOST_RELEASE(godunov_updater || !thermal_conduction_updater, "Cannot have thermal conduction without hydro !");

    int rank = m_communicator.MPI_Comm_rank();
    if (rank==0) {
      std::cout << "##########################" << "\n";
      std::cout << "Godunov updater    : " << godunov_updater_id << std::endl;
      std::cout << "IO Manager         : " << iomanager_id << std::endl;
      std::cout << "Gravity solver     : " << gravity_solver_id << std::endl;
      std::cout << "Initial conditions : " ;
        for( const std::string& id : initial_conditions_ids )
          std::cout << "`" << id << "` ";
      std::cout << std::endl;
      std::cout << "Refine condition   : " << refine_condition_id << std::endl;
      std::cout << "Compute dt         : ";
        for( const std::string& id : compute_dt_ids )
          std::cout << "`" << id << "` ";
        std::cout << std::endl;
      if (viscosity_updater_id != "none")
        std::cout << std::endl << "Viscosity solver : " << viscosity_updater_id << std::endl;
      if (tc_updater_id != "none")
        std::cout << "Thermal conduction solver : " << tc_updater_id << std::endl;
      std::cout << "Source Terms : ";
      for(const std::string &id : source_updater_ids )
        std::cout << "`" << id << "` ";
      std::cout << std::endl;
      std::cout << "##########################" << std::endl;
    }

    std::ofstream out_ini("last.ini" );
    configMap.output( out_ini );

    timers.get("Init").stop();
  }

  static int interrupted;
  static void interrupt_handler( int sig )
  {
    if( !interrupted )
    {
      std::cout << "Signal " << sig << " : ending simulation..." << std::endl;
      interrupted = true;
    }
    else
    {
      std::cout << "Signal " << sig << " : killing simulation!" << std::endl;
      raise(SIGABRT);
    }
  }

  void timer_file(Timers& timers)
  {
    dyablo::MpiComm mpi_comm = dyablo::GlobalMpiSession::get_comm_world();
    int tag = 10;

    std::vector<std::string> names;
    std::vector<real_t> cpu_times_local;

    timers.get_timers( names, cpu_times_local );

    if( mpi_comm.MPI_Comm_rank() != 0 )
      mpi_comm.MPI_Send(cpu_times_local.data(), cpu_times_local.size(), 0, tag);
    else
    {
      std::ofstream out("timers.txt");
      out << "Rank";
      for( const std::string& name : names )
      {
        out << " ; " << name;
      }
      out << std::endl;
      out << 0;
      for( real_t time : cpu_times_local )
      {
        out << " ; " << time;
      }
      out << std::endl;
      for( int r=1; r<mpi_comm.MPI_Comm_size(); r++ )
      {
        std::vector<real_t> cpu_times_remote(cpu_times_local.size());
        mpi_comm.MPI_Recv(cpu_times_remote.data(), cpu_times_remote.size(), r, tag);
        out << r;
        for( real_t time : cpu_times_remote )
        {
          out << " ; " << time;
        }
        out << std::endl;
      }
    }
  }

  /**
   * Run the simulation for multiple timesteps until a stop condition is met
   **/
  void run()
  {
    

    // Stop simulation when recieving SIGINT
    // NOTE : mpirun catches SIGINT and forwards SIGTERM to child process
    // NOTE : scancel sends SIGTERM by default, you can use 'scancel -s INT'
    signal( SIGINT, interrupt_handler );
    bool finished = false;
    while( !finished )
    {
      timers.get("TimeLoop").start();
      step();      
      timers.get("TimeLoop").stop();
      int any_interrupted;
      m_communicator.MPI_Allreduce(&interrupted, &any_interrupted, 1, MpiComm::MPI_Op_t::LOR);
      finished = m_iteration_handler->stop_criterion( m_scalar_data ) || any_interrupted;
    }
    signal( SIGINT, SIG_DFL );

    

    // Always output after last iteration
    timers.get("outputs").start();
    
    // Renaming passive scalars before outputting
    if (n_passive_scalars > 0) {
      for (int i=0; i < n_passive_scalars; ++i) {
        std::ostringstream oss;
        oss << "rho_scalar_" << i;
        U.move_field(passive_scalars_names[i], oss.str());
      }
    }
    if( m_enable_output )
      io_manager->save_snapshot(U, m_scalar_data);
    if( output_movie )
      output_movie->save_frame(U, m_scalar_data);
    timers.get("outputs").stop();
    timers.get("checkpoint").start();
    if( m_enable_checkpoint )
      io_manager_checkpoint->save_snapshot(U, m_scalar_data);
    timers.get("checkpoint").stop();

    int rank = m_communicator.MPI_Comm_rank();
    if ( rank == 0 )
    {
      std::cout << "Final ";
      m_scalar_data.print();

      timers.print();

      // Todo : count cell updates
      //real_t t_tot   = timers.get("total").elapsed(Timers::Timer::Elapsed_mode_t::ELAPSED_CPU);
      // printf("Perf             : %5.3f number of Mcell-updates/s\n",
      //       1.0 * m_total_num_cell_updates * (bx*by*bz) / t_tot * 1e-6);

      // printf("Total number of cell-updates : %ld\n",
      //       m_total_num_cell_updates * (bx*by*bz));
    }
    timer_file(timers);

  }

  /**
   * Run one timestep of the simulation
   **/
  void step()
  {
    // Renaming passive scalars before outputting
    if (n_passive_scalars > 0) {
      for (int i=0; i < n_passive_scalars; ++i) {
        std::ostringstream oss;
        oss << "rho_scalar_" << i;
        U.move_field(passive_scalars_names[i], oss.str());
      }
    }

    // Write output files
    {
      timers.get("outputs").start();
      bool output_written = false;
      if( m_enable_output && m_iteration_handler->output_trigger(m_scalar_data) )
      {
        int rank = m_communicator.MPI_Comm_rank();
        if( rank == 0 )
        {
          std::cout << "Output: ";
          m_scalar_data.print();
        }
        io_manager->save_snapshot(U, m_scalar_data);
        output_written = true;
      }
      if( output_movie && output_movie->should_output(m_scalar_data, output_written) )
      {
        int rank = m_communicator.MPI_Comm_rank();
        if( rank == 0 )
        {
          std::cout << "Movie: ";
          m_scalar_data.print();
        }
        output_movie->save_frame(U, m_scalar_data);
      }
      timers.get("outputs").stop();
    }

    // Write checkpoint
    {
      timers.get("checkpoint").start();
      if( m_enable_checkpoint && m_iteration_handler->checkpoint_trigger(m_scalar_data) )
      {
        int rank = m_communicator.MPI_Comm_rank();
        if( rank == 0 )
        {
          std::cout << "Checkpoint: ";
          m_scalar_data.print();
        }
        io_manager_checkpoint->save_snapshot(U, m_scalar_data);
      }
      timers.get("checkpoint").stop();
    }

    // And moving back
    if (n_passive_scalars > 0) {
      for (int i=0; i < n_passive_scalars; ++i) {
        std::ostringstream oss;
        oss << "rho_scalar_" << i;
        U.move_field(oss.str(), passive_scalars_names[i]);
      }
    }

    // Compute new dt
    {
      timers.get("dt").start();
      // Getting the smallest timestep according to all the given kernels
      real_t dt = std::numeric_limits<real_t>::max();
      for (auto &dt_kernel: compute_dt) {
        dt_kernel->compute_dt( U, m_scalar_data );
        const real_t loc_dt = m_scalar_data.get<real_t>("dt");
        dt = std::min(dt, loc_dt);
      }

      m_scalar_data.set("dt", dt);

      // correct dt if end of simulation
      m_iteration_handler->correct_dt_end(m_scalar_data);

      timers.get("dt").stop();
    }

    // Log iteration
    { // Todo make a logger
      int rank = m_communicator.MPI_Comm_rank();
      if( m_scalar_data.get<int>("iter") % m_nlog == 0 )
      {
        if( rank == 0 )
          m_scalar_data.print();

        uint32_t nbOcts = m_amr_mesh->getNumOctants();
        uint32_t nbGhosts = m_amr_mesh->getNumGhosts();
        std::cout << "Mesh - rank " << rank << " octs : " << nbOcts << " (" << nbGhosts << ")" << std::endl;
      }
    }


    GhostCommunicator ghost_comm(*m_amr_mesh, U.getShape(), ghost_count, false, m_communicator);

    auto communicate_ghosts = [&](std::vector< std::string > exchange_vars)
    {
      std::vector<UserData::FieldAccessor::FieldInfo> field_info;
      int nvars = exchange_vars.size();
      if (nvars == 0)
        return;
      for(int i=0; i<nvars; i++)
        field_info.push_back( {exchange_vars[i],i} );
      auto Uexchange = U.getAccessor(field_info);
      ghost_comm.exchange_ghosts( Uexchange );
    };


    if (m_gravity_type & GRAVITY_FIELD) {
      if( !U.has_field("gx") )
        U.new_fields({"gx", "gy", "gz"});
      if( !U.has_field("gphi") )
        U.new_fields({"gphi"});
    }

    //U.exchange_ghosts( ghost_comm );
    std::vector<std::string> fields_to_exchange;
    if( godunov_updater )
    {
      fields_to_exchange.push_back("rho");
      fields_to_exchange.push_back("e_tot");
      fields_to_exchange.push_back("rho_vx");
      fields_to_exchange.push_back("rho_vy");
      fields_to_exchange.push_back("rho_vz");
      for (int i=0; i < n_passive_scalars; ++i) {
        std::ostringstream oss;
        oss << "rho_scalar_" << i;
        fields_to_exchange.push_back(oss.str());
      }
    }
    if( this->has_mhd )
    {
      fields_to_exchange.push_back("Bx");
      fields_to_exchange.push_back("By");
      fields_to_exchange.push_back("Bz");
      if (this->is_glm)
        fields_to_exchange.push_back({"psi"});
    }
    if (m_gravity_type & GRAVITY_FIELD)
    {
      fields_to_exchange.push_back("gphi");
    }

    if( this->rad_updater ){
      const int n_groups = m_scalar_data.get<int>("n_groups");
      // Push by group
      for (int g = 0; g < n_groups; g++) {
        fields_to_exchange.push_back("e_rad_" + std::to_string(g));
        fields_to_exchange.push_back("fx_rad_" + std::to_string(g));
        fields_to_exchange.push_back("fy_rad_" + std::to_string(g));
        fields_to_exchange.push_back("fz_rad_" + std::to_string(g));
      }
    }

    timers.get("MPI ghosts").start();
    communicate_ghosts( fields_to_exchange );
    timers.get("MPI ghosts").stop();

    // Update gravity
    if( gravity_solver )
    {
      if( particle_update_density )
      {
        U.new_fields({"rho_g"});
        particle_update_density->update( U, m_scalar_data );

        // Backup rho without projected particles
        U.move_field("rho_bak", "rho");
        U.move_field("rho", "rho_g");
      }

      gravity_solver->update_gravity_field(U, m_scalar_data);

      // Maybe put this only in CIC move since we don't need it for NGP
      timers.get("MPI ghosts").start();
      communicate_ghosts( {"gx", "gy", "gz"} );
      timers.get("MPI ghosts").stop();

      // Restore rho before projection (only if particle projection)
      if( particle_update_density )
        U.move_field("rho", "rho_bak");
    }

    // Move particles
    if( particle_position_updater )
    {
      particle_position_updater->update( U, m_scalar_data );
      U.distributeParticles("dark_matter");
    }

    // Update hydro
    if( godunov_updater )
    {
      U.new_fields({"rho_next", "e_tot_next", "rho_vx_next", "rho_vy_next", "rho_vz_next"});    
      if (n_passive_scalars > 0) {
        std::set<std::string> scalar_next_fields;
        for (int i=0; i < n_passive_scalars; ++i) {
          scalar_next_fields.insert("rho_scalar_" + std::to_string(i) + "_next");
        }
        U.new_fields(scalar_next_fields);
      }

      // TODO automatic new fields according to kernel
      if( this->has_mhd ) {
        U.new_fields({"Bx_next", "By_next", "Bz_next"});
        if (this->is_glm)
          U.new_fields({"psi_next"});
      }

      godunov_updater->update( U, m_scalar_data );

      if( rad_updater )
      {
        const int n_groups = m_scalar_data.get<int>("n_groups");
        // generate update fields for each group
        for (int g = 0; g < n_groups; g++) {
          auto suffix = "_" + std::to_string(g) + "_next";
          U.new_fields({"e_rad" + suffix, "fx_rad" + suffix, "fy_rad" + suffix, "fz_rad" + suffix});
        }
        // This now has an overriden update in RadUpdate_euler
        // that cycles over radiation groups and updates them.
        rad_updater->update( U, m_scalar_data );
      }

      if(gravity_update)
        gravity_update->update( U, m_scalar_data );

      if ( viscosity_updater )
        viscosity_updater->update( U, m_scalar_data );
      if ( thermal_conduction_updater )
        thermal_conduction_updater->update( U, m_scalar_data );

      // Renaming passive scalars before source update to have the correct names in the source updater kernels
      if (n_passive_scalars > 0) {
        for (int i=0; i < n_passive_scalars; ++i) {
          std::ostringstream oss;
          oss << "rho_scalar_" << i << "_next";
          U.move_field(passive_scalars_names[i], oss.str());
        }
      }

      printf("Updating source terms...\n");
      for (auto &source_updater : source_updaters)
        source_updater->update( U, m_scalar_data );
      printf("Source terms updated.\n");

      // Moving back passive scalars after source update
      if (n_passive_scalars > 0) {
        for (int i=0; i < n_passive_scalars; ++i) {
          std::ostringstream oss;
          oss << "rho_scalar_" << i << "_next";
          U.move_field(oss.str(), passive_scalars_names[i]);
        }
      }

      U.move_field( "rho", "rho_next" );
      U.move_field( "e_tot", "e_tot_next" );
      U.move_field( "rho_vx", "rho_vx_next" );
      U.move_field( "rho_vy", "rho_vy_next" );
      U.move_field( "rho_vz", "rho_vz_next" );

      for (int i=0; i < n_passive_scalars; ++i) {
        std::ostringstream oss;
        oss << "rho_scalar_" << i;
        U.move_field(oss.str(), oss.str() + "_next");
      }

      // U.move_field( "rho_scalar_0", "rho_scalar_0_next" );
      // U.move_field( "rho_scalar_1", "rho_scalar_1_next" );
      // U.move_field( "rho_scalar_2", "rho_scalar_2_next" );
      // U.move_field( "rho_scalar_3", "rho_scalar_3_next" );
      if( this->has_mhd )
      {
        U.move_field( "Bx", "Bx_next" );
        U.move_field( "By", "By_next" );
        U.move_field( "Bz", "Bz_next" );
        if (this->is_glm)
          U.move_field( "psi", "psi_next" );
      }
      if(rad_updater)
      {
        const int n_groups = m_scalar_data.get<int>("n_groups");
        // Move by group
        for (int g = 0; g < n_groups; g++) {
          auto suffix = "_" + std::to_string(g);
          U.move_field("e_rad"  + suffix, "e_rad"  + suffix + "_next");
          U.move_field("fx_rad" + suffix, "fx_rad" + suffix + "_next");
          U.move_field("fy_rad" + suffix, "fy_rad" + suffix + "_next");
          U.move_field("fz_rad" + suffix, "fz_rad" + suffix + "_next");
        }
      }
    }

    if( particle_spawn )
    {
      particle_spawn->update( U, m_scalar_data );
    }

    // Particle source terms
    for (auto &particle_source_updater : particle_source_updaters)
      particle_source_updater->update( U, m_scalar_data );

    m_iteration_handler->next_iter(m_scalar_data);

    if (m_gravity_type & GRAVITY_FIELD)
    {
      // U.delete_field("gx");
      // U.delete_field("gy");
      // U.delete_field("gz");
    }

    // AMR cycle
    {
      if( m_iteration_handler->amr_cycle_trigger(m_scalar_data) )
      {
        timers.get("AMR").start();

        timers.get("MPI ghosts").start();
        std::vector<std::string> all_fields;
        for (auto field : U.getEnabledFields())
          all_fields.push_back(field);
        communicate_ghosts( all_fields );
        timers.get("MPI ghosts").stop();

        timers.get("AMR: Mark cells").start();
        refine_condition->mark_cells( U, m_scalar_data );
        timers.get("AMR: Mark cells").stop();

        // Backup old mesh
        mapUserData->save_old_mesh(U);

        timers.get("AMR: adapt").start();
        // 1. adapt mesh with mapper enabled
        m_amr_mesh->adapt();
        // Verify that adapt() doesn't need another iteration (expensive : only debug)
        timers.get("AMR: adapt").stop();

        // Resize and fill U with copied/interpolated/extrapolated data
        timers.get("AMR: remap userdata").start();
        mapUserData->remap(U);

        //TODO
        //std::cout << "Resize U after remap : " << DataArrayBlock::required_allocation_size(U2.U.extent(0), U2.U.extent(1), U2.U.extent(2)) * (2/1e6)
        //    << " -> " << DataArrayBlock::required_allocation_size(U2.U.extent(0), U2.U.extent(1), m_amr_mesh->getNumOctants()) * (2/1e6) << " MBytes" << std::endl;

        timers.get("AMR: remap userdata").stop();

        U.distributeAllParticles();

        timers.get("AMR").stop();
      }
    }

    //Load Balancing
    {
      if( m_iteration_handler->loadbalance_trigger(m_scalar_data)  )
      {
        timers.get("AMR: load-balance").start();

        m_amr_mesh->loadBalance_userdata(m_loadbalance_coherent_levels, U);
        U.distributeAllParticles();

        timers.get("AMR: load-balance").stop();
      }
    }
  }

private:
  // Simulation parameters
  int m_nlog; //! Timestep log frequency
  bool m_enable_output; //! Enable vizualization output and output at least at beginning and end of simulation
  bool m_enable_checkpoint; //! Enable checkpoint output and output at least at beginning and end of simulation

  GravityType m_gravity_type;

  int m_iter_start; //! First iteration (for restart)

  ScalarSimulationData m_scalar_data;

  MpiComm m_communicator;
  std::shared_ptr<AMRmesh> m_amr_mesh;
  ForeachCell m_foreach_cell;
  UserData U;
  int m_loadbalance_coherent_levels;

  using CellArray = ForeachCell::CellArray_global_ghosted;

  std::unique_ptr<IterationHandler> m_iteration_handler;
  std::vector<std::unique_ptr<Compute_dt>> compute_dt;
  std::unique_ptr<RefineCondition> refine_condition;
  std::unique_ptr<HyperbolicUpdate> godunov_updater;
  std::unique_ptr<HyperbolicUpdate> rad_updater;
  bool has_mhd, is_glm; // TODO : remove this
  int ghost_count; // TODO : remove this
  std::unique_ptr<ParticleUpdate> particle_position_updater, particle_update_density, particle_spawn;
  std::unique_ptr<MapUserData> mapUserData;
  std::unique_ptr<IOManager> io_manager, io_manager_checkpoint;
  std::unique_ptr<Output_Movie> output_movie;
  std::unique_ptr<GravitySolver> gravity_solver;
  std::unique_ptr<HyperbolicUpdate> gravity_update;

  std::unique_ptr<CosmoManager> cosmo_manager;

  std::unique_ptr<ParabolicUpdate> thermal_conduction_updater;
  std::unique_ptr<ParabolicUpdate> viscosity_updater;
  std::vector<std::unique_ptr<SourceUpdate>> source_updaters;
  std::vector<std::unique_ptr<ParticleUpdate>> particle_source_updaters;

  int n_passive_scalars;
  std::vector<std::string> passive_scalars_names;

  Timers timers;
};

int DyabloTimeLoop::interrupted = false;


} // namespace dyablo
