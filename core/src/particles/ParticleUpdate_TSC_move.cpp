#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo {

class ParticleUpdate_TSC_move : public ParticleUpdate {
public:
  ParticleUpdate_TSC_move(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    data{
      .xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0),
      .xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0),      
      .ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0),
      .ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0),
      .zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0),
      .zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0)
    }
  {}

  ~ParticleUpdate_TSC_move() {}

  void update( UserData& U, ScalarSimulationData& scalar_data) 
  {
    if( foreach_cell.getDim() == 2 )
      update_aux<2>(U, scalar_data);
    else
      update_aux<3>(U, scalar_data);
  }
  
  template< int ndim>
  void update_aux( UserData& U, ScalarSimulationData& scalar_data) 
  {
    using pos_t = Kokkos::Array<real_t, 3>;

    timers.get("ParticleUpdate_TSC_move").start();

    const real_t dt = scalar_data.get<real_t>("dt"); 

    enum VarIndex_g{
      IGX,IGY,IGZ
    };
    enum VarIndex_particle{
      IVX,IVY,IVZ
    };

    auto Uin = U.getAccessor( {{"gx", IGX},{"gy", IGY},{"gz", IGZ}} );
    const ForeachParticle::ParticleArray& Ppos = U.getParticleArray( "particles" );
    UserData::ParticleAccessor Pdata = U.getParticleAccessor( "particles", {{"vx", IVX},{"vy", IVY},{"vz", IVZ}} );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    const Data& d = this->data;

    foreach_particle.foreach_particle( "particles_update_position", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
      ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );
      ForeachCell::SearchMode_neighbor search_neighbor( cells.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

      pos_t cell_size = cells.getCellSize( iCell );
      pos_t cell_pos = cells.getCellCenter( iCell );

      real_t gx=0, gy=0, gz=0;

      pos_t p = { // Local position relative to center [-0.5, 0.5]^3
        (part_pos[IX] - cell_pos[IX])/cell_size[IX], 
        (part_pos[IY] - cell_pos[IY])/cell_size[IY], 
        (part_pos[IZ] - cell_pos[IZ])/cell_size[IZ] 
      };
      
      pos_t v_in =  {0.75-p[IX]*p[IX], 0.75-p[IY]*p[IY], 0.75-p[IZ]*p[IZ]};  // volume fraction in local cell [-0.5, 0.5]
      pos_t v_out_minus = { 0.5*(0.5 - p[IX])*(0.5 - p[IX]), 
                            0.5*(0.5 - p[IY])*(0.5 - p[IY]),
                            0.5*(0.5 - p[IZ])*(0.5 - p[IZ])};     // volume fraction in neighbor cells [-1.5, -0.5]
      pos_t v_out_plus = {  0.5*(0.5 + p[IX])*(0.5 + p[IX]), 
                            0.5*(0.5 + p[IY])*(0.5 + p[IY]),
                            0.5*(0.5 + p[IZ])*(0.5 + p[IZ])};     // volume fraction in neighbor cells [0.5, 1.5]

      auto apply_g_contrib = [&]( const ForeachCell::CellIndex::offset_t& offset )
      {
        ForeachCell::CellIndex iCell_neighbor = iCell.getNeighbor(offset, search_neighbor);
        real_t cx = (offset[IX]==0)?v_in[IX]:(offset[IX]==-1)?v_out_minus[IX]: v_out_plus[IX];
        real_t cy = (offset[IY]==0)?v_in[IY]:(offset[IY]==-1)?v_out_minus[IY]: v_out_plus[IY];
        real_t cz = (offset[IZ]==0)?v_in[IZ]:(offset[IZ]==-1)?v_out_minus[IZ]: v_out_plus[IZ];
        real_t volume_fraction = cx*cy*cz;

        assert( abs( iCell_neighbor.level_diff() ) <= 1 );

        if( iCell_neighbor.level_diff() >= 0 )
        { // bigger or same size : only one cell to write
          gx += Uin.at( iCell_neighbor, IGX) * volume_fraction;
          gy += Uin.at( iCell_neighbor, IGY) * volume_fraction;
          gz += Uin.at( iCell_neighbor, IGZ) * volume_fraction;
        }
        else 
        {
          //int nneighbors = 2*(ndim-1);
          int di_count = (offset[IX]==0)?2:1;
          int dj_count = (offset[IY]==0)?2:1;
          int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
          int nneighbors = di_count*dj_count*dk_count;
          foreach_smaller_neighbor<ndim>( iCell_neighbor, offset, search_neighbor,
            [&]( const ForeachCell::CellIndex& iCell_sn )
          {
            gx += Uin.at( iCell_sn, IGX) * volume_fraction / nneighbors;
            gy += Uin.at( iCell_sn, IGY) * volume_fraction / nneighbors;
            gz += Uin.at( iCell_sn, IGZ) * volume_fraction / nneighbors;
          });
        }
      };

      for (int16_t ix=-1; ix<=1; ix++)
      for (int16_t iy=-1; iy<=1; iy++)
      for (int16_t iz=(ndim==3)?-1:0; iz<=(ndim==3)?1:0; iz++)
        apply_g_contrib( {ix, iy, iz} );

      Pdata.at( iPart, IVX ) += dt * gx;
      Pdata.at( iPart, IVY ) += dt * gy;
      Pdata.at( iPart, IVZ ) += dt * gz;
      Ppos.pos(iPart, IX) += dt * Pdata.at( iPart, IVX );
      Ppos.pos(iPart, IY) += dt * Pdata.at( iPart, IVY );
      Ppos.pos(iPart, IZ) += dt * Pdata.at( iPart, IVZ );

      // Compute periodic position
      Ppos.pos(iPart, IX) = fmod( (Ppos.pos(iPart, IX) - d.xmin) + (d.xmax-d.xmin) , d.xmax-d.xmin) + d.xmin;
      Ppos.pos(iPart, IY) = fmod( (Ppos.pos(iPart, IY) - d.ymin) + (d.ymax-d.ymin) , d.ymax-d.ymin) + d.ymin;
      Ppos.pos(iPart, IZ) = fmod( (Ppos.pos(iPart, IZ) - d.zmin) + (d.zmax-d.zmin) , d.zmax-d.zmin) + d.zmin;

    });   

    timers.get("ParticleUpdate_TSC_move").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;  
public: // Needed for nvcc
  struct Data {
    real_t xmin,xmax,ymin,ymax,zmin,zmax;
  } data;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory, 
                  dyablo::ParticleUpdate_TSC_move, 
                  "ParticleUpdate_TSC_move")