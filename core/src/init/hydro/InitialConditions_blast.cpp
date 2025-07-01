#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_hydro.hpp"

namespace dyablo{

namespace{

Kokkos::Array<real_t, 3> vector_to_array( const std::vector<real_t>& in )
{
    return Kokkos::Array<real_t, 3> {
        in.size() >= 1 ? in[0] : 0,
        in.size() >= 2 ? in[1] : 0,
        in.size() >= 3 ? in[2] : 0,
    };
};
}

struct AnalyticalFormula_blast : public AnalyticalFormula_base_hydro
{
     // blast problem parameters
    const int ndim;
    const real_t blast_radius;
    const real_t blast_center_x;
    const real_t blast_center_y;
    const real_t blast_center_z;
    const real_t blast_density_in;
    const real_t blast_density_out;
    const real_t blast_pressure_in;
    const real_t blast_pressure_out;
    Kokkos::Array<real_t, 3> blast_velocity_in;
    Kokkos::Array<real_t, 3> blast_velocity_out;
    const int blast_nx;
    const int blast_ny;
    const int blast_nz;
    const real_t error_max;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;    
    const real_t gamma0, smallr, smallc, smallp;
    
    AnalyticalFormula_blast( ConfigMap& configMap ) :
        ndim( configMap.getValue<int>("mesh", "ndim", 3) ),
        // Length are scaled by quadrant width (0.5,0.5,0.5 is center of quadrant when blast_n* != 1)
        blast_radius ( configMap.getValue<real_t>("blast","radius", 0.1) ),
        blast_center_x ( configMap.getValue<real_t>("blast","center_x", 0.5) ),
        blast_center_y ( configMap.getValue<real_t>("blast","center_y", 0.5) ),
        blast_center_z ( configMap.getValue<real_t>("blast","center_z", 0.5) ),
        blast_density_in ( configMap.getValue<real_t>("blast","density_in", 1.0) ),
        blast_density_out ( configMap.getValue<real_t>("blast","density_out", 1.2) ),
        blast_pressure_in ( configMap.getValue<real_t>("blast","pressure_in", 10.0) ),
        blast_pressure_out ( configMap.getValue<real_t>("blast","pressure_out", 0.1) ),
        blast_velocity_in (vector_to_array( configMap.getValue<std::vector<real_t>>("blast","velocity_in", {0, 0, 0}))),
        blast_velocity_out (vector_to_array( configMap.getValue<std::vector<real_t>>("blast","velocity_out", {0, 0, 0}))),
        // Number of quadrants in each direction
        blast_nx ( configMap.getValue<int>("blast", "blast_nx", 1) ),
        blast_ny ( configMap.getValue<int>("blast", "blast_ny", 1) ),
        blast_nz ( configMap.getValue<int>("blast", "blast_nz", 1) ),  
        error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),      
        xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
        ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
        zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
        gamma0 ( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
        smallr ( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
        smallc ( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
        smallp ( smallc*smallc / gamma0 )
    {}

    // Geometrical version
    // KOKKOS_INLINE_FUNCTION
    // bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    // {
    //     // Quadrant size
    //     real_t qsx = 1.0 / this->blast_nx;
    //     real_t qsy = 1.0 / this->blast_ny;
    //     real_t qsz = (this->ndim == 3) ? 1.0 / this->blast_nz : 1.0;
    //     real_t qs = FMIN(qsx, FMIN( qsy, qsz ) );
    //     real_t radius = this->blast_radius*qs;
    //     // Quadrant logical position
    //     int qix = (int)(x / qsx);
    //     int qiy = (int)(y / qsy);
    //     int qiz = (int)(z / qsz);
    //     // Quadrant physical center
    //     real_t qcx = (qix+0.5)*qsx;
    //     real_t qcy = (qiy+0.5)*qsy;
    //     real_t qcz = (qiz+0.5)*qsz;

    //     // Two refinement criteria are used : 
    //     //  1- If the cell size is larger than a quadrant we refine        
    //     bool should_refine = dx > qsx || dy > qsy || dz > qsz;

    //     //  2- If the distance to the interface is smaller than the size of
    //     //     half a diagonal we refine
    //     // Squared distance to quadrant center
    //     real_t r2 = (x-qcx)*(x-qcx) + (y-qcy)*(y-qcy);
    //     if( this->ndim == 3 ) r2 += (z-qcz)*(z-qcz);

    //     real_t half_cell_diag = (this->ndim == 3) ?
    //         sqrt(dx*dx+dy*dy+dz*dz)/2 :
    //         sqrt(dx*dx+dy*dy)/2 ;

    //     if( std::abs( std::sqrt(r2) - radius ) < half_cell_diag )
    //         should_refine = true;

    //     return should_refine;
    // } 

    KOKKOS_INLINE_FUNCTION
    State value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        // Quadrant size
        real_t qsx = 1.0 / this->blast_nx;
        real_t qsy = 1.0 / this->blast_ny;
        real_t qsz = (this->ndim == 3 ? 1.0 / this->blast_nz : 1.0);
        real_t qs = FMIN(qsx, FMIN( qsy, qsz ) );
        real_t radius = this->blast_radius*qs;
        // Quadrant logical position
        int qix = (int)(x / qsx);
        int qiy = (int)(y / qsy);
        int qiz = (int)(z / qsz);
        // Quadrant physical center
        real_t qcx = (qix+0.5)*qsx;
        real_t qcy = (qiy+0.5)*qsy;
        real_t qcz = (qiz+0.5)*qsz;

        real_t r2 = (x-qcx)*(x-qcx) + (y-qcy)*(y-qcy);
        if( this->ndim == 3 ) r2 += (z-qcz)*(z-qcz);
        
        HyperbolicPolicy_State_Hydro::PrimState q;

        if (r2 < radius*radius) {
            q.rho = blast_density_in;
            q.p = blast_pressure_in;
            q.u = blast_velocity_in[IX];
            q.v = ndim >= 2 ? blast_velocity_in[IY] : 0; 
            q.w = ndim >= 3 ? blast_velocity_in[IZ] : 0;
        } else {
            q.rho = blast_density_out;
            q.p = blast_pressure_out;
            q.u = blast_velocity_out[IX];
            q.v = ndim >= 2 ? blast_velocity_out[IY] : 0;
            q.w = ndim >= 3 ? blast_velocity_out[IZ] : 0;
        }

        return HyperbolicPolicy_State_Hydro({ndim, gamma0}).primToCons(q);
    } 
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_blast>, 
                "blast");