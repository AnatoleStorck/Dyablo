#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ScalarSimulationData.h"
#include "UserData.h"
#include "foreach_cell/ForeachCell.h"
#include "utils/config/ConfigMap.h"
#include "utils/units/Units.h"

namespace dyablo {

/**
 * Movie output : column-integrated projections of the simulation.
 *
 * The simulation may use an AMR grid : each cell is deposited onto a fixed
 * resolution output image weighted by the physical overlap between the cell and
 * the image pixels, so that cells of any refinement level are handled
 * consistently. The stored quantity is the area-averaged column integral
 *   I(a,b) = (1/A_pixel) * integral_over_pixel( integral_along_axis( value dl ) )
 * which has units [value] * [length] and is independent of the chosen output
 * resolution.
 *
 * The projected region can be restricted to a sub-volume centered (by default)
 * on the box center, which is useful for deep zoom-in simulations.
 *
 * Relevant parameters (section [movie]) :
 *   enabled            : (bool)   enable movie output
 *   output_frequency   : (int)    write every N iterations
 *   time_cadence       : (time)   write roughly every <time> of simulation time, e.g. "0.5 Myr"
 *   trigger_on_output  : (bool)   also write whenever a regular output is written
 *   output_first_iter  : (bool)   write on the first iteration
 *   projection_axis    : (csv)    one or more of x,y,z
 *   output_variables   : (csv)    fields to project (default : all enabled fields)
 *   image_resolution   : (int)    number of pixels along the longest side of the image
 *   width              : (length) transverse field of view (square). <=0 => full box
 *   depth              : (length) line-of-sight integration depth. <=0 => width if set, else full box
 *   center_x/y/z       : (length) center of the projected region (default : box center)
 */
class Output_Movie
{
public:
  Output_Movie(ConfigMap& configMap, ForeachCell& foreach_cell)
    : m_foreach_cell(foreach_cell),
      m_output_dir(std::filesystem::path(configMap.getValue<std::string>("output", "outputDir", ".")) / "movie_files"),
      m_frequency(configMap.getValue<int>("movie", "output_frequency", -1)),
      m_trigger_on_output(configMap.getValue<bool>("movie", "trigger_on_output", true)),
      m_output_first_iter(configMap.getValue<bool>("movie", "output_first_iter", true)),
      m_image_resolution(configMap.getValue<int>("movie", "image_resolution", 800)),
      m_xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
      m_xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
      m_ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
      m_ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
      m_zmin(configMap.getValue<real_t>("mesh", "zmin", 0.0)),
      m_zmax(configMap.getValue<real_t>("mesh", "zmax", 1.0))
  {

    m_enabled = configMap.getValue<bool>("movie", "enabled", false);

    // Transverse field of view and line-of-sight depth (code units, 0 => auto).
    // "width"/"depth" accept either a raw number (already in code units) or a
    // value with a physical unit, e.g. "1 kpc".
    m_width = configMap.getValue_in_code_unit<Units::Length>("movie", "width", "0");
    m_depth = configMap.getValue_in_code_unit<Units::Length>("movie", "depth", "0");

    // Center of the projected region, default to the box center.
    m_center_x = configMap.getValue_in_code_unit<Units::Length>("movie", "center_x", real_t(0.5) * (m_xmin + m_xmax));
    m_center_y = configMap.getValue_in_code_unit<Units::Length>("movie", "center_y", real_t(0.5) * (m_ymin + m_ymax));
    m_center_z = configMap.getValue_in_code_unit<Units::Length>("movie", "center_z", real_t(0.5) * (m_zmin + m_zmax));

    // Output (approximately) every m_time_cadence of simulation time. This is a
    // physical time (code units, 0 => disabled) and, because the timestep
    // varies, only guarantees that at least this much time has elapsed since the
    // previous time-cadence frame. Accepts a raw number (code units) or a value
    // with a unit, e.g. "0.5 Myr".
    m_time_cadence = configMap.getValue_in_code_unit<Units::Time>("movie", "time_cadence", "0");

    std::string axis_str = configMap.getValue<std::string>(
      "movie",
      "projection_axis",
      configMap.getValue<std::string>("movie", "axis", "z"));
    std::vector<std::string> axis_tokens = parse_csv(axis_str);
    if( axis_tokens.empty() )
      axis_tokens.push_back("z");
    std::vector<std::string> invalid_axis_tokens;

    std::set<Axis> seen_axes;
    for( const std::string& axis_token : axis_tokens )
    {
      Axis axis;
      if( try_parse_axis(axis_token, axis) )
      {
        if( seen_axes.insert(axis).second )
          m_axes.push_back(axis);
      }
      else
      {
        invalid_axis_tokens.push_back(axis_token);
      }
    }

    if( m_axes.empty() )
      m_axes.push_back(Axis::Z);

    m_requested_variables = parse_csv(configMap.getValue<std::string>("movie", "output_variables", ""));

    int mpi_rank = m_foreach_cell.get_amr_mesh().getMpiComm().MPI_Comm_rank();
    if( mpi_rank == 0 && m_enabled )
    {
      std::cout << "Movie output enabled in '" << m_output_dir.string() << "'" << std::endl;
      if( !invalid_axis_tokens.empty() )
      {
        std::cout << "WARNING : invalid movie/projection_axis value(s) ignored : ";
        for( size_t i = 0; i < invalid_axis_tokens.size(); ++i )
        {
          if( i > 0 )
            std::cout << ", ";
          std::cout << "'" << invalid_axis_tokens[i] << "'";
        }
        std::cout << std::endl;
      }
    }
  }

  bool is_enabled() const
  {
    return m_enabled;
  }

  bool should_output(const ScalarSimulationData& scalar_data, bool regular_output_written)
  {
    if( !m_enabled )
      return false;

    bool first_iter = m_output_first_iter;
    m_output_first_iter = false;

    int iter = scalar_data.get<int>("iter");
    bool frequency_trigger = (m_frequency > 0) && (iter % m_frequency == 0);
    bool output_trigger = m_trigger_on_output && regular_output_written;
    bool cadence_trigger = time_cadence_elapsed(scalar_data);

    return first_iter || frequency_trigger || output_trigger || cadence_trigger;
  }

  void save_frame(const UserData& U, const ScalarSimulationData& scalar_data)
  {
    if( !m_enabled )
      return;

    AMRmesh& amr_mesh = m_foreach_cell.get_amr_mesh();
    MpiComm mpi_comm = amr_mesh.getMpiComm();
    int mpi_rank = mpi_comm.MPI_Comm_rank();

    std::vector<std::string> output_variables = resolve_output_variables(U, mpi_rank);
    if( output_variables.empty() )
      return;

    std::vector<AxisPlan> valid_axis_plans = resolve_axis_plans(mpi_rank);

    if( valid_axis_plans.empty() )
      return;

    if( mpi_rank == 0 )
      std::filesystem::create_directories(m_output_dir);

    auto storage = amr_mesh.getStorage();

    const uint32_t bx = m_foreach_cell.blockSize()[IX];
    const uint32_t by = m_foreach_cell.blockSize()[IY];
    const uint32_t bz = m_foreach_cell.blockSize()[IZ];
    const uint32_t nbOcts_local = amr_mesh.getNumOctants();

    const int level_min = amr_mesh.get_level_min();
    const auto coarse = amr_mesh.get_coarse_grid_size();

    const double Lx = static_cast<double>(m_xmax) - static_cast<double>(m_xmin);
    const double Ly = static_cast<double>(m_ymax) - static_cast<double>(m_ymin);
    const double Lz = static_cast<double>(m_zmax) - static_cast<double>(m_zmin);

    int iter = scalar_data.get<int>("iter");

    for( const std::string& var_name : output_variables )
    {
      auto field = U.getField(var_name);
      auto field_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.U);

      for( const AxisPlan& axis_plan : valid_axis_plans )
      {
        const Axis axis = axis_plan.axis;
        const ProjectionGeometry& geom = axis_plan.geom;

        std::vector<double> projection_local(geom.projection_size, 0.0);

        for( uint32_t iOct = 0; iOct < nbOcts_local; ++iOct )
        {
          const int level = static_cast<int>(storage.getLevel({iOct, false}));
          const int shift = level - level_min;

          // Number of cells along each dimension at this octant's level, taking
          // the block subdivision into account.
          const uint64_t ncx = (static_cast<uint64_t>(coarse[IX]) << shift) * bx;
          const uint64_t ncy = (static_cast<uint64_t>(coarse[IY]) << shift) * by;
          const uint64_t ncz = (static_cast<uint64_t>(coarse[IZ]) << shift) * bz;

          const double dxc = Lx / static_cast<double>(ncx);
          const double dyc = Ly / static_cast<double>(ncy);
          const double dzc = Lz / static_cast<double>(ncz);

          auto logical = storage.get_logical_coords({iOct, false});
          const uint64_t oct_x = logical[IX];
          const uint64_t oct_y = logical[IY];
          const uint64_t oct_z = logical[IZ];

          for( uint32_t k = 0; k < bz; ++k )
          {
            for( uint32_t j = 0; j < by; ++j )
            {
              for( uint32_t i = 0; i < bx; ++i )
              {
                // Physical bounding box of this cell (code units).
                const double x0 = static_cast<double>(m_xmin) + static_cast<double>(oct_x * bx + i) * dxc;
                const double y0 = static_cast<double>(m_ymin) + static_cast<double>(oct_y * by + j) * dyc;
                const double z0 = static_cast<double>(m_zmin) + static_cast<double>(oct_z * bz + k) * dzc;

                const uint32_t iCell = i + bx * (j + by * k);
                const double value = static_cast<double>(field_host(iCell, 0, iOct));

                deposit_cell(axis, x0, x0 + dxc, y0, y0 + dyc, z0, z0 + dzc,
                             value, geom, projection_local);
              }
            }
          }
        }

        std::vector<double> projection_global(geom.projection_size, 0.0);
        mpi_comm.MPI_Allreduce(
          projection_local.data(),
          projection_global.data(),
          static_cast<int>(geom.projection_size),
          MpiComm::MPI_Op_t::SUM);

        if( mpi_rank == 0 )
        {
          // Normalize the accumulated (value * depth * area) by the pixel area
          // to obtain the area-averaged column integral.
          for( double& v : projection_global )
            v *= geom.inv_pixel_area;

          write_projection(axis, var_name, iter, projection_global, geom);
        }
      }
    }
  }

private:
  enum class Axis
  {
    X,
    Y,
    Z
  };

  // Output image geometry and the physical region it covers, for one axis.
  // The image has numpy shape (res_b, res_a) and is stored in C order, so the
  // pixel at column a, row b lives at index a + res_a * b.
  struct ProjectionGeometry
  {
    // Image dimensions.
    size_t res_a = 0; // horizontal axis, fast (column) index
    size_t res_b = 0; // vertical axis, slow (row) index
    size_t projection_size = 0;

    // Projected region in the image plane (code units).
    double a_min = 0;
    double b_min = 0;
    double a_max = 0;
    double b_max = 0;
    double ps_a = 0; // pixel size along a
    double ps_b = 0; // pixel size along b

    // Integration range along the projection axis (code units).
    double d_min = 0;
    double d_max = 0;

    double inv_pixel_area = 0; // 1 / (ps_a * ps_b)
  };

  struct AxisPlan
  {
    Axis axis = Axis::Z;
    ProjectionGeometry geom;
  };

  // Box geometry expressed in the (a, b, depth) frame of a projection axis,
  // where (a, b) are the two image-plane axes and depth is the projection axis.
  struct AxisFrame
  {
    double La = 0;            // box length along a
    double Lb = 0;            // box length along b
    double Ldepth = 0;        // box length along the projection axis
    double a_box_min = 0;     // box minimum along a
    double b_box_min = 0;     // box minimum along b
    double depth_box_min = 0; // box minimum along the projection axis
    double ca = 0;            // region center along a
    double cb = 0;            // region center along b
    double cdepth = 0;        // region center along the projection axis
  };

  // Returns true when at least m_time_cadence of simulation time has elapsed
  // since the last time-cadence frame. The reference time is snapped to a
  // regular grid (multiples of m_time_cadence) so that the movie sampling does
  // not slowly drift even though the timestep varies between iterations.
  bool time_cadence_elapsed(const ScalarSimulationData& scalar_data)
  {
    if( m_time_cadence <= 0 )
      return false;

    real_t time = scalar_data.get<real_t>("time");

    if( !m_cadence_initialized )
    {
      m_last_cadence_time = std::floor(time / m_time_cadence) * m_time_cadence;
      m_cadence_initialized = true;
    }

    if( time - m_last_cadence_time >= m_time_cadence )
    {
      m_last_cadence_time = std::floor(time / m_time_cadence) * m_time_cadence;
      return true;
    }

    return false;
  }

  static std::string trim_copy(std::string s)
  {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
  }

  static std::string to_lower_copy(std::string s)
  {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return s;
  }

  static bool try_parse_axis(const std::string& axis, Axis& parsed_axis)
  {
    std::string axis_lower = to_lower_copy(trim_copy(axis));
    if( axis_lower == "x" )
    {
      parsed_axis = Axis::X;
      return true;
    }
    if( axis_lower == "y" )
    {
      parsed_axis = Axis::Y;
      return true;
    }
    if( axis_lower == "z" )
    {
      parsed_axis = Axis::Z;
      return true;
    }

    return false;
  }

  static const char* axis_tag(Axis axis)
  {
    switch( axis )
    {
    case Axis::X:
      return "x";
    case Axis::Y:
      return "y";
    case Axis::Z:
      return "z";
    }

    return "z";
  }

  static std::vector<std::string> parse_csv(const std::string& csv)
  {
    std::vector<std::string> values;
    std::stringstream sstream(csv);
    std::string token;
    while( std::getline(sstream, token, ',') )
    {
      token = trim_copy(token);
      if( !token.empty() )
        values.push_back(token);
    }
    return values;
  }

  // Express the box and the region center in the (a, b, depth) frame of the
  // given projection axis :
  //   Axis Z : a = x, b = y, depth = z
  //   Axis X : a = y, b = z, depth = x
  //   Axis Y : a = x, b = z, depth = y
  AxisFrame axis_frame(Axis axis) const
  {
    const double Lx = static_cast<double>(m_xmax) - static_cast<double>(m_xmin);
    const double Ly = static_cast<double>(m_ymax) - static_cast<double>(m_ymin);
    const double Lz = static_cast<double>(m_zmax) - static_cast<double>(m_zmin);

    AxisFrame f;
    switch( axis )
    {
    case Axis::Z:
      f.La = Lx; f.Lb = Ly; f.Ldepth = Lz;
      f.a_box_min = m_xmin; f.b_box_min = m_ymin; f.depth_box_min = m_zmin;
      f.ca = m_center_x; f.cb = m_center_y; f.cdepth = m_center_z;
      break;
    case Axis::X:
      f.La = Ly; f.Lb = Lz; f.Ldepth = Lx;
      f.a_box_min = m_ymin; f.b_box_min = m_zmin; f.depth_box_min = m_xmin;
      f.ca = m_center_y; f.cb = m_center_z; f.cdepth = m_center_x;
      break;
    case Axis::Y:
      f.La = Lx; f.Lb = Lz; f.Ldepth = Ly;
      f.a_box_min = m_xmin; f.b_box_min = m_zmin; f.depth_box_min = m_ymin;
      f.ca = m_center_x; f.cb = m_center_z; f.cdepth = m_center_y;
      break;
    }
    return f;
  }

  bool compute_geometry(Axis axis, ProjectionGeometry& geom, std::string& reason) const
  {
    if( m_image_resolution <= 0 )
    {
      reason = "movie/image_resolution must be strictly positive";
      return false;
    }

    const AxisFrame f = axis_frame(axis);

    if( f.La <= 0 || f.Lb <= 0 || f.Ldepth <= 0 )
    {
      reason = "degenerate box dimensions";
      return false;
    }

    // Transverse field of view : square region of side m_width if requested,
    // otherwise the full box extent.
    double region_a, region_b;
    if( m_width > 0 )
    {
      region_a = m_width;
      region_b = m_width;
      geom.a_min = f.ca - 0.5 * m_width;
      geom.b_min = f.cb - 0.5 * m_width;
    }
    else
    {
      region_a = f.La;
      region_b = f.Lb;
      geom.a_min = f.a_box_min;
      geom.b_min = f.b_box_min;
    }
    geom.a_max = geom.a_min + region_a;
    geom.b_max = geom.b_min + region_b;

    // Line-of-sight integration range. Defaults to a cube (depth == width) when
    // a width is set, otherwise to the full box depth.
    double half_depth;
    bool restrict_depth = true;
    if( m_depth > 0 )
      half_depth = 0.5 * m_depth;
    else if( m_width > 0 )
      half_depth = 0.5 * m_width;
    else
      restrict_depth = false;

    if( restrict_depth )
    {
      geom.d_min = f.cdepth - half_depth;
      geom.d_max = f.cdepth + half_depth;
    }
    else
    {
      geom.d_min = f.depth_box_min;
      geom.d_max = f.depth_box_min + f.Ldepth;
    }

    // Pixel grid : square pixels, the longest side gets m_image_resolution pixels.
    const double longest = std::max(region_a, region_b);
    const double pixel_size = longest / static_cast<double>(m_image_resolution);
    if( pixel_size <= 0 )
    {
      reason = "non-positive pixel size";
      return false;
    }

    long res_a = std::max<long>(1, std::llround(region_a / pixel_size));
    long res_b = std::max<long>(1, std::llround(region_b / pixel_size));

    geom.res_a = static_cast<size_t>(res_a);
    geom.res_b = static_cast<size_t>(res_b);

    if( geom.res_b != 0 && geom.res_a > std::numeric_limits<size_t>::max() / geom.res_b )
    {
      reason = "projection dimensions overflow";
      return false;
    }
    geom.projection_size = geom.res_a * geom.res_b;

    if( geom.projection_size > static_cast<size_t>(std::numeric_limits<int>::max()) )
    {
      reason = "projection array is too large for MPI_Allreduce count";
      return false;
    }

    geom.ps_a = region_a / static_cast<double>(geom.res_a);
    geom.ps_b = region_b / static_cast<double>(geom.res_b);
    geom.inv_pixel_area = 1.0 / (geom.ps_a * geom.ps_b);

    return true;
  }

  std::vector<std::string> resolve_output_variables(const UserData& U, int mpi_rank)
  {
    std::vector<std::string> variables;
    if( m_requested_variables.empty() )
    {
      const std::set<std::string> enabled_fields = U.getEnabledFields();
      variables.assign(enabled_fields.begin(), enabled_fields.end());
      return variables;
    }

    variables.reserve(m_requested_variables.size());
    std::set<std::string> seen_variables;
    for( const std::string& name : m_requested_variables )
    {
      if( !seen_variables.insert(name).second )
        continue;

      if( U.has_field(name) )
      {
        variables.push_back(name);
      }
      else if( m_warned_missing_variables.insert(name).second && mpi_rank == 0 )
      {
        std::cout << "WARNING : movie output variable requested but not enabled : '" << name << "'" << std::endl;
      }
    }

    return variables;
  }

  std::vector<AxisPlan> resolve_axis_plans(int mpi_rank)
  {
    std::vector<AxisPlan> axis_plans;
    axis_plans.reserve(m_axes.size());

    for( Axis axis : m_axes )
    {
      ProjectionGeometry geom;
      std::string unsupported_reason;
      if( !compute_geometry(axis, geom, unsupported_reason) )
      {
        if( mpi_rank == 0 && m_warned_unsupported_axes.insert(axis).second )
        {
          std::cout << "WARNING : movie output skipped for axis '" << axis_tag(axis)
                    << "' - " << unsupported_reason << std::endl;
        }
        continue;
      }

      axis_plans.push_back({axis, geom});
    }

    return axis_plans;
  }

  // Deposit a single cell (physical bounding box, code units) onto the output
  // image, weighting by the overlap area with each pixel and by the overlap of
  // the cell along the projection axis with the integration range. The pixel
  // area normalization is applied later, once, to the reduced image.
  static void deposit_cell(Axis axis,
                           double x0, double x1,
                           double y0, double y1,
                           double z0, double z1,
                           double value,
                           const ProjectionGeometry& g,
                           std::vector<double>& projection)
  {
    double a0, a1, b0, b1, d0, d1;
    switch( axis )
    {
    case Axis::Z:
      a0 = x0; a1 = x1; b0 = y0; b1 = y1; d0 = z0; d1 = z1;
      break;
    case Axis::X:
      a0 = y0; a1 = y1; b0 = z0; b1 = z1; d0 = x0; d1 = x1;
      break;
    case Axis::Y:
      a0 = x0; a1 = x1; b0 = z0; b1 = z1; d0 = y0; d1 = y1;
      break;
    default:
      return;
    }

    // Overlap of the cell with the integration range along the axis.
    const double depth_overlap = std::min(d1, g.d_max) - std::max(d0, g.d_min);
    if( depth_overlap <= 0 )
      return;

    // Overlap of the cell footprint with the projected region.
    const double a_lo = std::max(a0, g.a_min);
    const double a_hi = std::min(a1, g.a_max);
    if( a_hi <= a_lo )
      return;
    const double b_lo = std::max(b0, g.b_min);
    const double b_hi = std::min(b1, g.b_max);
    if( b_hi <= b_lo )
      return;

    const long res_a = static_cast<long>(g.res_a);
    const long res_b = static_cast<long>(g.res_b);

    long ja0 = static_cast<long>(std::floor((a_lo - g.a_min) / g.ps_a));
    long ja1 = static_cast<long>(std::ceil((a_hi - g.a_min) / g.ps_a));
    long jb0 = static_cast<long>(std::floor((b_lo - g.b_min) / g.ps_b));
    long jb1 = static_cast<long>(std::ceil((b_hi - g.b_min) / g.ps_b));

    ja0 = std::max<long>(ja0, 0);
    jb0 = std::max<long>(jb0, 0);
    ja1 = std::min<long>(ja1, res_a);
    jb1 = std::min<long>(jb1, res_b);

    const double column = value * depth_overlap;

    for( long jb = jb0; jb < jb1; ++jb )
    {
      const double pb_lo = g.b_min + static_cast<double>(jb) * g.ps_b;
      const double overlap_b = std::min(b_hi, pb_lo + g.ps_b) - std::max(b_lo, pb_lo);
      if( overlap_b <= 0 )
        continue;

      const size_t row_offset = static_cast<size_t>(jb) * g.res_a;
      for( long ja = ja0; ja < ja1; ++ja )
      {
        const double pa_lo = g.a_min + static_cast<double>(ja) * g.ps_a;
        const double overlap_a = std::min(a_hi, pa_lo + g.ps_a) - std::max(a_lo, pa_lo);
        if( overlap_a <= 0 )
          continue;

        projection[row_offset + static_cast<size_t>(ja)] += column * overlap_a * overlap_b;
      }
    }
  }

  static void write_npy_2d(const std::filesystem::path& file_path,
                           const std::vector<double>& data,
                           size_t n0,
                           size_t n1)
  {
    std::ofstream out(file_path, std::ios::binary);
    DYABLO_ASSERT_HOST_RELEASE(out.good(), "Cannot open movie output file : " << file_path.string());

    std::ostringstream header_stream;
    header_stream << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << n0 << ", " << n1 << "), }";
    std::string header = header_stream.str();

    const size_t preamble_size = 10; // magic(6) + version(2) + header_len(2)
    size_t header_with_newline = header.size() + 1;
    size_t pad = (16 - ((preamble_size + header_with_newline) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');

    DYABLO_ASSERT_HOST_RELEASE(
      header.size() <= static_cast<size_t>(std::numeric_limits<uint16_t>::max()),
      "NumPy header too long for v1.0 format");

    const uint16_t header_len = static_cast<uint16_t>(header.size());

    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);

    const char version[2] = {1, 0};
    out.write(version, 2);

    const char header_len_le[2] = {
      static_cast<char>(header_len & 0xFF),
      static_cast<char>((header_len >> 8) & 0xFF)};
    out.write(header_len_le, 2);

    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(double)));

    DYABLO_ASSERT_HOST_RELEASE(out.good(), "Failed to write movie output file : " << file_path.string());
  }

  void write_projection(Axis axis,
                        const std::string& var_name,
                        int iter,
                        const std::vector<double>& projection,
                        const ProjectionGeometry& geom) const
  {
    std::ostringstream iter_suffix;
    iter_suffix << "_iter" << std::setw(6) << std::setfill('0') << iter;
    std::filesystem::path file_path = m_output_dir / (var_name + "_" + axis_tag(axis) + iter_suffix.str() + ".npy");

    write_npy_2d(file_path, projection, geom.res_b, geom.res_a);
  }

private:
  ForeachCell& m_foreach_cell;

  std::filesystem::path m_output_dir;
  int m_frequency = -1;
  bool m_trigger_on_output = true;
  bool m_output_first_iter = true;
  bool m_enabled = false;

  int m_image_resolution = 800;

  std::vector<Axis> m_axes = {Axis::Z};

  std::vector<std::string> m_requested_variables;

  real_t m_xmin = 0;
  real_t m_xmax = 1;
  real_t m_ymin = 0;
  real_t m_ymax = 1;
  real_t m_zmin = 0;
  real_t m_zmax = 1;

  // Projected region (code units). Width/depth <= 0 mean "full box".
  double m_width = 0;
  double m_depth = 0;
  double m_center_x = 0;
  double m_center_y = 0;
  double m_center_z = 0;

  // Time-based output cadence (simulation time, code units). 0 => disabled.
  real_t m_time_cadence = 0;
  bool m_cadence_initialized = false;
  real_t m_last_cadence_time = 0;

  std::set<Axis> m_warned_unsupported_axes;
  std::set<std::string> m_warned_missing_variables;
};

} // namespace dyablo
