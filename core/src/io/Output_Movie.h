#pragma once

#include <algorithm>
#include <cctype>
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

namespace dyablo {

class Output_Movie
{
public:
  Output_Movie(ConfigMap& configMap, ForeachCell& foreach_cell)
    : m_foreach_cell(foreach_cell),
      m_output_dir(std::filesystem::path(configMap.getValue<std::string>("output", "outputDir", ".")) / "movie_files"),
      m_frequency(configMap.getValue<int>("movie", "output_frequency", -1)),
      m_trigger_on_output(configMap.getValue<bool>("movie", "trigger_on_output", true)),
      m_output_first_iter(configMap.getValue<bool>("movie", "output_first_iter", true)),
      m_xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
      m_xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
      m_ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
      m_ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
      m_zmin(configMap.getValue<real_t>("mesh", "zmin", 0.0)),
      m_zmax(configMap.getValue<real_t>("mesh", "zmax", 1.0))
  {

    m_enabled = configMap.getValue<bool>("movie", "enabled", false);

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

    return first_iter || frequency_trigger || output_trigger;
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

    int iter = scalar_data.get<int>("iter");

    for( const std::string& var_name : output_variables )
    {
      auto field = U.getField(var_name);
      auto field_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.U);

      for( const AxisPlan& axis_plan : valid_axis_plans )
      {
        Axis axis = axis_plan.axis;
        const MeshInfo& mesh_info = axis_plan.mesh_info;
        const double dl = axis_plan.dl;

        std::vector<double> projection_local(mesh_info.projection_size, 0.0);

        for( uint32_t iOct = 0; iOct < nbOcts_local; ++iOct )
        {
          auto logical = storage.get_logical_coords({iOct, false});
          uint64_t oct_x = logical[IX];
          uint64_t oct_y = logical[IY];
          uint64_t oct_z = logical[IZ];

          for( uint32_t k = 0; k < bz; ++k )
          {
            for( uint32_t j = 0; j < by; ++j )
            {
              for( uint32_t i = 0; i < bx; ++i )
              {
                uint64_t gx = oct_x * bx + i;
                uint64_t gy = oct_y * by + j;
                uint64_t gz = oct_z * bz + k;

                uint32_t iCell = i + bx * (j + by * k);
                double value = static_cast<double>(field_host(iCell, 0, iOct));

                size_t idx = projection_index(axis, gx, gy, gz, mesh_info);
                projection_local[idx] += value * dl;
              }
            }
          }
        }

        std::vector<double> projection_global(mesh_info.projection_size, 0.0);
        mpi_comm.MPI_Allreduce(
          projection_local.data(),
          projection_global.data(),
          static_cast<int>(mesh_info.projection_size),
          MpiComm::MPI_Op_t::SUM);

        if( mpi_rank == 0 )
        {
          write_projection(axis, var_name, iter, projection_global, mesh_info);
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

  struct MeshInfo
  {
    uint64_t nx = 0;
    uint64_t ny = 0;
    uint64_t nz = 0;

    size_t out_n0 = 0;
    size_t out_n1 = 0;
    size_t projection_size = 0;
  };

  struct AxisPlan
  {
    Axis axis = Axis::Z;
    MeshInfo mesh_info;
    double dl = 0;
  };

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

  bool compute_mesh_info(Axis axis, MeshInfo& info, std::string& reason)
  {
    AMRmesh& amr_mesh = m_foreach_cell.get_amr_mesh();
    MpiComm mpi_comm = amr_mesh.getMpiComm();
    auto storage = amr_mesh.getStorage();

    const uint32_t nbOcts_local = amr_mesh.getNumOctants();

    int local_level_min = std::numeric_limits<int>::max();
    int local_level_max = std::numeric_limits<int>::min();

    for( uint32_t iOct = 0; iOct < nbOcts_local; ++iOct )
    {
      int level = storage.getLevel({iOct, false});
      local_level_min = std::min(local_level_min, level);
      local_level_max = std::max(local_level_max, level);
    }

    if( nbOcts_local == 0 )
    {
      local_level_min = std::numeric_limits<int>::max() / 2;
      local_level_max = std::numeric_limits<int>::min() / 2;
    }

    int global_level_min = 0;
    int global_level_max = 0;
    mpi_comm.MPI_Allreduce(&local_level_min, &global_level_min, 1, MpiComm::MPI_Op_t::MIN);
    mpi_comm.MPI_Allreduce(&local_level_max, &global_level_max, 1, MpiComm::MPI_Op_t::MAX);

    if( global_level_min != global_level_max )
    {
      reason = "mesh is not uniform in refinement level (AMR detected)";
      return false;
    }

    int level_min = amr_mesh.get_level_min();
    int shift = global_level_min - level_min;
    if( shift < 0 )
    {
      reason = "internal level mismatch in AMR mesh";
      return false;
    }
    if( shift >= 63 )
    {
      reason = "grid level is too large for movie projection indexing";
      return false;
    }

    auto coarse = amr_mesh.get_coarse_grid_size();

    uint64_t oct_nx = static_cast<uint64_t>(coarse[IX]) << shift;
    uint64_t oct_ny = static_cast<uint64_t>(coarse[IY]) << shift;
    uint64_t oct_nz = static_cast<uint64_t>(coarse[IZ]) << shift;

    uint64_t expected_global_octs = oct_nx * oct_ny * oct_nz;
    if( expected_global_octs != amr_mesh.getGlobalNumOctants() )
    {
      reason = "mesh octant count does not match a full uniform grid";
      return false;
    }

    const uint32_t bx = m_foreach_cell.blockSize()[IX];
    const uint32_t by = m_foreach_cell.blockSize()[IY];
    const uint32_t bz = m_foreach_cell.blockSize()[IZ];

    info.nx = oct_nx * bx;
    info.ny = oct_ny * by;
    info.nz = oct_nz * bz;

    if( info.nx == 0 || info.ny == 0 || info.nz == 0 )
    {
      reason = "empty grid dimensions";
      return false;
    }

    switch( axis )
    {
    case Axis::X:
      info.out_n0 = static_cast<size_t>(info.nz);
      info.out_n1 = static_cast<size_t>(info.ny);
      break;
    case Axis::Y:
      info.out_n0 = static_cast<size_t>(info.nz);
      info.out_n1 = static_cast<size_t>(info.nx);
      break;
    case Axis::Z:
      info.out_n0 = static_cast<size_t>(info.ny);
      info.out_n1 = static_cast<size_t>(info.nx);
      break;
    }

    if( info.out_n1 != 0 && info.out_n0 > std::numeric_limits<size_t>::max() / info.out_n1 )
    {
      reason = "projection dimensions overflow";
      return false;
    }

    info.projection_size = info.out_n0 * info.out_n1;

    if( info.projection_size > static_cast<size_t>(std::numeric_limits<int>::max()) )
    {
      reason = "projection array is too large for MPI_Allreduce count";
      return false;
    }

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
      MeshInfo mesh_info;
      std::string unsupported_reason;
      if( !compute_mesh_info(axis, mesh_info, unsupported_reason) )
      {
        if( mpi_rank == 0 && m_warned_unsupported_axes.insert(axis).second )
        {
          std::cout << "WARNING : movie output skipped for axis '" << axis_tag(axis)
                    << "' - " << unsupported_reason << std::endl;
        }
        continue;
      }

      axis_plans.push_back({axis, mesh_info, line_element(axis, mesh_info)});
    }

    return axis_plans;
  }

  double line_element(Axis axis, const MeshInfo& info) const
  {
    const double dx = (m_xmax - m_xmin) / static_cast<double>(info.nx);
    const double dy = (m_ymax - m_ymin) / static_cast<double>(info.ny);
    const double dz = (m_zmax - m_zmin) / static_cast<double>(info.nz);

    switch( axis )
    {
    case Axis::X:
      return dx;
    case Axis::Y:
      return dy;
    case Axis::Z:
      return dz;
    }

    return dz;
  }

  size_t projection_index(Axis axis,
                          uint64_t gx,
                          uint64_t gy,
                          uint64_t gz,
                          const MeshInfo& info) const
  {
    switch( axis )
    {
    case Axis::X:
      return static_cast<size_t>(gy + info.ny * gz);
    case Axis::Y:
      return static_cast<size_t>(gx + info.nx * gz);
    case Axis::Z:
      return static_cast<size_t>(gx + info.nx * gy);
    }

    return 0;
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
                        const MeshInfo& info) const
  {
    std::ostringstream iter_suffix;
    iter_suffix << "_iter" << std::setw(6) << std::setfill('0') << iter;
    std::filesystem::path file_path = m_output_dir / (var_name + "_" + axis_tag(axis) + iter_suffix.str() + ".npy");

    write_npy_2d(file_path, projection, info.out_n0, info.out_n1);
  }

private:
  ForeachCell& m_foreach_cell;

  std::filesystem::path m_output_dir;
  int m_frequency = -1;
  bool m_trigger_on_output = true;
  bool m_output_first_iter = true;
  bool m_enabled = false;

  std::vector<Axis> m_axes = {Axis::Z};

  std::vector<std::string> m_requested_variables;

  real_t m_xmin = 0;
  real_t m_xmax = 1;
  real_t m_ymin = 0;
  real_t m_ymax = 1;
  real_t m_zmin = 0;
  real_t m_zmax = 1;

  std::set<Axis> m_warned_unsupported_axes;
  std::set<std::string> m_warned_missing_variables;
};

} // namespace dyablo
