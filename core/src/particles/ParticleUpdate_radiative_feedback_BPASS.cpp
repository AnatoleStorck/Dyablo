#include "ParticleUpdate_base.h"
#include "utils/units/Units.h"
#include "ForeachParticle.h"
#include "states/State_hydro.h"
#include "utils/misc/Dyablo_assert.h"

#include <Kokkos_Core.hpp>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <cstring>
// #include <cstdint>
// #include <sstream>

namespace dyablo {

namespace {

// Physical constants used to convert spectra to photon rates.
constexpr real_t HC_EV_A   = 12398.419;        // h*c [eV * Å]
constexpr real_t HC_ERG_A  = 1.9864458e-8;     // h*c [erg * Å]

// Energy range covered by the reduced BPASS spectra produced by
// BPASS_reduction.py. Bin edges given in the .ini file must lie inside
// this range.
constexpr real_t BPASS_E_MIN_EV = 5.6;
constexpr real_t BPASS_E_MAX_EV = 500.0;

// Read a 2D float64, C-order NumPy .npy file.
// Returns the flattened data (row-major) and writes the shape via out params.
std::vector<double> read_npy_2d_f8(
  const std::string& path, size_t& n0_out, size_t& n1_out)
{
  std::ifstream f(path, std::ios::binary);
  DYABLO_ASSERT_HOST_RELEASE(f.is_open(), "BPASS: cannot open " << path);

  char magic[6];
  f.read(magic, 6);
  DYABLO_ASSERT_HOST_RELEASE(
    std::memcmp(magic, "\x93NUMPY", 6) == 0,
    "BPASS: not a .npy file: " << path);

  uint8_t v_major = 0, v_minor = 0;
  f.read(reinterpret_cast<char*>(&v_major), 1);
  f.read(reinterpret_cast<char*>(&v_minor), 1);

  uint32_t header_len = 0;
  if (v_major == 1) {
    uint16_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 2);
    header_len = hl;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  DYABLO_ASSERT_HOST_RELEASE(
    header.find("'descr': '<f8'") != std::string::npos,
    "BPASS: expected float64 little-endian dtype in " << path);
  DYABLO_ASSERT_HOST_RELEASE(
    header.find("'fortran_order': False") != std::string::npos,
    "BPASS: expected fortran_order=False in " << path);

  // Extract 'shape': (a, b)
  auto sp = header.find("'shape':");
  DYABLO_ASSERT_HOST_RELEASE(sp != std::string::npos,
    "BPASS: missing shape in " << path);
  auto open_p  = header.find('(', sp);
  auto close_p = header.find(')', open_p);
  std::string shape_str = header.substr(open_p + 1, close_p - open_p - 1);

  std::vector<size_t> shape;
  std::stringstream ss(shape_str);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    auto a = tok.find_first_not_of(" \t");
    auto b = tok.find_last_not_of(" \t");
    if (a == std::string::npos) continue;
    tok = tok.substr(a, b - a + 1);
    if (tok.empty()) continue;
    shape.push_back(std::stoull(tok));
  }
  DYABLO_ASSERT_HOST_RELEASE(shape.size() == 2,
    "BPASS: expected 2D array in " << path);

  n0_out = shape[0];
  n1_out = shape[1];
  std::vector<double> data(n0_out * n1_out);
  f.read(reinterpret_cast<char*>(data.data()),
         static_cast<std::streamsize>(data.size() * sizeof(double)));
  DYABLO_ASSERT_HOST_RELEASE(static_cast<bool>(f),
    "BPASS: truncated data in " << path);
  return data;
}

// Trapezoid integral of (spectrum * λ / hc) over [lo_A, hi_A], with linear
// interpolation at the bin edges to handle edges falling between grid
// points. Returns photons/s for the (per-Msun) input spectrum.
double integrate_photon_rate(
  const double* spec, const double* wvls, size_t n_wvls,
  double lo_A, double hi_A)
{
  if (hi_A <= wvls[0] || lo_A >= wvls[n_wvls - 1]) return 0.0;
  if (lo_A < wvls[0])          lo_A = wvls[0];
  if (hi_A > wvls[n_wvls - 1]) hi_A = wvls[n_wvls - 1];

  auto pdens = [&](size_t i) { return spec[i] * wvls[i] / HC_ERG_A; };

  size_t i_lo = 0;
  while (i_lo < n_wvls && wvls[i_lo] < lo_A) ++i_lo;
  size_t i_hi = n_wvls - 1;
  while (i_hi > 0 && wvls[i_hi] > hi_A) --i_hi;

  // Degenerate case: both edges fall strictly inside one grid interval
  // (i_hi == i_lo - 1, so there is no grid point between them). Integrate the
  // single sub-interval [lo_A, hi_A] directly to avoid the left/right fragments
  // overlapping and double-counting.
  if (i_lo > i_hi) {
    double dw = wvls[i_lo] - wvls[i_hi];
    double f_lo = (lo_A - wvls[i_hi]) / dw;
    double f_hi = (hi_A - wvls[i_hi]) / dw;
    double p_lo = (1.0 - f_lo) * pdens(i_hi) + f_lo * pdens(i_lo);
    double p_hi = (1.0 - f_hi) * pdens(i_hi) + f_hi * pdens(i_lo);
    return 0.5 * (p_lo + p_hi) * (hi_A - lo_A);
  }

  double sum = 0.0;

  // Left fragment: (lo_A, wvls[i_lo])
  if (i_lo > 0 && wvls[i_lo] > lo_A) {
    double f = (lo_A - wvls[i_lo - 1]) / (wvls[i_lo] - wvls[i_lo - 1]);
    double p_lo = (1.0 - f) * pdens(i_lo - 1) + f * pdens(i_lo);
    sum += 0.5 * (p_lo + pdens(i_lo)) * (wvls[i_lo] - lo_A);
  }

  // Interior trapezoids
  for (size_t i = i_lo; i < i_hi; ++i)
    sum += 0.5 * (pdens(i) + pdens(i + 1)) * (wvls[i + 1] - wvls[i]);

  // Right fragment: (wvls[i_hi], hi_A)
  if (i_hi < n_wvls - 1 && wvls[i_hi] < hi_A) {
    double f = (hi_A - wvls[i_hi]) / (wvls[i_hi + 1] - wvls[i_hi]);
    double p_hi = (1.0 - f) * pdens(i_hi) + f * pdens(i_hi + 1);
    sum += 0.5 * (pdens(i_hi) + p_hi) * (hi_A - wvls[i_hi]);
  }

  return sum;
}

} // anonymous namespace

class ParticleUpdate_radiative_feedback_BPASS : public ParticleUpdate {
public:
  using pos_t = Kokkos::Array<real_t, 3>;

  // BPASS lookup-table dimensions (must match BPASS_reduction.py output).
  static constexpr int N_METALS = 13;
  static constexpr int N_AGES   = 51;
  // Age grid is log-uniform: log10(age/yr) = LOG_AGE_MIN + LOG_AGE_STEP * i
  static constexpr real_t LOG_AGE_MIN  = 6.0;
  static constexpr real_t LOG_AGE_STEP = 0.1;
  // Wavelength grid of the reduced spectra (1 Å spacing, 24..2214 Å).
  static constexpr int N_WVLS = 2191;

  ParticleUpdate_radiative_feedback_BPASS(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers)
  : foreach_cell    ( foreach_cell ),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    timers          ( timers ),
    n_groups        ( configMap.getValue<int>("rad", "n_groups", 0) ),
    bpass_data_path ( configMap.getValue<std::string>(
                        "star_feedback", "bpass_data_path",
                        "/data/anatole/BPASS") ),
    star_family     ( configMap.getValue<std::string>("star_feedback", "star_family", "star") ),
    temp_metallicity( configMap.getValue<real_t>("star_feedback", "temp_metallicity", 1e-4) ),
    cosmology       ( configMap.getValue<bool>("cosmology", "active", false) ),
    photon_rates    ( "BPASS_photon_rates",
                      N_METALS, N_AGES, n_groups > 0 ? n_groups : 1 )
  {
    DYABLO_ASSERT_HOST_RELEASE(n_groups > 0,
      "ParticleUpdate_radiative_feedback_BPASS: rad.n_groups must be > 0");

    // Read user-defined RT bin edges (in eV) from the .ini file
    auto bin_lo_eV = configMap.getValue<std::vector<real_t>>(
      "rad", "photon_groups_lower", {});
    auto bin_hi_eV = configMap.getValue<std::vector<real_t>>(
      "rad", "photon_groups_upper", {});

    DYABLO_ASSERT_HOST_RELEASE(
      static_cast<int>(bin_lo_eV.size()) == n_groups &&
      static_cast<int>(bin_hi_eV.size()) == n_groups,
      "ParticleUpdate_radiative_feedback_BPASS: rad.photon_groups_lower/upper "
      "must each have n_groups (=" << n_groups << ") entries");

    for (int g = 0; g < n_groups; ++g) {
      DYABLO_ASSERT_HOST_RELEASE(
        bin_lo_eV[g] >= BPASS_E_MIN_EV,
        "ParticleUpdate_radiative_feedback_BPASS: rad.photon_groups_lower[" << g
          << "]=" << bin_lo_eV[g] << " eV is below the BPASS lower bound of "
          << BPASS_E_MIN_EV << " eV");
      DYABLO_ASSERT_HOST_RELEASE(
        bin_hi_eV[g] <= BPASS_E_MAX_EV,
        "ParticleUpdate_radiative_feedback_BPASS: rad.photon_groups_upper[" << g
          << "]=" << bin_hi_eV[g] << " eV is above the BPASS upper bound of "
          << BPASS_E_MAX_EV << " eV");
      DYABLO_ASSERT_HOST_RELEASE(
        bin_lo_eV[g] < bin_hi_eV[g],
        "ParticleUpdate_radiative_feedback_BPASS: rad.photon_groups_lower[" << g
          << "] must be strictly less than rad.photon_groups_upper[" << g << "]");
    }

    // Convert each (E_lo, E_hi) bin in eV to (λ_lo, λ_hi) in Å
    // (high E ↔ short λ, so the upper energy gives the lower wavelength).
    std::vector<double> bin_lo_A(n_groups), bin_hi_A(n_groups);
    for (int g = 0; g < n_groups; ++g) {
      bin_lo_A[g] = HC_EV_A / bin_hi_eV[g];
      bin_hi_A[g] = HC_EV_A / bin_lo_eV[g];
    }

    // BPASS metallicity tags (must match the file naming used by BPASS_reduction.py).
    static const char* METAL_NAMES[N_METALS] = {
      "zem5", "zem4", "z001", "z002", "z003", "z004",
      "z006", "z008", "z010", "z014", "z020", "z030", "z040"
    };

    // Reduced wavelength grid: integer Å from 24 to 2214 inclusive
    std::vector<double> wvls(N_WVLS);
    const int wvl_lo = static_cast<int>(HC_EV_A / BPASS_E_MAX_EV); // 24
    for (int i = 0; i < N_WVLS; ++i) wvls[i] = static_cast<double>(wvl_lo + i);

    // Load each metallicity's spectrum and integrate into the user's bins
    auto h_rates = Kokkos::create_mirror_view(photon_rates);
    std::vector<double> spec(N_WVLS);
    for (int im = 0; im < N_METALS; ++im) {
      const std::string fname = bpass_data_path
        + "/reduced_spectra-bin-imf_chab300."
        + METAL_NAMES[im]
        + ".5.6to500eV.dat.npy";
      size_t n_wvls = 0, n_ages = 0;
      auto data = read_npy_2d_f8(fname, n_wvls, n_ages);
      DYABLO_ASSERT_HOST_RELEASE(
        n_wvls == static_cast<size_t>(N_WVLS) &&
        n_ages == static_cast<size_t>(N_AGES),
        "ParticleUpdate_radiative_feedback_BPASS: unexpected shape ("
          << n_wvls << "," << n_ages << ") in " << fname
          << ", expected (" << N_WVLS << "," << N_AGES << ")");

      // data is row-major: data[i_wvl * N_AGES + i_age]
      for (int ia = 0; ia < N_AGES; ++ia) {
        for (int i = 0; i < N_WVLS; ++i)
          spec[i] = data[i * N_AGES + ia];
        for (int g = 0; g < n_groups; ++g) {
          h_rates(im, ia, g) = integrate_photon_rate(
            spec.data(), wvls.data(), N_WVLS,
            bin_lo_A[g], bin_hi_A[g]);
        }
      }
    }
    Kokkos::deep_copy(photon_rates, h_rates);
  }

  ~ParticleUpdate_radiative_feedback_BPASS() {}

  void update(UserData& U, ScalarSimulationData& scalar_data)
  {

    const real_t aexp = scalar_data.get<real_t>("aexp");

    const real_t code2cm3 = Units::supercomoving_to_physical<Units::Volume>(
      (1 * Units::code_units().getUnit<Units::Volume>()).convert_to(Units::cm3()),
      aexp
    );
    const auto code_time = Units::code_units().getUnit<Units::Time>();
    const auto code_mass = Units::code_units().getUnit<Units::Mass>();

    const real_t dt   = scalar_data.get<real_t>("dt");

    const real_t t_now_code = cosmology ? scalar_data.get<real_t>("time_physical")
                                        : scalar_data.get<real_t>("time");
    const real_t t_phys_yr = (t_now_code * code_time).convert_to(Units::yr());

    enum VarIndex_particle {
      IBIRTHMASS, IBIRTH, IMETAL,
    };

    timers.get("ParticleUpdate_radiative_feedback_BPASS").start();

    if( !U.has_ParticleArray( star_family ) )
    {
      timers.get("ParticleUpdate_radiative_feedback_BPASS").stop();
      return;
    }

    std::vector<UserData::FieldAccessor_FieldInfo> Uout_infos;
    for (int g = 0; g < n_groups; ++g)
      Uout_infos.push_back({"e_rad_" + std::to_string(g), g});

    // DYABLO_ASSERT_HOST_RELEASE(
    //   U.has_ParticleAttribute(star_family, "metallicity"),
    //   "ParticleUpdate_radiative_feedback_BPASS requires a 'metallicity' particle attribute"
    // );
    std::vector<UserData::ParticleAccessor_AttributeInfo> pinfos = {
      {"birth_mass",  IBIRTHMASS},
      {"birth_time",  IBIRTH},
      //{"metallicity", IMETAL},
    };

    auto Ppos  = U.getParticleArray(star_family);
    auto Pdata = U.getParticleAccessor(star_family, pinfos);
    auto Uout  = U.getAccessor(Uout_infos);

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    const real_t dt_phys_s = Units::supercomoving_to_physical<Units::Time>(
      (dt * code_time).convert_to(Units::s()),
      aexp
    );

    // Local copies for KOKKOS_LAMBDA capture (View is ref-counted, cheap).
    auto photon_rates_l = this->photon_rates;
    const real_t temp_metallicity = this->temp_metallicity;
    const int n_groups_l = this->n_groups;

    foreach_particle.foreach_particle(
      "particles_update_radiative_feedback_BPASS", Ppos,
      KOKKOS_LAMBDA(const ForeachParticle::ParticleIndex& iPart)
    {
      // BPASS metallicity grid (mass fraction)
      const real_t metal_grid[N_METALS] = {
        1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3,
        6e-3, 8e-3, 1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2
      };

      // Locate host cell
      pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
      ForeachCell::CellIndex iCell = cells.getCellFromPos(part_pos);

      pos_t cell_size = cells.getCellSize(iCell);
      const real_t cell_volume_physical =
        cell_size[IX] * cell_size[IY] * cell_size[IZ] * code2cm3;

      // Particle age in physical years. birth_time is already a physical code
      // time (see comment above), so this is a plain difference.
      const real_t birth_time_phys_yr =
        (Pdata.at(iPart, IBIRTH) * code_time).convert_to(Units::yr());
      const real_t age_phys_yr = t_phys_yr - birth_time_phys_yr;

      const real_t Mstar_phys_Msun =
        (Pdata.at(iPart, IBIRTHMASS) * code_mass).convert_to(Units::solar_mass());

      
      //const real_t Z = Pdata.at(iPart, IMETAL);
      const real_t Z = temp_metallicity; // Temporary: ignore particle metallicity and use a fixed value

      // --- Bilinear interpolation in (Z, log10 age), with clipping ---
      const real_t Z_clip = FMIN(FMAX(Z, metal_grid[0]), metal_grid[N_METALS - 1]);
      const real_t log_age = (age_phys_yr > 0) ? log10(age_phys_yr) : LOG_AGE_MIN;
      const real_t log_age_clip = FMIN(
        FMAX(log_age, LOG_AGE_MIN),
        LOG_AGE_MIN + LOG_AGE_STEP * (N_AGES - 1)
      );

      // Metallicity bracket (linear search over 13 values)
      int im = 0;
      while (im < N_METALS - 2 && metal_grid[im + 1] < Z_clip) ++im;
      const real_t fz = (Z_clip - metal_grid[im])
                      / (metal_grid[im + 1] - metal_grid[im]);

      // Age bracket (uniform in log10)
      const real_t age_idx = (log_age_clip - LOG_AGE_MIN) / LOG_AGE_STEP;
      int ia = static_cast<int>(age_idx);
      if (ia >= N_AGES - 1) ia = N_AGES - 2;
      const real_t fa = age_idx - ia;

      for (int g = 0; g < n_groups_l; ++g) {
        const real_t r00 = photon_rates_l(im,     ia,     g);
        const real_t r01 = photon_rates_l(im,     ia + 1, g);
        const real_t r10 = photon_rates_l(im + 1, ia,     g);
        const real_t r11 = photon_rates_l(im + 1, ia + 1, g);
        const real_t rate_per_Msun =
          (1 - fz) * ((1 - fa) * r00 + fa * r01) +
                fz  * ((1 - fa) * r10 + fa * r11);

        const real_t photons_per_s = rate_per_Msun * Mstar_phys_Msun;

        // Atomic: multiple particles can land in the same cell
        Kokkos::atomic_add(
          &Uout.at(iCell, g),
          photons_per_s * dt_phys_s / cell_volume_physical
        );
      }
    });

    Kokkos::fence();
    timers.get("ParticleUpdate_radiative_feedback_BPASS").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;

  int n_groups;
  std::string bpass_data_path;
  std::string star_family;
  real_t temp_metallicity;
  bool cosmology;

  Kokkos::View<real_t***> photon_rates;  // [N_METALS][N_AGES][n_groups]
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_radiative_feedback_BPASS,
                  "ParticleUpdate_radiative_feedback_BPASS")
