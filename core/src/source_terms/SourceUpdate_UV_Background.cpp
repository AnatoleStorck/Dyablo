#include "SourceUpdate_base.h"
#include "utils/units/Units.h"
#include "utils/misc/Dyablo_assert.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace dyablo {

namespace {

// Physical constants used to convert the tabulated spectrum to photon densities.
constexpr real_t HC_EV_A = 12398.419;       // h*c [eV * Å]
constexpr double H_ERG_S = 6.62607015e-27;  // Planck constant [erg * s]
constexpr double C_CGS   = 2.99792458e10;   // speed of light [cm/s]
// H ionization threshold; same value PRISM uses for the upper edge of the
// G0 (Habing) band.
constexpr double E_H_ION_EV = 13.6;

// Capacity of the per-group array passed by value to the device kernel.
constexpr int MAX_UVB_GROUPS = 8;

// Parse a CUBA/Haardt-Madau UVB output file (e.g. HM12 "UVB.out"):
// '#'-prefixed header lines, then one line with the sampling redshifts,
// then one line per wavelength: λ [Å] followed by J_ν [erg/s/cm²/Hz/sr]
// at each sampling redshift. J_out is row-major [n_wvl][n_z].
void parse_cuba_file(
  const std::string& path,
  std::vector<double>& z_out,
  std::vector<double>& wvls_out,
  std::vector<double>& J_out)
{
  std::ifstream f(path);
  DYABLO_ASSERT_HOST_RELEASE(f.is_open(), "UVB: cannot open " << path);

  z_out.clear(); wvls_out.clear(); J_out.clear();

  std::string line;
  while (std::getline(f, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;

    std::istringstream iss(line);
    std::vector<double> vals;
    double v;
    while (iss >> v) vals.push_back(v);
    if (vals.empty()) continue;

    if (z_out.empty()) {
      // First data line: the sampling redshifts
      z_out = vals;
      DYABLO_ASSERT_HOST_RELEASE(z_out.size() >= 2,
        "UVB: expected at least 2 sampling redshifts in " << path);
      for (size_t i = 1; i < z_out.size(); ++i)
        DYABLO_ASSERT_HOST_RELEASE(z_out[i] > z_out[i-1],
          "UVB: sampling redshifts not increasing in " << path);
    } else {
      DYABLO_ASSERT_HOST_RELEASE(vals.size() == z_out.size() + 1,
        "UVB: expected " << z_out.size() + 1 << " columns, got "
          << vals.size() << " in " << path);
      // CUBA lists spectral discontinuities as duplicate wavelengths,
      // so non-decreasing (not strictly increasing) is the requirement.
      DYABLO_ASSERT_HOST_RELEASE(wvls_out.empty() || vals[0] >= wvls_out.back(),
        "UVB: wavelengths not sorted in " << path);
      wvls_out.push_back(vals[0]);
      J_out.insert(J_out.end(), vals.begin() + 1, vals.end());
    }
  }
  DYABLO_ASSERT_HOST_RELEASE(wvls_out.size() >= 2,
    "UVB: no spectral rows found in " << path);
}

// Trapezoid integral of J(λ)/(h λ) over [lo_A, hi_A] with linear interpolation
// at the bin edges (same scheme as integrate_photon_rate in
// ParticleUpdate_radiative_feedback_BPASS.cpp). λ in Å works directly since
// dλ/λ is scale-free. Multiplied by 4π/c this gives the photon number density
// [photons/cm³] of an isotropic background of intensity J_ν.
// Duplicate wavelengths (CUBA discontinuities) form zero-width intervals that
// contribute nothing; the dw guards keep the edge fragments safe.
double integrate_uvb_photon_density(
  const double* J, const double* wvls, size_t n_wvls,
  double lo_A, double hi_A)
{
  if (hi_A <= wvls[0] || lo_A >= wvls[n_wvls - 1]) return 0.0;
  if (lo_A < wvls[0])          lo_A = wvls[0];
  if (hi_A > wvls[n_wvls - 1]) hi_A = wvls[n_wvls - 1];

  auto pdens = [&](size_t i) { return J[i] / (H_ERG_S * wvls[i]); };

  size_t i_lo = 0;
  while (i_lo < n_wvls && wvls[i_lo] < lo_A) ++i_lo;
  size_t i_hi = n_wvls - 1;
  while (i_hi > 0 && wvls[i_hi] > hi_A) --i_hi;

  // Degenerate case: both edges fall strictly inside one grid interval.
  if (i_lo > i_hi) {
    double dw = wvls[i_lo] - wvls[i_hi];
    if (dw <= 0.0) return 0.0;
    double f_lo = (lo_A - wvls[i_hi]) / dw;
    double f_hi = (hi_A - wvls[i_hi]) / dw;
    double p_lo = (1.0 - f_lo) * pdens(i_hi) + f_lo * pdens(i_lo);
    double p_hi = (1.0 - f_hi) * pdens(i_hi) + f_hi * pdens(i_lo);
    return 0.5 * (p_lo + p_hi) * (hi_A - lo_A);
  }

  double sum = 0.0;

  // Left fragment: (lo_A, wvls[i_lo])
  if (i_lo > 0 && wvls[i_lo] > lo_A) {
    double dw = wvls[i_lo] - wvls[i_lo - 1];
    if (dw > 0.0) {
      double f = (lo_A - wvls[i_lo - 1]) / dw;
      double p_lo = (1.0 - f) * pdens(i_lo - 1) + f * pdens(i_lo);
      sum += 0.5 * (p_lo + pdens(i_lo)) * (wvls[i_lo] - lo_A);
    }
  }

  // Interior trapezoids (zero-width intervals contribute 0)
  for (size_t i = i_lo; i < i_hi; ++i)
    sum += 0.5 * (pdens(i) + pdens(i + 1)) * (wvls[i + 1] - wvls[i]);

  // Right fragment: (wvls[i_hi], hi_A)
  if (i_hi < n_wvls - 1 && wvls[i_hi] < hi_A) {
    double dw = wvls[i_hi + 1] - wvls[i_hi];
    if (dw > 0.0) {
      double f = (hi_A - wvls[i_hi]) / dw;
      double p_hi = (1.0 - f) * pdens(i_hi) + f * pdens(i_hi + 1);
      sum += 0.5 * (pdens(i_hi) + p_hi) * (hi_A - wvls[i_hi]);
    }
  }

  return sum;
}

} // anonymous namespace

/**
 * @brief Inject a (redshift-dependent) UV background into the cells on the
 *        box boundary so the M1 solver advects it inward and photoabsorption
 *        shields it self-consistently.
 *
 * Each step, every cell touching a non-periodic RT boundary face gets its
 * per-group photon density topped up to the background value N_g computed
 * from a tabulated CUBA/Haardt-Madau spectrum:
 *
 *   N_g = uvb_boost × 4π/(c·c_frac) × ∫_group J_ν/(hν) dν
 *
 * The 1/c_frac factor compensates the reduced-speed-of-light photo-rates in
 * PRISM (cs_ph = σ·c·c_frac), so the optically thin ionization/heating rates
 * match the physical UVB. Fields are in physical photons/cm³, matching what
 * CoolingUpdate_PRISM reads. This term must run BEFORE CoolingUpdate_PRISM
 * in [source_terms] updates so the chemistry sees refilled boundary cells.
 */
class SourceUpdate_UV_Background : public SourceUpdate
{
private:
  enum InjectionMode {
    FLOOR,      // e_rad = max(e_rad, N_g), fluxes untouched
    DIRICHLET,  // e_rad = N_g, fluxes = 0 (isotropic bath)
    BEAM        // e_rad = N_g, F = flux_frac·c̃·N_g along the inward normal
  };

  ForeachCell& foreach_cell;
  Timers& timers;

  int n_groups;
  bool cosmo_run;
  real_t fixed_redshift;
  InjectionMode mode;
  real_t c_rad_code;   // reduced speed of light, physical code units (BEAM)
  real_t flux_frac;    // |F| / (c̃·N_g) of the injected beam (BEAM)
  bool initial_fill;
  bool first_call = true;

  real_t xmin, xmax, ymin, ymax, zmin, zmax;
  Kokkos::Array<bool, 3> inject_min, inject_max;

  std::vector<double> z_table;  // sampling redshifts [n_z]
  std::vector<double> N_table;  // photons/cm³, RSLA-corrected [n_z][n_groups]

  Kokkos::Array<real_t, MAX_UVB_GROUPS> interp_Ng(real_t z) const
  {
    Kokkos::Array<real_t, MAX_UVB_GROUPS> Ng {};
    const size_t n_z = z_table.size();
    size_t i0 = 0, i1 = 0;
    double f = 0.0;
    if (z <= z_table.front())
      i0 = i1 = 0;
    else if (z >= z_table.back())
      i0 = i1 = n_z - 1;
    else {
      while (z_table[i0 + 1] < z) ++i0;
      i1 = i0 + 1;
      f = (z - z_table[i0]) / (z_table[i1] - z_table[i0]);
    }
    for (int g = 0; g < n_groups; ++g)
      Ng[g] = (1.0 - f) * N_table[i0 * n_groups + g]
                    + f * N_table[i1 * n_groups + g];
    return Ng;
  }

public:
  SourceUpdate_UV_Background(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    timers(timers),
    n_groups(configMap.getValue<int>("rad", "n_groups", 0)),
    cosmo_run(configMap.getValue<bool>("cosmology", "active", false)),
    fixed_redshift(configMap.getValue<real_t>("rad", "uvb_redshift", 0.0)),
    initial_fill(configMap.getValue<bool>("rad", "uvb_initial_fill", false)),
    xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
    xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
    ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
    ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
    zmin(configMap.getValue<real_t>("mesh", "zmin", 0.0)),
    zmax(configMap.getValue<real_t>("mesh", "zmax", 1.0))
  {
    DYABLO_ASSERT_HOST_RELEASE(n_groups > 0 && n_groups <= MAX_UVB_GROUPS,
      "SourceUpdate_UV_Background: rad/n_groups must be in [1," << MAX_UVB_GROUPS
        << "], got " << n_groups);

    std::string mode_str = configMap.getValue<std::string>(
      "rad", "uvb_injection_mode", "floor");
    if (mode_str == "floor")
      this->mode = FLOOR;
    else if (mode_str == "dirichlet")
      this->mode = DIRICHLET;
    else if (mode_str == "beam")
      this->mode = BEAM;
    else {
      std::ostringstream err_msg;
      err_msg << "Unknown value for rad/uvb_injection_mode : " << mode_str
              << "; Available are : floor, dirichlet, beam" << std::endl;
      throw std::runtime_error(err_msg.str());
    }

    // Beam mode: fraction of the free-streaming flux c̃·N_g to inject.
    // 1 = fully directed beam (M1 reduced flux f=1); 0.5 mimics the net flux
    // of a half-isotropic field entering through the face.
    flux_frac = configMap.getValue<real_t>("rad", "uvb_flux_fraction", 1.0);
    DYABLO_ASSERT_HOST_RELEASE(flux_frac >= 0.0 && flux_frac <= 1.0,
      "SourceUpdate_UV_Background: rad/uvb_flux_fraction must be in [0,1], got "
        << flux_frac);

    // Photon group bin edges (eV), shared with the RT solver and PRISM
    auto bin_lo_eV = configMap.getValue<std::vector<real_t>>(
      "rad", "photon_groups_lower", {});
    auto bin_hi_eV = configMap.getValue<std::vector<real_t>>(
      "rad", "photon_groups_upper", {});
    DYABLO_ASSERT_HOST_RELEASE(
      static_cast<int>(bin_lo_eV.size()) == n_groups &&
      static_cast<int>(bin_hi_eV.size()) == n_groups,
      "SourceUpdate_UV_Background: rad/photon_groups_lower/upper must each "
      "have n_groups (=" << n_groups << ") entries");
    for (int g = 0; g < n_groups; ++g)
      DYABLO_ASSERT_HOST_RELEASE(bin_lo_eV[g] < bin_hi_eV[g],
        "SourceUpdate_UV_Background: rad/photon_groups_lower[" << g
          << "] must be strictly less than rad/photon_groups_upper[" << g << "]");

    // Reduced speed of light fraction, read exactly as CoolingUpdate_PRISM
    // does so both agree.
    auto code_velocity = Units::code_units().getUnit<Units::Velocity>();
    real_t c_rad = configMap.getValue_in_code_unit<Units::Velocity>(
      "rad", "c_rad", "speedoflight");
    this->c_rad_code = c_rad;
    real_t c_frac = (c_rad * code_velocity).convert_to(Units::SPEEDOFLIGHT());
    DYABLO_ASSERT_HOST_RELEASE(c_frac > 0,
      "SourceUpdate_UV_Background: invalid rad/c_rad");

    real_t boost = configMap.getValue<real_t>("rad", "uvb_boost", 1.0);

    // Optionally deposit only the sub-ionizing (< 13.6 eV) part of the
    // spectrum (e.g. turbulent boxes that want the FUV/G0 background without
    // ionizing radiation). Group bands are clipped at the H-ionization
    // threshold; fully ionizing groups get zero injection and are skipped.
    const bool sub_ionizing_only =
      configMap.getValue<bool>("rad", "uvb_sub_ionizing_only", false);

    // Faces to inject through: the M1 solver forces absorbing BCs on every
    // face when rad/absorbing_bc is set (HyperbolicPolicy_Rad), otherwise it
    // follows the mesh boundary types. Never inject through a periodic RT
    // face (photons would double up with their periodic images).
    const bool rad_absorbing =
      configMap.getValue<bool>("rad", "absorbing_bc", false);
    const BoundaryConditionType bc_min[3] = {
      configMap.getValue<BoundaryConditionType>("mesh", "boundary_type_xmin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh", "boundary_type_ymin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh", "boundary_type_zmin", BC_ABSORBING)};
    const BoundaryConditionType bc_max[3] = {
      configMap.getValue<BoundaryConditionType>("mesh", "boundary_type_xmax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh", "boundary_type_ymax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh", "boundary_type_zmax", BC_ABSORBING)};
    bool any_face = false;
    for (int d = 0; d < 3; ++d) {
      inject_min[d] = rad_absorbing || (bc_min[d] != BC_PERIODIC);
      inject_max[d] = rad_absorbing || (bc_max[d] != BC_PERIODIC);
      any_face = any_face || inject_min[d] || inject_max[d];
    }
    if (!any_face)
      printf("WARNING: SourceUpdate_UV_Background: all RT boundaries are "
             "periodic, no UVB will be injected\n");

    // Load the spectrum and integrate each group band at each redshift
    std::string uvb_file = configMap.getValue<std::string>(
      "rad", "uvb_spectrum_file");
    std::vector<double> wvls, Jtab;
    parse_cuba_file(uvb_file, z_table, wvls, Jtab);
    const size_t n_z   = z_table.size();
    const size_t n_wvl = wvls.size();

    // Effective per-group integration bands (clipped when sub-ionizing only)
    std::vector<double> E_lo_eff(n_groups), E_hi_eff(n_groups);
    for (int g = 0; g < n_groups; ++g) {
      E_lo_eff[g] = bin_lo_eV[g];
      E_hi_eff[g] = sub_ionizing_only
        ? std::min<double>(bin_hi_eV[g], E_H_ION_EV)
        : (double)bin_hi_eV[g];
    }

    N_table.assign(n_z * n_groups, 0.0);
    std::vector<double> Jcol(n_wvl);
    const double prefactor = boost * 4.0 * M_PI / (C_CGS * c_frac);
    for (size_t iz = 0; iz < n_z; ++iz) {
      for (size_t i = 0; i < n_wvl; ++i)
        Jcol[i] = Jtab[i * n_z + iz];
      for (int g = 0; g < n_groups; ++g) {
        if (E_lo_eff[g] >= E_hi_eff[g]) continue; // fully ionizing group: stays 0
        const double lo_A = HC_EV_A / E_hi_eff[g];
        const double hi_A = HC_EV_A / E_lo_eff[g];
        N_table[iz * n_groups + g] = prefactor * integrate_uvb_photon_density(
          Jcol.data(), wvls.data(), n_wvl, lo_A, hi_A);
      }
    }

    const real_t z_print = cosmo_run ? z_table.front() : fixed_redshift;
    auto Ng = interp_Ng(z_print);
    printf("SourceUpdate_UV_Background: %s (%zu wavelengths x %zu redshifts, "
           "z=%.2f..%.2f)\n", uvb_file.c_str(), n_wvl, n_z,
           z_table.front(), z_table.back());
    if (sub_ionizing_only)
      printf("SourceUpdate_UV_Background: sub-ionizing only, group bands "
             "clipped at %.1f eV\n", E_H_ION_EV);
    printf("SourceUpdate_UV_Background: injected densities at z=%.3f "
           "(incl. RSLA boost 1/c_frac=%.4g, uvb_boost=%.4g):\n",
           z_print, 1.0 / c_frac, boost);
    for (int g = 0; g < n_groups; ++g)
      printf("  group %d [%7.2f, %7.2f] eV : N = %.6e photons/cm^3\n",
             g, bin_lo_eV[g], bin_hi_eV[g], Ng[g]);
  }

  void update( UserData &U, ScalarSimulationData& scalar_data)
  {
    timers.get("SourceUpdate_UV_Background").start();

    const real_t z = cosmo_run
      ? 1.0 / scalar_data.get<real_t>("aexp") - 1.0
      : fixed_redshift;
    const Kokkos::Array<real_t, MAX_UVB_GROUPS> Ng = interp_Ng(z);

    // Stride-4 accessor over the freshly-advected radiation fields
    // (same layout CoolingUpdate_PRISM uses)
    std::vector<UserData::FieldAccessor_FieldInfo> rt_fields;
    for (int g = 0; g < n_groups; ++g) {
      const int off = 4 * g;
      rt_fields.push_back({"e_rad_"  + std::to_string(g) + "_next", off + 0});
      rt_fields.push_back({"fx_rad_" + std::to_string(g) + "_next", off + 1});
      rt_fields.push_back({"fy_rad_" + std::to_string(g) + "_next", off + 2});
      rt_fields.push_back({"fz_rad_" + std::to_string(g) + "_next", off + 3});
    }
    UserData::FieldAccessor Uout = U.getAccessor(rt_fields);

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    // Reduced speed of light in supercomoving code units: the M1 closure
    // normalizes the flux as f = |F|/(c̃·e_rad), so the free-streaming beam
    // is F = c̃·N_g regardless of the (physical cgs) units of e_rad.
    const real_t ctilde = Units::physical_to_supercomoving<Units::Velocity>(
      c_rad_code, scalar_data.get<real_t>("aexp"));

    // Local copies for KOKKOS_LAMBDA capture
    const int ndim = foreach_cell.getDim();
    const bool do_fill = initial_fill && first_call;
    first_call = false;
    const int n_groups = this->n_groups;
    const InjectionMode mode = this->mode;
    const real_t flux_frac = this->flux_frac;
    const real_t xmin = this->xmin, xmax = this->xmax;
    const real_t ymin = this->ymin, ymax = this->ymax;
    const real_t zmin = this->zmin, zmax = this->zmax;
    const auto inject_min = this->inject_min;
    const auto inject_max = this->inject_max;

    foreach_cell.foreach_cell( "SourceUpdate_UV_Background", Uout.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
      auto pos  = cells.getCellCenter(iCell);
      auto size = cells.getCellSize(iCell);

      // Cells whose center is within one own-size of a face are exactly the
      // outermost leaf layer, at any AMR level. Each injecting face the cell
      // touches contributes its inward normal; edge/corner cells sum them.
      real_t nrm[3] = {0.0, 0.0, 0.0};
      bool bdy = false;
      if (inject_min[IX] && pos[IX] < xmin + size[IX]) { nrm[IX] += 1.0; bdy = true; }
      if (inject_max[IX] && pos[IX] > xmax - size[IX]) { nrm[IX] -= 1.0; bdy = true; }
      if (inject_min[IY] && pos[IY] < ymin + size[IY]) { nrm[IY] += 1.0; bdy = true; }
      if (inject_max[IY] && pos[IY] > ymax - size[IY]) { nrm[IY] -= 1.0; bdy = true; }
      if (ndim == 3) {
        if (inject_min[IZ] && pos[IZ] < zmin + size[IZ]) { nrm[IZ] += 1.0; bdy = true; }
        if (inject_max[IZ] && pos[IZ] > zmax - size[IZ]) { nrm[IZ] -= 1.0; bdy = true; }
      }

      if (!bdy && !do_fill) return;

      // Unit inward normal for the beam (zero if opposite faces cancel,
      // e.g. a single-cell-thick box: fall back to zero net flux).
      const real_t nrm2 = nrm[IX]*nrm[IX] + nrm[IY]*nrm[IY] + nrm[IZ]*nrm[IZ];
      if (nrm2 > 0.0) {
        const real_t inv = 1.0 / Kokkos::sqrt(nrm2);
        nrm[IX] *= inv; nrm[IY] *= inv; nrm[IZ] *= inv;
      }

      for (int g = 0; g < n_groups; ++g) {
        const int off = 4 * g;
        // No background in this group (e.g. ionizing groups with
        // uvb_sub_ionizing_only): leave the field completely untouched so
        // even Dirichlet mode cannot erase locally-produced radiation.
        if (Ng[g] <= 0.0) continue;
        if (bdy && mode == DIRICHLET) {
          // Isotropic bath: pin the density, zero net flux
          Uout.at(iCell, off + 0) = Ng[g];
          Uout.at(iCell, off + 1) = 0.0;
          Uout.at(iCell, off + 2) = 0.0;
          Uout.at(iCell, off + 3) = 0.0;
        } else if (bdy && mode == BEAM) {
          // Directed beam: pin the density and point the flux into the box,
          // perpendicular to the boundary face, so the M1 solver advects the
          // background inward instead of spreading it isotropically.
          const real_t F = flux_frac * ctilde * Ng[g];
          Uout.at(iCell, off + 0) = Ng[g];
          Uout.at(iCell, off + 1) = F * nrm[IX];
          Uout.at(iCell, off + 2) = F * nrm[IY];
          Uout.at(iCell, off + 3) = F * nrm[IZ];
        } else {
          // Floor: only top up, never remove locally-produced photons.
          // Fluxes untouched: raising e_rad at fixed flux lowers |F|/(c e_rad),
          // which keeps the M1 closure valid.
          if (Uout.at(iCell, off + 0) < Ng[g])
            Uout.at(iCell, off + 0) = Ng[g];
        }
      }
    });

    timers.get("SourceUpdate_UV_Background").stop();
  }
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::SourceUpdate_UV_Background,
                  "SourceUpdate_UV_Background" );
