#include "PassiveScalar_IC_base.h"

#include <fstream>
#include <sstream>
#include <set>
#include <map>

// Sets, on an arbitrary density field, evenly (mass) distributed number densities for a list of metals given a total metallicity.

namespace dyablo{

namespace {
  // Parse "xH_I" -> element="H", state=0 (Roman I = neutral = first data column)
  // Also accepts "H_I" without leading 'x'.
  //
  // Special case: molecular hydrogen "xH2" (or "H2") carries no Roman-numeral
  // ionization state. It is stored as the third column (index 2) of the hydrogen
  // CIE table, after H_I (neutral, col 0) and H_II (ionized, col 1).
  void parse_ion_name( const std::string& ion, std::string& element, int& state ) {
    size_t start = 0;
    if( !ion.empty() && ion[0] == 'x' ) start = 1;

    // Molecular hydrogen: no underscore, maps to the third hydrogen column.
    if( ion.compare(start, std::string::npos, "H2") == 0 ) {
      element = "H";
      state   = 2;
      return;
    }

    size_t us = ion.find('_', start);
    DYABLO_ASSERT_HOST_RELEASE( us != std::string::npos, "PassiveScalar_IC_constant_metallicity: malformed ion name '" << ion << "' (expected x<Elem>_<Roman>)" );
    element = ion.substr(start, us - start);
    std::string roman = ion.substr(us + 1);

    auto value = [](char c) {
      switch(c) {
        case 'I': return 1;
        case 'V': return 5;
        case 'X': return 10;
        default:  return 0;
      }
    };
    int total = 0;
    for( size_t i = 0; i < roman.size(); ++i ) {
      int v = value(roman[i]);
      DYABLO_ASSERT_HOST_RELEASE( v > 0, "PassiveScalar_IC_constant_metallicity: bad Roman numeral in ion '" << ion << "'" );
      if( i + 1 < roman.size() && v < value(roman[i+1]) ) total -= v;
      else total += v;
    }
    state = total - 1; // Roman I (neutral) -> first data column (state 0), II -> state 1, ...
  }
}

struct PassiveScalar_IC_constant_metallicity : public PassiveScalar_IC {
  ForeachCell &foreach_cell;

  //std::vector<std::string> passive_names;

  const real_t Z_over_Zsun;
  const bool cie_init;
  std::string data_path;
  const std::vector<std::string> metals;
  const std::vector<std::string> passive_scalar_names;
  const std::vector<std::string> ions;
  const std::vector<real_t> ion_fracs;
  const real_t gamma0;

  // CIE interpolation tables (device views, only used when cie_init == true)
  Kokkos::View<real_t*>  T_grid_d;
  Kokkos::View<real_t**> ion_table_d; // (n_ions, n_T)

  PassiveScalar_IC_constant_metallicity(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers,
        std::string passive_scalar_names) :
    foreach_cell(foreach_cell),
    //passive_names(passive_scalar_names),
    Z_over_Zsun ( configMap.getValue<real_t>("constant_metallicity", "Z_over_Zsun") ),
    cie_init    ( configMap.getValue<bool>("constant_metallicity", "cie_initialization", false) ),
    data_path   ( configMap.getValue<std::string>("cooling", "data_path") ),
    metals      ( configMap.getValue<std::vector<std::string>>("constant_metallicity", "metals", std::vector<std::string>{}) ),
    passive_scalar_names ( configMap.getValue<std::vector<std::string>>("passive_scalars", "passive_scalars_names", std::vector<std::string>{}) ),
    ions        ( configMap.getValue<std::vector<std::string>>("constant_metallicity", "ions", std::vector<std::string>{}) ),
    ion_fracs   ( configMap.getValue<std::vector<real_t>>("constant_metallicity", "ion_fracs", std::vector<real_t>{}) ),
    gamma0      ( configMap.getValue<real_t>("hydro", "gamma0", 1.4) )
  {
    if( cie_init ) {
      // ---------------------------------------------------------------
      // Load CIE ionization-fraction tables for each unique element
      // ---------------------------------------------------------------
      // Each file ${data_path}/CIE_init/cie_<Elem>.dat has the format:
      //   # T_K Elem_1 Elem_2 ...
      //   T1   f1     f2     ...
      //   T2   f1     f2     ...
      //   ...
      // Ionization states are 1-indexed in the header (Elem_1 = neutral, e.g. H_1 = HI),
      // matching the Roman-numeral convention of the ion names (xH_I = HI). The header
      // labels are documentation only: the reader maps each ion to a data column by
      // position (first data column = neutral), so the labels are never parsed.
      // Molecular hydrogen, when present, is the third hydrogen column (H2, after H_1
      // and H_2) and is requested via the ion name "xH2"; it is simply skipped when not
      // listed in the ions.
      // All files share the same T grid (101 log-spaced points from 1e3 to 1e9 K).
      std::set<std::string> unique_elements;
      for( const std::string& ion : ions ) {
        std::string elem; int state;
        parse_ion_name(ion, elem, state);
        unique_elements.insert(elem);
      }

      // element -> per-ion-state columns of fractions (column_idx -> values across T)
      std::map<std::string, std::vector<std::vector<real_t>>> tables_per_elem;
      std::vector<real_t> T_grid_host;

      for( const std::string& elem : unique_elements ) {
        std::string fpath = data_path + "/CIE_init/cie_" + elem + ".dat";
        std::ifstream f(fpath);
        DYABLO_ASSERT_HOST_RELEASE( f.is_open(), "PassiveScalar_IC_constant_metallicity: could not open CIE table '" << fpath << "'" );

        std::string header;
        std::getline(f, header);
        std::istringstream hss(header);
        std::string tok;
        int n_cols = -2; // skip '#' and 'T_K'
        while( hss >> tok ) n_cols++;
        DYABLO_ASSERT_HOST_RELEASE( n_cols > 0, "PassiveScalar_IC_constant_metallicity: empty CIE table '" << fpath << "'" );

        std::vector<std::vector<real_t>> table(n_cols);
        std::vector<real_t> Ts;
        std::string line;
        while( std::getline(f, line) ) {
          if( line.empty() ) continue;
          std::istringstream rs(line);
          real_t T;
          if( !(rs >> T) ) continue;
          Ts.push_back(T);
          for( int c = 0; c < n_cols; ++c ) {
            real_t v;
            rs >> v;
            table[c].push_back(v);
          }
        }
        tables_per_elem[elem] = std::move(table);

        if( T_grid_host.empty() ) {
          T_grid_host = std::move(Ts);
        } else {
          DYABLO_ASSERT_HOST_RELEASE( Ts.size() == T_grid_host.size(),
            "PassiveScalar_IC_constant_metallicity: CIE table '" << fpath << "' has a different T grid size than previously loaded tables" );
        }
      }

      const int n_T    = static_cast<int>(T_grid_host.size());
      const int n_ions = static_cast<int>(ions.size());

      // Pack into a (n_ions, n_T) host mirror, then deep_copy to device
      Kokkos::View<real_t*>  T_grid_dev    ("PassiveScalar_IC_constant_metallicity::T_grid",   n_T);
      Kokkos::View<real_t**> ion_table_dev ("PassiveScalar_IC_constant_metallicity::ion_table", n_ions, n_T);

      auto T_grid_h    = Kokkos::create_mirror_view(T_grid_dev);
      auto ion_table_h = Kokkos::create_mirror_view(ion_table_dev);

      for( int j = 0; j < n_T; ++j )
        T_grid_h(j) = T_grid_host[j];

      for( int i = 0; i < n_ions; ++i ) {
        std::string elem; int state;
        parse_ion_name(ions[i], elem, state);
        const auto& table = tables_per_elem.at(elem);
        DYABLO_ASSERT_HOST_RELEASE( state >= 0 && state < (int)table.size(),
          "PassiveScalar_IC_constant_metallicity: ion '" << ions[i] << "' state " << state
          << " is out of range for element '" << elem << "' (table has " << table.size() << " columns)" );
        const auto& col = table[state];
        for( int j = 0; j < n_T; ++j )
          ion_table_h(i, j) = col[j];
      }

      Kokkos::deep_copy(T_grid_dev,    T_grid_h);
      Kokkos::deep_copy(ion_table_dev, ion_table_h);

      T_grid_d    = T_grid_dev;
      ion_table_d = ion_table_dev;
    }
  }

  void init( UserData &U ) {

    std::vector<UserData::FieldAccessor::FieldInfo> new_metal_fields_info;
    for( const std::string& metal : metals ) {
        // add the number density field for the metal
        std::string field = "n" + metal;

        VarIndex ivar = new_metal_fields_info.size();
        new_metal_fields_info.push_back({field, ivar});
    }

    // if nCO in passive_scalar_names, add the nCO field
    bool CO_exists = false;
    if( std::find(passive_scalar_names.begin(), passive_scalar_names.end(), "nCO") != passive_scalar_names.end() )
      CO_exists = true;

    std::vector<UserData::FieldAccessor::FieldInfo> new_CO_fields_info;
    if( CO_exists ) {
      std::string field = "nCO";

      VarIndex ivar = new_CO_fields_info.size();
      new_CO_fields_info.push_back({field, ivar});
    }

    std::vector<UserData::FieldAccessor::FieldInfo> new_ion_fields_info;
    for( const std::string& ion : ions ) {
        // add the ion fraction field for the metal
        std::string field = ion;

        VarIndex ivar = new_ion_fields_info.size();
        new_ion_fields_info.push_back({field, ivar});
    }

    // Hydro fields needed both for the metal mass-weighting (rho) and, when
    // cie_init is enabled, for the per-cell T/mu used to interpolate the
    // ionization fractions.
    enum HydroIdx { IRHO = 0, IRHOVX, IRHOVY, IRHOVZ, IETOT };
    std::vector<dyablo::UserData::FieldAccessor_FieldInfo> fields_info;
    fields_info.push_back({"rho", IRHO});
    if( cie_init ) {
      fields_info.push_back({"rho_vx", IRHOVX});
      fields_info.push_back({"rho_vy", IRHOVY});
      fields_info.push_back({"rho_vz", IRHOVZ});
      fields_info.push_back({"e_tot",  IETOT});
    }


    auto Umetal = U.getAccessor( new_metal_fields_info );
    auto UCO    = U.getAccessor( new_CO_fields_info );
    auto Uion   = U.getAccessor( new_ion_fields_info );
    auto Uhydro = U.getAccessor( fields_info );



    // // generate atomic table // //
    struct Element_Info
    {
        int    atomic_number = -1;
        int    n_ions        = -1;
        double atomic_mass   = -1.0;
        double z_solar       = -1.0;
        double depletion     = 1.0;
    };

    std::map<std::string, Element_Info> elements = {
        //                                           Grevesse (2010)
        //    symbol      Z    n_ions   atomic_mass  nZ/nH (solar)  depletion
            { "H",      { 1,   1  + 1,  1.008,       1.0,           1.0  } },
            { "He",     { 2,   2  + 1,  4.0026,      8.51E-02,      1.0  } },
            { "C",      { 6,   6  + 1,  12.0107,     2.69E-04,      0.5  } },
            { "N",      { 7,   7  + 1,  14.0067,     6.76E-05,      0.6  } },
            { "O",      { 8,   8  + 1,  15.9994,     4.90E-04,      0.73 } },
            { "Ne",     { 10,  10 + 1,  20.1797,     8.51E-05,      1.0  } },
            { "Mg",     { 12,  12 + 1,  24.305,      3.98E-05,      0.16 } },
            { "Si",     { 14,  14 + 1,  28.0855,     3.24E-05,      0.1  } },
            { "S",      { 16,  16 + 1,  32.065,      1.32E-05,      1.0  } },
            { "Fe",     { 26,  26 + 1,  55.854,      3.16E-05,      0.01 } },
    };

    // loop over non-H and non-He metals to scale their nZ/nH (solar) by the metallicity
    for( size_t i=0; i<metals.size(); i++ ) {
        if( metals[i] == "H" || metals[i] == "He" )
            continue;
        auto it = elements.find(metals[i]);
        DYABLO_ASSERT_HOST_RELEASE( it != elements.end(),
          "PassiveScalar_IC_constant_metallicity: metal '" << metals[i] << "' is not in the atomic table" );
        it->second.z_solar *= Z_over_Zsun;
    }


    // Grevesse (2010) solar hydrogen mass fraction
    const real_t X = 0.7380;

    // number density coefficients for each metal (TOTAL, gas + dust), n_i/n_H = (nZ/nH)_solar
    std::vector<real_t> coeff_host(metals.size(), 0.0);
    for( size_t i=0; i<metals.size(); i++ ) {
        std::string metal_name = metals[i];
        auto it = elements.find(metal_name);
        DYABLO_ASSERT_HOST_RELEASE( it != elements.end(),
          "PassiveScalar_IC_constant_metallicity: metal '" << metals[i] << "' is not in the atomic table" );
        coeff_host[i] = it->second.z_solar;
    }


    auto amu_per_cc     = Units::amu() / Units::cm3();
    auto code_density   = Units::code_units().getUnit<Units::Density>();
    auto code_pressure  = Units::code_units().getUnit<Units::Pressure>();
    auto mp_over_kb     = Units::PROTON_MASS() / Units::KBOLTZ();
    auto K              = Units::Kelvin();

    // Pre-compute the (code_pressure / code_density * mp / kB) -> Kelvin conversion factor
    // so that T_over_mu [K] = (p_code / rho_code) * T_over_mu_unit_factor
    const real_t T_over_mu_factor = ( (1.0 * code_pressure) / (1.0 * code_density) * mp_over_kb ).convert_to(K);

    const real_t gamma_m1 = gamma0 - 1.0;

    Kokkos::View<real_t*> coeff_view("InitialConditions_uniform_metallicity::coeff", coeff_host.size());
    {
        Kokkos::View<real_t*> coeff_view_cpu( coeff_host.data(), coeff_host.size() );
        Kokkos::deep_copy(coeff_view, coeff_view_cpu);
    }

    // Capture CIE views by value so the device lambda can access them.
    const bool cie_init_local = cie_init;
    Kokkos::View<real_t*>  T_grid    = T_grid_d;
    Kokkos::View<real_t**> ion_table = ion_table_d;
    const int n_T = cie_init ? static_cast<int>(T_grid_d.extent(0)) : 0;

    // Also need a device view for the fallback constant ion_fracs case.
    Kokkos::View<real_t*> ion_fracs_view("PassiveScalar_IC_constant_metallicity::ion_fracs", ion_fracs.size());
    {
        Kokkos::View<real_t*, Kokkos::HostSpace> ion_fracs_cpu(const_cast<real_t*>(ion_fracs.data()), ion_fracs.size());
        Kokkos::deep_copy(ion_fracs_view, ion_fracs_cpu);
    }
    const size_t n_ions_local = ions.size();


    foreach_cell.foreach_cell( "PassiveScalar_IC_constant_metallicity::fill_U", U.getShape(),
                KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
    {
        const real_t rho = Uhydro.at(iCell_U, IRHO);
        const real_t nH = (rho * code_density.convert_to(amu_per_cc)) * X;

        // Compute T/mu (in K) from the local hydro state when CIE init is active.
        real_t T_over_mu = 0.0;
        if( cie_init_local ) {
            const real_t rho_vx = Uhydro.at(iCell_U, IRHOVX);
            const real_t rho_vy = Uhydro.at(iCell_U, IRHOVY);
            const real_t rho_vz = Uhydro.at(iCell_U, IRHOVZ);
            const real_t e_tot  = Uhydro.at(iCell_U, IETOT);
            const real_t ekin   = 0.5 * (rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz) / rho;
            const real_t e_int  = e_tot - ekin;
            const real_t p_code = gamma_m1 * e_int;
            T_over_mu = (p_code / rho) * T_over_mu_factor;
        }

        // Do the element number densities
        for( size_t i=0; i<coeff_view.size(); i++ )
            // Store conserved element number density n_i so hydro transports n_i with mass flux.
            Umetal.at(iCell_U, i) = nH * coeff_view(i);

        if ( CO_exists ) {
          UCO.at(iCell_U, 0) = nH * 1e-20; // some small number whatever
        }

        // Do the ion fractions
        for( size_t i=0; i<n_ions_local; i++ ) {
            if( cie_init_local ) {
                // 1D linear interpolation in T of the precomputed CIE fraction column
                // for ion i. Clamp to the edge values if T is outside the table range.
                real_t frac;
                if( T_over_mu <= T_grid(0) ) {
                    frac = ion_table(i, 0);
                } else if( T_over_mu >= T_grid(n_T - 1) ) {
                    frac = ion_table(i, n_T - 1);
                } else {
                    int lo = 0, hi = n_T - 1;
                    while( hi - lo > 1 ) {
                        int mid = (lo + hi) >> 1;
                        if( T_grid(mid) <= T_over_mu ) lo = mid;
                        else                            hi = mid;
                    }
                    const real_t t = (T_over_mu - T_grid(lo)) / (T_grid(lo+1) - T_grid(lo));
                    frac = ion_table(i, lo) * (1.0 - t) + ion_table(i, lo+1) * t;
                }
                // Store conserved ion (frac * rho) so hydro transports it with mass flux.
                Uion.at(iCell_U, i) = frac * rho;
            }
            else {
                Uion.at(iCell_U, i) = ion_fracs_view(i) * rho;
            }
        }
    });
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::PassiveScalar_IC_Factory,
                 dyablo::PassiveScalar_IC_constant_metallicity,
                 "constant_metallicity");
