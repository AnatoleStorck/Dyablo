#pragma once

#include "source_terms/SourceUpdate_base.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

namespace dyablo {

class SourceUpdate_Cooling_FF;
class SourceUpdate_GLM;
class SourceUpdate_Photons_Beam;
class SourceUpdate_Photons_Stromgren;
class SourceUpdate_Ionization_Stromgren;

#ifdef DYABLO_USE_FFT
class SourceUpdate_Turbulent_Forcing;
#endif

class SourceUpdate_UV_Background;

template< typename Policy >
class SourceUpdate_cooling_grackle_table;

#ifdef DYABLO_USE_PRISM
template< typename Policy >
class CoolingUpdate_PRISM;
#endif

} //namespace dyablo


template<>
inline bool dyablo::SourceUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::SourceUpdate_Cooling_FF);
  DECLARE_REGISTERED(dyablo::SourceUpdate_GLM);
  DECLARE_REGISTERED(dyablo::SourceUpdate_Photons_Beam);
  DECLARE_REGISTERED(dyablo::SourceUpdate_Photons_Stromgren);
  DECLARE_REGISTERED(dyablo::SourceUpdate_Ionization_Stromgren);
  DECLARE_REGISTERED(dyablo::SourceUpdate_cooling_grackle_table<dyablo::HyperbolicPolicy_Hydro>);
  DECLARE_REGISTERED(dyablo::SourceUpdate_cooling_grackle_table<dyablo::HyperbolicPolicy_GLMMHD>);

#ifdef DYABLO_USE_FFT
  DECLARE_REGISTERED(dyablo::SourceUpdate_Turbulent_Forcing);
#endif
  DECLARE_REGISTERED(dyablo::SourceUpdate_UV_Background);

#ifdef DYABLO_USE_PRISM
  DECLARE_REGISTERED(dyablo::CoolingUpdate_PRISM<dyablo::HyperbolicPolicy_Hydro>);
#endif

  dyablo::load_dyablo_plugins_lib();
  return true;
}

