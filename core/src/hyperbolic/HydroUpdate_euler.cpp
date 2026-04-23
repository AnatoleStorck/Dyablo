#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "hyperbolic/policy/HyperbolicPolicy_passive_scalars.h"

#include "hyperbolic/scheme/Hyperbolic_euler.h"

namespace dyablo{

class HydroUpdate_euler 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro>::Hyperbolic_euler;
};

namespace {
  using HyperbolicPolicy_Hydro_2_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_Hydro_impl, 2> >;
  using HyperbolicPolicy_Hydro_4_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_Hydro_impl, 4> >;
}

class HydroUpdate_euler_2_passive_scalars 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro_2_passive_scalars>
{
  
public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro_2_passive_scalars>::Hyperbolic_euler;
};

class HydroUpdate_euler_4_passive_scalars 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro_4_passive_scalars>
{
  
public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro_4_passive_scalars>::Hyperbolic_euler;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_euler, 
                  "HydroUpdate_euler")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_euler_2_passive_scalars, 
                  "HydroUpdate_euler_2_passive_scalars")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_euler_4_passive_scalars, 
                  "HydroUpdate_euler_4_passive_scalars")