#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "hyperbolic/policy/HyperbolicPolicy_passive_scalars.h"

#include "hyperbolic/scheme/Hyperbolic_RK2.h"

namespace dyablo{

class HydroUpdate_RK2
  : public Hyperbolic_RK2<HyperbolicPolicy_Hydro>
{
public:
  using Hyperbolic_RK2<HyperbolicPolicy_Hydro>::Hyperbolic_RK2;
};

namespace {
  using HyperbolicPolicy_Hydro_2_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_Hydro_impl, 2> >;
}

class HydroUpdate_RK2_2_passive_scalars 
  : public Hyperbolic_RK2<HyperbolicPolicy_Hydro_2_passive_scalars>
{
public:
  using Hyperbolic_RK2<HyperbolicPolicy_Hydro_2_passive_scalars>::Hyperbolic_RK2;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_RK2, 
                  "HydroUpdate_RK2")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory,
                  dyablo::HydroUpdate_RK2_2_passive_scalars,
                  "HydroUpdate_RK2_2_passive_scalars")