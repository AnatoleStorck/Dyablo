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
  using HyperbolicPolicy_Hydro_7_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_Hydro_impl, 7> >;
  using HyperbolicPolicy_Hydro_16_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_Hydro_impl, 16> >;
  using HyperbolicPolicy_Hydro_25_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_Hydro_impl, 25> >;
  using HyperbolicPolicy_Hydro_27_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_Hydro_impl, 27> >;
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

class HydroUpdate_euler_7_passive_scalars 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro_7_passive_scalars>
{
  
public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro_7_passive_scalars>::Hyperbolic_euler;
};

class HydroUpdate_euler_16_passive_scalars 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro_16_passive_scalars>
{

public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro_16_passive_scalars>::Hyperbolic_euler;
};

class HydroUpdate_euler_25_passive_scalars 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro_25_passive_scalars>
{
  
public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro_25_passive_scalars>::Hyperbolic_euler;
};

class HydroUpdate_euler_27_passive_scalars 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro_27_passive_scalars>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro_27_passive_scalars>::Hyperbolic_euler;
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

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_euler_7_passive_scalars, 
                  "HydroUpdate_euler_7_passive_scalars")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_euler_16_passive_scalars, 
                  "HydroUpdate_euler_16_passive_scalars")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_euler_25_passive_scalars, 
                  "HydroUpdate_euler_25_passive_scalars")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_euler_27_passive_scalars, 
                  "HydroUpdate_euler_27_passive_scalars")