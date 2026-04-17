#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"
#include "hyperbolic/policy/HyperbolicPolicy_passive_scalars.h"

#include "hyperbolic/scheme/Hyperbolic_euler.h"

namespace dyablo{

class GLMMHDUpdate_euler 
  : public Hyperbolic_euler<HyperbolicPolicy_GLMMHD>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_GLMMHD>::Hyperbolic_euler;
};

namespace {
using HyperbolicPolicy_GLMMHD_2_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_GLMMHD_impl, 2> >;
}

class GLMMHDUpdate_euler_2_passive_scalars 
  : public Hyperbolic_euler<HyperbolicPolicy_GLMMHD_2_passive_scalars>
{
  
public:
  using Hyperbolic_euler<HyperbolicPolicy_GLMMHD_2_passive_scalars>::Hyperbolic_euler;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::GLMMHDUpdate_euler, 
                  "GLMMHDUpdate_euler")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::GLMMHDUpdate_euler_2_passive_scalars, 
                  "GLMMHDUpdate_euler_2_passive_scalars")