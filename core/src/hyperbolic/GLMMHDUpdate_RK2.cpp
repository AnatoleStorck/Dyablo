#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"
#include "hyperbolic/policy/HyperbolicPolicy_passive_scalars.h"

#include "hyperbolic/scheme/Hyperbolic_RK2.h"

namespace dyablo{

class GLMMHDUpdate_RK2
  : public Hyperbolic_RK2<HyperbolicPolicy_GLMMHD>
{
public:
  using Hyperbolic_RK2<HyperbolicPolicy_GLMMHD>::Hyperbolic_RK2;
};

namespace {
using HyperbolicPolicy_GLMMHD_2_passive_scalars = HyperbolicPolicy_base< HyperbolicPolicy_passive_scalars_impl<HyperbolicPolicy_GLMMHD_impl, 2> >;
}

class GLMMHDUpdate_RK2_2_passive_scalars 
  : public Hyperbolic_RK2<HyperbolicPolicy_GLMMHD_2_passive_scalars>
{
  
public:
  using Hyperbolic_RK2<HyperbolicPolicy_GLMMHD_2_passive_scalars>::Hyperbolic_RK2;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::GLMMHDUpdate_RK2, 
                  "GLMMHDUpdate_RK2")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::GLMMHDUpdate_RK2_2_passive_scalars, 
                  "GLMMHDUpdate_RK2_2_passive_scalars")