#include "hyperbolic/policy/HyperbolicPolicy_Rad.h"
#include "hyperbolic/scheme/Hyperbolic_euler.h"
#include "types.hpp"

namespace dyablo{

class RadUpdate_euler 
  : public Hyperbolic_euler<HyperbolicPolicy_Rad>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_Rad>::Hyperbolic_euler;

  // Update method is overridden to loop over radiation groups and set "rad_group_id" in scalar_data for each group, so that HyperbolicPolicy_Rad can use it to select the correct group in the state variables
  void update( UserData& U, ScalarSimulationData& scalar_data ) override
  {
    const bool had_prev_group = scalar_data.hasValue<int>("rad_group_id");
    const int prev_group = had_prev_group ? scalar_data.get<int>("rad_group_id") : -1;
    const int n_groups = scalar_data.hasValue<int>("n_groups") ? scalar_data.get<int>("n_groups") : 1;

    for (int g = 0; g < n_groups; ++g)
    {
      scalar_data.set<int>("rad_group_id", g);
      Hyperbolic_euler::update(U, scalar_data);
    }

    if (had_prev_group)
      scalar_data.set<int>("rad_group_id", prev_group);
  }
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::RadUpdate_euler, 
                  "RadUpdate_euler_M1")