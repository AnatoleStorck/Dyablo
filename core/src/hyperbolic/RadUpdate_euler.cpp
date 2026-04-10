#include "hyperbolic/policy/HyperbolicPolicy_Rad.h"
#include "hyperbolic/scheme/Hyperbolic_euler.h"

namespace dyablo{

class RadUpdate_euler
  : public Hyperbolic_euler<HyperbolicPolicy_Rad>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_Rad>::Hyperbolic_euler;


  // Update method is overridden to loop over radiation groups and set "rad_group_id" in scalar_data for each group, so that HyperbolicPolicy_Rad can use it to select the correct group in the state variables
  void update( UserData& U, ScalarSimulationData& scalar_data ) override
  {
    const int n_groups = policy_params.policy_params.n_groups;
    for (int g = 0; g < n_groups; ++g)
    {
      const std::string group_str = std::to_string(g);
      // Rename variables for the current group so the solver operates on the correct group data
      U.move_field("e_rad", "e_rad_" + group_str);
      U.move_field("fx_rad", "fx_rad_" + group_str);
      U.move_field("fy_rad", "fy_rad_" + group_str);
      U.move_field("fz_rad", "fz_rad_" + group_str);

      U.move_field("e_rad_next", "e_rad_" + group_str + "_next");
      U.move_field("fx_rad_next", "fx_rad_" + group_str + "_next");
      U.move_field("fy_rad_next", "fy_rad_" + group_str + "_next");
      U.move_field("fz_rad_next", "fz_rad_" + group_str + "_next");

      Hyperbolic_euler::update(U, scalar_data);

      // Rename back for consistency with the next group iteration
      U.move_field("e_rad_" + group_str, "e_rad");
      U.move_field("fx_rad_" + group_str, "fx_rad");
      U.move_field("fy_rad_" + group_str, "fy_rad");
      U.move_field("fz_rad_" + group_str, "fz_rad");

      U.move_field("e_rad_" + group_str + "_next", "e_rad_next");
      U.move_field("fx_rad_" + group_str + "_next", "fx_rad_next");
      U.move_field("fy_rad_" + group_str + "_next", "fy_rad_next");
      U.move_field("fz_rad_" + group_str + "_next", "fz_rad_next");
    }
  }

};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory,
                  dyablo::RadUpdate_euler,
                  "RadUpdate_euler_M1")