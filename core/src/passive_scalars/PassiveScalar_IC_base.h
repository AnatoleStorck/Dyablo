#pragma once 

#include "kokkos_shared.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "UserData.h"

namespace dyablo{


class PassiveScalar_IC{
public:
  // PassiveScalar_IC(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,  
  //     Timers& timers
  //     std::string passive_scalar_name);
  virtual void init( UserData& U ) = 0;
  virtual ~PassiveScalar_IC(){}
};

using PassiveScalar_IC_Factory = RegisteringFactory<PassiveScalar_IC, 
  ConfigMap&,
  ForeachCell&, 
  Timers&,
  std::string>;


} // namespace dyablo