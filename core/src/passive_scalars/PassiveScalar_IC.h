#pragma once

#include "passive_scalars/PassiveScalar_IC_base.h"
#include "states/State_forward.h"

namespace dyablo{

class PassiveScalar_IC_uniform;
class PassiveScalar_IC_kelvin_helmholtz;

} // namespace dyablo



template<>
bool dyablo::PassiveScalar_IC_Factory::init()
{

  DECLARE_REGISTERED( dyablo::PassiveScalar_IC_uniform );
  DECLARE_REGISTERED( dyablo::PassiveScalar_IC_kelvin_helmholtz );

  return true;
}
