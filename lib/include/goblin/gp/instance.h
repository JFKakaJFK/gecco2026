#pragma once
#ifndef _GOBLIN_GP_INSTANCE_H
#define _GOBLIN_GP_INSTANCE_H

#include "goblin/lib/instance.h"
#include "goblin/gp/context.h"

namespace goblin {

class GPInstanceBase : public InstanceBase {
 public:
  virtual const GPContext& context() const = 0;

  virtual ~GPInstanceBase() = default;
};

};  // namespace goblin

#endif /* _GOBLIN_GP_INSTANCE_H */
