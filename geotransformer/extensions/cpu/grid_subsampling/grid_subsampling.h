#pragma once

#include <vector>
#include <cstdint>
#include "../../common/torch_helper.h"

std::vector<at::Tensor> grid_subsampling(
  at::Tensor points,
  at::Tensor lengths,
  float voxel_size
);
