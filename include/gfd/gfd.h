#pragma once

// GFD: GPU-Functional-Descriptor with CPU-side optimized descriptor writing
//
// Host-side unified header — include this for CPU-side library access.
//
// Usage (host code):
//   #include <gfd/gfd.h>
//
// Usage (device code / fused kernels):
//   #include <gfd/device.cuh>        // __device__ building blocks
//   #include <gfd/warp_spec.cuh>     // warp-specialized framework
//
// Or include individual components:
//   #include <gfd/descriptor_queue.h>
//   #include <gfd/tiled_queue.h>
//   #include <gfd/copy_engine.h>
//   #include <gfd/cpu_polling.h>
//   #include <gfd/warp_spec_session.h>
//   #include <gfd/staging_pool.h>
//   #include <gfd/pcie_topology.h>

#include "gfd/descriptor_queue.h"
#include "gfd/tiled_queue.h"
#include "gfd/sg_task_queue.h"
#include "gfd/copy_engine.h"
#include "gfd/cpu_polling.h"
#include "gfd/warp_spec_session.h"
#include "gfd/staging_pool.h"
#include "gfd/pcie_topology.h"
