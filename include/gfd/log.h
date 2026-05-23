#pragma once

// GFD Structured Logging
//
// Compile-time log level control:
//   0 = silent (no output)
//   1 = error only
//   2 = info + error (default)
//   3 = debug + info + error
//
// Override at compile time: -DGFD_LOG_LEVEL=1

#include <cstdio>

#ifndef GFD_LOG_LEVEL
#define GFD_LOG_LEVEL 2
#endif

#define GFD_LOG_ERROR(...) do { \
    if (GFD_LOG_LEVEL >= 1) fprintf(stderr, "[GFD ERROR] " __VA_ARGS__); \
} while(0)

#define GFD_LOG_INFO(...) do { \
    if (GFD_LOG_LEVEL >= 2) fprintf(stderr, "[GFD] " __VA_ARGS__); \
} while(0)

#define GFD_LOG_DEBUG(...) do { \
    if (GFD_LOG_LEVEL >= 3) fprintf(stderr, "[GFD DBG] " __VA_ARGS__); \
} while(0)
