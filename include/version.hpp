#pragma once

#define DSDECOMP_VERSION_MAJOR 0
#define DSDECOMP_VERSION_MINOR 3
#define DSDECOMP_VERSION_PATCH 0
#define DSDECOMP_VERSION_STRING "0.3.0"

namespace dsdecomp {
inline constexpr int version_major = DSDECOMP_VERSION_MAJOR;
inline constexpr int version_minor = DSDECOMP_VERSION_MINOR;
inline constexpr int version_patch = DSDECOMP_VERSION_PATCH;
inline constexpr const char* version_string = DSDECOMP_VERSION_STRING;
} // namespace dsdecomp
