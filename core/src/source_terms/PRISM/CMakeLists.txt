add_library(PRISM "")

target_sources(PRISM
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/CoolingUpdate_PRISM.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CoolingUpdate_PRISM_GPU.cpp
  )

target_include_directories(PRISM
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/inih ) 

target_link_libraries(PRISM PUBLIC dyablo::utils::mpi)

# Bound on CompactIonData's flat ion-fraction arrays. The default (113) covers
# every element PRISM has tabulated data for; setting it to the number of slots
# the configured network actually uses shrinks the per-cell solver state, which
# every kernel in the subcycle streams. The driver aborts at startup if the
# network does not fit.
set(DYABLO_PRISM_MAX_TOTAL_IONS "113" CACHE STRING
    "Max total ion+molecule states across all elements in the PRISM network")
target_compile_definitions(PRISM PUBLIC MAX_TOTAL_IONS=${DYABLO_PRISM_MAX_TOTAL_IONS})

# Build gizmo_rtz as an independent static library and link it
add_subdirectory(gizmo_rtz)
target_link_libraries(PRISM PUBLIC gizmo_rtz)

#Add an alias so that library can be used inside the build tree
add_library(dyablo::PRISM ALIAS PRISM)


