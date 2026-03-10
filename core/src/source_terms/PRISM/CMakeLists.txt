add_library(PRISM "")

target_sources(PRISM
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/CoolingUpdate_PRISM.cpp
  )

target_include_directories(PRISM
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/inih ) 

target_link_libraries(PRISM PUBLIC dyablo::utils::mpi)

# Build gizmo_rtz as an independent static library and link it
add_subdirectory(gizmo_rtz)
target_link_libraries(PRISM PUBLIC gizmo_rtz)

#Add an alias so that library can be used inside the build tree
add_library(dyablo::PRISM ALIAS PRISM)


