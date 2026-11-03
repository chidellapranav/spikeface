cmake_minimum_required(VERSION 3.24)
project(SpikeFace LANGUAGES NONE)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build/hardware")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
option(SPIKEFACE_ENABLE_HLS "Enable optimized LFW HLS targets" ON)
option(SPIKEFACE_ENABLE_VIVADO "Enable Vivado targets" ON)
set(SPIKEFACE_HLS_VERSION "2022.2" CACHE STRING "Expected HLS release (empty disables check)")
set(SPIKEFACE_VIVADO_VERSION "2022.2" CACHE STRING "Expected Vivado release (empty disables check)")
set(SPIKEFACE_BOARD_REPO "${CMAKE_CURRENT_SOURCE_DIR}/PL_Design/board_files" CACHE PATH "Repository containing pynq-z2/A.0/board.xml")
option(SPIKEFACE_USE_PACKAGED_IP "Unpack committed IP archives for Vivado instead of HLS build exports" OFF)
include(FindXilinxTools)

# Independent stage directories: the existing Tcl resets its selected projects.
if(SPIKEFACE_ENABLE_HLS)
  spikeface_find_tool(HLS vitis_hls)
  set(hls_source "${CMAKE_CURRENT_SOURCE_DIR}/newarch_HLS")
  foreach(stage csim synth cosim export)
    set(run_stage "${stage}")
    if(stage STREQUAL "export")
      set(run_stage all) # Verify RTL before packaging IP.
    endif()
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hls/${stage}")
    add_custom_target(hls-${stage}
      COMMAND "${CMAKE_COMMAND}" -E env
        --unset=SPIKEFACE_HLS_BLOCK
        "SPIKEFACE_HLS_STAGE=${run_stage}"
        "SPIKEFACE_HLS_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/hls/${stage}"
        "SPIKEFACE_CLOCK_NS=20"
        "SPIKEFACE_FP_ADD_IMPL=fabric"
        "SPIKEFACE_COSIM_TEST=stress"
        "SPIKEFACE_COSIM_STALL=0"
        "${SPIKEFACE_HLS_EXECUTABLE}" -f "${hls_source}/build_lfw_hls.tcl"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hls/${stage}"
      USES_TERMINAL VERBATIM
      COMMENT "Running ${stage} for all three optimized LFW IPs")
  endforeach()
endif()

if(SPIKEFACE_ENABLE_VIVADO)
  spikeface_find_tool(VIVADO vivado)
  foreach(block CONV1 CONV2 READOUT)
    string(TOLOWER "${block}" block_lower)
    set(default_ip "${CMAKE_CURRENT_BINARY_DIR}/hls/export/lfw_${block_lower}_axi/sol/impl/ip")
    if(SPIKEFACE_USE_PACKAGED_IP)
      set(archive "${CMAKE_CURRENT_SOURCE_DIR}/hls_ip/lfw_${block_lower}/export.zip")
      set(default_ip "${CMAKE_CURRENT_BINARY_DIR}/packaged_ip/lfw_${block_lower}")
      if(NOT EXISTS "${archive}")
        message(FATAL_ERROR "Missing packaged IP archive: ${archive}")
      endif()
      file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${default_ip}")
      if(NOT EXISTS "${default_ip}/component.xml")
        message(FATAL_ERROR "Archive does not contain component.xml: ${archive}")
      endif()
    endif()
    set(SPIKEFACE_${block}_IP
      "${default_ip}"
      CACHE PATH "Unpacked ${block} IP repository containing component.xml")
  endforeach()
  foreach(mode bd bit)
    add_custom_target(vivado-${mode}
      COMMAND "${CMAKE_COMMAND}"
        "-DVIVADO=${SPIKEFACE_VIVADO_EXECUTABLE}"
        "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/PL_Design/create_lfw_pl.tcl"
        "-DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/vivado/${mode}"
        "-DBOARD_REPO=${SPIKEFACE_BOARD_REPO}"
        "-DCONV1_IP=${SPIKEFACE_CONV1_IP}"
        "-DCONV2_IP=${SPIKEFACE_CONV2_IP}"
        "-DREADOUT_IP=${SPIKEFACE_READOUT_IP}"
        "-DMODE=${mode}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunVivado.cmake"
      USES_TERMINAL VERBATIM)
  endforeach()
endif()
