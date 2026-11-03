# Explicit executable paths take priority; otherwise search a tool root and PATH.
function(spikeface_find_tool kind name)
  set(SPIKEFACE_${kind}_ROOT "" CACHE PATH "${kind} installation directory")
  set(tool_hints)
  if(SPIKEFACE_${kind}_ROOT)
    list(APPEND tool_hints "${SPIKEFACE_${kind}_ROOT}/bin")
  endif()
  find_program(SPIKEFACE_${kind}_EXECUTABLE
    NAMES "${name}" "${name}.bat"
    HINTS ${tool_hints}
    DOC "Full path to ${name} executable or Windows launcher")
  if(NOT SPIKEFACE_${kind}_EXECUTABLE OR NOT EXISTS "${SPIKEFACE_${kind}_EXECUTABLE}")
    message(FATAL_ERROR
      "${name} not found. Set SPIKEFACE_${kind}_EXECUTABLE or SPIKEFACE_${kind}_ROOT, or disable SPIKEFACE_ENABLE_${kind}.")
  endif()
  execute_process(COMMAND "${SPIKEFACE_${kind}_EXECUTABLE}" -version
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
    TIMEOUT 60)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Cannot query ${name} version: ${result}\n${output}\n${error}")
  endif()
  string(REGEX MATCH "[0-9][0-9][0-9][0-9]\\.[0-9]+" detected "${output}\n${error}")
  set(expected "${SPIKEFACE_${kind}_VERSION}")
  if(NOT expected STREQUAL "" AND NOT detected STREQUAL expected)
    message(FATAL_ERROR "${name}: expected ${expected}, detected '${detected}'. Select the matching installation or change SPIKEFACE_${kind}_VERSION.")
  endif()
  message(STATUS "${kind}: ${SPIKEFACE_${kind}_EXECUTABLE} (release ${detected})")
endfunction()
