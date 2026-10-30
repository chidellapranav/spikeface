# Run from Spike_old/: vitis_hls -f build_hls_ips.tcl
# SPIKEFACE_HLS_STAGE: csim (default), synth, cosim, export.
# SPIKEFACE_HLS_BLOCK: optional top function; default builds all six.
set repo [file dirname [file normalize [info script]]]
set src [file join $repo Spikeface_HLS]
set stage csim
if {[info exists ::env(SPIKEFACE_HLS_STAGE)]} {set stage $::env(SPIKEFACE_HLS_STAGE)}
if {$stage ni {csim synth cosim export}} {error "Unknown SPIKEFACE_HLS_STAGE: $stage"}
set blocks {
 {conv1_layer_axi solution1 tb_conv1_array_axi.cpp conv1_layer_axi.cpp}
 {conv1_pool_axi solution2 tb_conv1_pool_axi.cpp conv1_pool_axi.cpp}
 {conv2_layer_axi solution3 tb_conv2_array_axi.cpp conv2_layer_axi.cpp}
 {conv2_pool_axi solution4 tb_conv2_pool_axi.cpp conv2_pool_axi.cpp}
 {fc1_layer_axi solnfc1 tb_fc1_layer_axi.cpp fc1_layer_axi.cpp}
 {fc2_layer_axi solnfc2 tb_fc2_layer_axi.cpp fc2_layer_axi.cpp}
}
set selected 0
file mkdir [file join $repo hls_build]
cd [file join $repo hls_build]
foreach block $blocks {
 lassign $block top destination tb design
 if {[info exists ::env(SPIKEFACE_HLS_BLOCK)] && $::env(SPIKEFACE_HLS_BLOCK) ne $top} {continue}
 incr selected
 open_project -reset $top
 set_top $top
 add_files [file join $src $design] -cflags "-I$src"
 add_files -tb [file join $src $tb] -cflags "-I$src -Wno-unknown-pragmas"
 open_solution -reset -flow_target vivado sol
 set_part xc7z020clg400-1
 create_clock -period 10 -name default
 csim_design
 if {$stage ne "csim"} {csynth_design}
 if {$stage eq "cosim"} {cosim_design -rtl verilog}
 if {$stage eq "export"} {
  export_design -format ip_catalog
  set target [file join $src $destination impl]
  file mkdir $target
  set ip [file join $target ip]
  # Avoid stale nested directories: refuse an ambiguous overwrite.
  if {[file exists $ip]} {error "$ip already exists; choose a clean build workspace for export"}
  file copy [file join $repo hls_build $top sol impl ip] $target
 }
 close_project
}
if {$selected == 0} {error "Requested block was not found"}
exit
