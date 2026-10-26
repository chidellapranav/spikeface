# Authoritative 64x64 / 53-class model. Bundled fixture is ready to use.
set repo [file dirname [file normalize [info script]]]
set src $repo
set fixture [file join $repo stress_sim build lfw_fixture.bin]
if {![file exists $fixture]} {error "Missing bundled stress_sim/build/lfw_fixture.bin"}
set stage csim
if {[info exists ::env(SPIKEFACE_HLS_STAGE)]} {set stage $::env(SPIKEFACE_HLS_STAGE)}
if {$stage ni {csim synth cosim export}} {error "Unknown stage: $stage"}
set build_dir [file join $repo hls_build]
if {[info exists ::env(SPIKEFACE_HLS_BUILD_DIR)]} {set build_dir [file normalize $::env(SPIKEFACE_HLS_BUILD_DIR)]}
file mkdir $build_dir
cd $build_dir
set selected 0
set period 20
if {[info exists ::env(SPIKEFACE_CLOCK_NS)]} {set period $::env(SPIKEFACE_CLOCK_NS)}
# Explicit IEEE float add/sub implementations avoid the 2022.2 DSP48 adder
# models' unknown OPMODE warnings. This is a hardware binding, not a change to
# reduction order, precision, reset semantics or test tolerances.
set add_impl fabric
if {[info exists ::env(SPIKEFACE_FP_ADD_IMPL)]} {set add_impl $::env(SPIKEFACE_FP_ADD_IMPL)}
if {$add_impl ni {fabric fulldsp}} {error "SPIKEFACE_FP_ADD_IMPL must be fabric or fulldsp"}
set tops {lfw_conv1_axi lfw_conv2_axi lfw_readout_axi}
if {[info exists ::env(SPIKEFACE_HLS_BLOCK)] && $::env(SPIKEFACE_HLS_BLOCK) eq "lfw_chain_verify"} {
 lappend tops lfw_chain_verify
}
foreach top $tops {
 if {[info exists ::env(SPIKEFACE_HLS_BLOCK)] && $::env(SPIKEFACE_HLS_BLOCK) ne $top} {continue}
 incr selected
 open_project -reset $top
 set_top $top
 add_files [file join $src lfw_model.cpp] -cflags "-I$src -ffp-contract=off"
 add_files [file join $src lfw_axi.cpp] -cflags "-I$src -ffp-contract=off"
 set testbench [file join $src tb_lfw_axi.cpp]
 if {[info exists ::env(SPIKEFACE_COSIM_TEST)] && $::env(SPIKEFACE_COSIM_TEST) eq "stress"} {
  set testbench [file join $repo stress_sim tb_lfw_rtl_stress.cpp]
 }
 if {$top eq "lfw_chain_verify"} {
  add_files [file join $repo stress_sim rtl_chain.cpp] -cflags "-I$src -ffp-contract=off"
  set testbench [file join $repo stress_sim tb_lfw_rtl_chain.cpp]
 }
 add_files -tb $testbench -cflags "-I$src -ffp-contract=off"
 add_files -tb $fixture
 open_solution -reset -flow_target vivado sol
 set_part xc7z020clg400-1
 create_clock -period $period -name default
 config_op fadd -impl $add_impl
 config_op fsub -impl $add_impl
 csim_design
 if {$stage ne "csim"} {csynth_design}
 if {$stage eq "cosim"} {
  if {[info exists ::env(SPIKEFACE_COSIM_STALL)] && $::env(SPIKEFACE_COSIM_STALL) eq "1"} {
   cosim_design -rtl verilog -random_stall
  } else {
   cosim_design -rtl verilog
  }
 }
 if {$stage eq "export"} {export_design -format ip_catalog}
 close_project
}
if {$selected==0} {error "Requested LFW block not found"}
exit
