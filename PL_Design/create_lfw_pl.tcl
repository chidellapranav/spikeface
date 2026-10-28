# Vivado 2022.2 / PYNQ-Z2 / authoritative 64x64, 53-class LFW model.
# Run: vivado -mode batch -source create_lfw_pl.tcl -tclargs bd /absolute/build/path
# Modes: bd (validate/generate), bit (route/check timing/bitstream/XSA).
# Build output MUST be outside PL_Design. Existing projects are never overwritten.
# Finish C/RTL co-simulation and Export RTL -> IP Catalog for ALL three GUI IPs.
# Optional environment overrides: SPIKEFACE_HLS_ROOT, SPIKEFACE_BOARD_REPO,
# SPIKEFACE_CONV1_IP, SPIKEFACE_CONV2_IP, SPIKEFACE_READOUT_IP.
# Each IP override is an unpacked IP directory containing component.xml.
# Software contract (PS/lfw_driver.py): two simple-mode DMA instances below;
# input 4096 float32 = 16384 bytes, output 53 float32 = 212 bytes, EVERY timestep.
# Params: 224 / 3248 / 217247 float32 values in contiguous, flushed PS DDR.
# Program params low/high 0x10/0x14, beta 0x1c; conv threshold 0x24,
# conv clear_state 0x2c; readout clear_state 0x24. Check exported headers/HWH.
# Set all clear_state=1 on first timestep of EACH image, then 0. No auto_restart.
# Start readout -> conv2 -> conv1; arm output DMA before input DMA; wait each step.
# Match model preprocessing/label order/checkpoint. PL cannot establish accuracy.
# Linux: XSA includes PS DDR/IO preset and five IRQs; generate device tree from
# this XSA. Use either PYNQ userspace DMA or kernel DMA ownership, never both.
# Hardware acceptance still requires completed RTL checks and golden board replay.

proc env_or {name fallback} {
    if {[info exists ::env($name)]} {return [file normalize $::env($name)]}
    return [file normalize $fallback]
}
set mode bd
if {[llength $argv] > 0} {set mode [lindex $argv 0]}
if {$mode ni {bd bit}} {error "Mode must be bd or bit"}
set script_dir [file dirname [file normalize [info script]]]
set hls [env_or SPIKEFACE_HLS_ROOT /home/pranavsai/Spikeface_Project/Branch_Linux/spikeface/newarch_HLS]
set board_repo [env_or SPIKEFACE_BOARD_REPO /home/pranavsai/PYNQZ2]
set out [file normalize [file join [pwd] lfw_pynq_z2_build]]
if {[llength $argv] > 1} {set out [file normalize [lindex $argv 1]]}
if {$out eq $script_dir || [string first "$script_dir/" "$out/"] == 0} {
    error "Choose a build directory outside the script folder"
}
if {[file exists [file join $out lfw_pynq_z2.xpr]]} {error "Project already exists in $out; choose a fresh output directory"}
set repos {}
foreach key {CONV1 CONV2 READOUT} top {lfw_conv1_axi lfw_conv2_axi lfw_readout_axi} {
    set p [env_or SPIKEFACE_${key}_IP [file join $hls gui_projects $top sol impl ip]]
    if {![file isfile [file join $p component.xml]]} {
        error "Missing packaged $top: $p/component.xml. Complete co-simulation and Export RTL (IP Catalog) first."
    }
    lappend repos $p
}
if {![file isfile [file join $board_repo pynq-z2 A.0 board.xml]]} {
    error "Missing PYNQ-Z2 board preset below $board_repo/pynq-z2/A.0"
}
# Register the board repository BEFORE looking up board parts / applying preset.
set_param board.repoPaths [lsort -unique [concat [get_param board.repoPaths] [list $board_repo]]]
set boards [get_board_parts -quiet tul.com.tw:pynq-z2:part0:*]
if {[llength $boards] != 1} {error "Expected one PYNQ-Z2 board part, found: $boards"}
create_project lfw_pynq_z2 $out -part xc7z020clg400-1
set_property board_part [lindex $boards 0] [current_project]
set_property target_language Verilog [current_project]
set_property ip_repo_paths $repos [current_project]
update_ip_catalog
foreach top {lfw_conv1_axi lfw_conv2_axi lfw_readout_axi} {
    if {[llength [get_ipdefs -quiet xilinx.com:hls:${top}:1.0]] != 1} {error "IP catalog missing xilinx.com:hls:${top}:1.0"}
}
create_bd_design lfw_system
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells processing_system7_0]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_USE_S_AXI_HP1 {1} CONFIG.PCW_EN_CLK0_PORT {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {50} CONFIG.PCW_USE_FABRIC_INTERRUPT {1} CONFIG.PCW_IRQ_F2P_INTR {1}] [get_bd_cells processing_system7_0]
foreach top {lfw_conv1_axi lfw_conv2_axi lfw_readout_axi} {
    create_bd_cell -type ip -vlnv xilinx.com:hls:${top}:1.0 ${top}_0
}
# Preserve the existing board driver's exact DMA instance names.
foreach i {0 1} {
    set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_$i]
    set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {23} CONFIG.c_addr_width {32} CONFIG.c_include_mm2s [expr {$i == 0}] CONFIG.c_include_s2mm [expr {$i == 1}]] $dma
    if {$i == 0} {
        set_property -dict [list CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_m_axis_mm2s_tdata_width {32} CONFIG.c_include_mm2s_dre {0} CONFIG.c_mm2s_burst_size {16}] $dma
    } else {
        set_property -dict [list CONFIG.c_m_axi_s2mm_data_width {32} CONFIG.c_s_axis_s2mm_tdata_width {32} CONFIG.c_include_s2mm_dre {0} CONFIG.c_s2mm_burst_size {16}] $dma
    }
}
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 control_interconnect
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {5}] [get_bd_cells control_interconnect]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 parameter_interconnect
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells parameter_interconnect]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 dma_interconnect
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells dma_interconnect]
proc intf {a b} {connect_bd_intf_net [get_bd_intf_pins $a] [get_bd_intf_pins $b]}
intf processing_system7_0/M_AXI_GP0 control_interconnect/S00_AXI
set i 0
foreach target {lfw_conv1_axi_0/s_axi_CTRL lfw_conv2_axi_0/s_axi_CTRL lfw_readout_axi_0/s_axi_CTRL axi_dma_0/S_AXI_LITE axi_dma_1/S_AXI_LITE} {
    intf [format control_interconnect/M%02d_AXI $i] $target
    incr i
}
set i 0
foreach top {lfw_conv1_axi lfw_conv2_axi lfw_readout_axi} {
    intf ${top}_0/m_axi_PARAM [format parameter_interconnect/S%02d_AXI $i]
    incr i
}
intf parameter_interconnect/M00_AXI processing_system7_0/S_AXI_HP0
intf axi_dma_0/M_AXI_MM2S dma_interconnect/S00_AXI
intf axi_dma_1/M_AXI_S2MM dma_interconnect/S01_AXI
intf dma_interconnect/M00_AXI processing_system7_0/S_AXI_HP1
# DMA MM2S TLAST/TKEEP are unused by fixed-count conv1 input. Internal byte
# streams have no TLAST. Readout alone provides TLAST/TKEEP to the output DMA.
intf axi_dma_0/M_AXIS_MM2S lfw_conv1_axi_0/input_r
intf lfw_conv1_axi_0/output_r lfw_conv2_axi_0/input_r
intf lfw_conv2_axi_0/output_r lfw_readout_axi_0/input_r
intf lfw_readout_axi_0/output_r axi_dma_1/S_AXIS_S2MM
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_fclk0
# Vivado derives reset polarity from FCLK_RESET0_N; verify after propagation.
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 clock_locked
set_property CONFIG.CONST_VAL {1} [get_bd_cells clock_locked]
connect_bd_net [get_bd_pins clock_locked/dout] [get_bd_pins rst_fclk0/dcm_locked]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_fclk0/ext_reset_in]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins rst_fclk0/slowest_sync_clk]
foreach p [get_bd_pins -hier -filter {TYPE == clk && DIR == I}] {
    if {![llength [get_bd_nets -quiet -of_objects $p]]} {connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] $p}
}
foreach p [get_bd_pins -hier -filter {TYPE == rst && DIR == I}] {
    if {![llength [get_bd_nets -quiet -of_objects $p]]} {
        # All remaining resets are peripheral active-low AXI/HLS resets.
        if {![string match *resetn $p] && ![string match *aresetn $p] && ![string match */ap_rst_n $p]} {continue}
        connect_bd_net [get_bd_pins rst_fclk0/peripheral_aresetn] $p
    }
}
foreach top {lfw_conv1_axi lfw_conv2_axi lfw_readout_axi} {
    if {![llength [get_bd_nets -quiet -of_objects [get_bd_pins ${top}_0/ap_rst_n]]]} {
        error "Unconnected HLS reset: $top"
    }
}
# IRQ_F2P[0:4] = input DMA, output DMA, conv1, conv2, readout.
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 interrupts
set_property CONFIG.NUM_PORTS {5} [get_bd_cells interrupts]
set i 0
foreach pin {axi_dma_0/mm2s_introut axi_dma_1/s2mm_introut lfw_conv1_axi_0/interrupt lfw_conv2_axi_0/interrupt lfw_readout_axi_0/interrupt} {
    connect_bd_net [get_bd_pins $pin] [get_bd_pins interrupts/In$i]
    incr i
}
connect_bd_net [get_bd_pins interrupts/dout] [get_bd_pins processing_system7_0/IRQ_F2P]
# Fixed software control map, 64 KiB each; DDR auto-assigned for all five masters.
foreach target {lfw_conv1_axi_0/s_axi_CTRL/* lfw_conv2_axi_0/s_axi_CTRL/* lfw_readout_axi_0/s_axi_CTRL/* axi_dma_0/S_AXI_LITE/* axi_dma_1/S_AXI_LITE/*} offset {0x43C00000 0x43C10000 0x43C20000 0x40400000 0x40410000} {
    if {[llength [get_bd_addr_segs $target]] != 1} {error "Expected one register segment: $target"}
    assign_bd_address -offset $offset -range 64K -target_address_space [get_bd_addr_spaces processing_system7_0/Data] [get_bd_addr_segs $target]
}
assign_bd_address
validate_bd_design
if {[get_property CONFIG.C_EXT_RESET_HIGH [get_bd_cells rst_fclk0]] != 0} {
    error "PS FCLK_RESET0_N must propagate as active-low to proc_sys_reset"
}
foreach p [get_bd_pins -hier -filter {TYPE == clk && DIR == I}] {
    if {![llength [get_bd_nets -quiet -of_objects $p]]} {error "Unconnected clock: $p"}
}
# The supplied A.0 preset has negative DQS-to-clock skews. Vivado 2022.2
# reports PSU-1..4 critical warnings. Preserve board values; DDR operation
# must be confirmed on this board before any accuracy result is accepted.
save_bd_design
set bd [get_files */lfw_system.bd]
generate_target all $bd
set wrapper [make_wrapper -files $bd -top]
add_files -norecurse $wrapper
set_property top lfw_system_wrapper [current_fileset]
update_compile_order -fileset sources_1
write_hw_platform -fixed -force -file [file join $out lfw_system.xsa]
puts "Validated 50 MHz PYNQ-Z2 BD with board DDR preset. Project: $out/lfw_pynq_z2.xpr"
if {$mode eq "bit"} {
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed"}
    launch_runs impl_1 -to_step route_design -jobs 4
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
    open_run impl_1
    report_timing_summary -report_unconstrained -file [file join $out timing_summary.rpt]
    report_utilization -file [file join $out utilization.rpt]
    report_drc -file [file join $out drc.rpt]
    set setup [get_timing_paths -delay_type max -max_paths 1]
    set hold [get_timing_paths -delay_type min -max_paths 1]
    if {![llength $setup] || ![llength $hold]} {error "Missing timing paths"}
    if {[get_property SLACK $setup] < 0 || [get_property SLACK $hold] < 0} {error "Routed setup/hold timing failed; see timing_summary.rpt"}
    if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error}]]} {error "DRC errors; see drc.rpt"}
    close_design
    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Bitstream generation failed"}
    open_run impl_1
    write_hw_platform -fixed -include_bit -force -file [file join $out lfw_system.xsa]
    file copy [file join $out lfw_pynq_z2.runs impl_1 lfw_system_wrapper.bit] [file join $out lfw_system.bit]
    file copy [file join $out lfw_pynq_z2.gen sources_1 bd lfw_system hw_handoff lfw_system.hwh] [file join $out lfw_system.hwh]
    puts "Timing passed; matching lfw_system.bit/.hwh/.xsa exported to $out"
}
