# Create GUI-ready projects without running simulation or synthesis.
set repo [file dirname [file normalize [info script]]]
set project_root [file join $repo gui_projects]
set fixture [file join $repo stress_sim build lfw_fixture.bin]
if {![file exists $fixture]} {error "Missing fixture: $fixture"}
file mkdir $project_root
cd $project_root
foreach top {lfw_conv1_axi lfw_conv2_axi lfw_readout_axi} {
    if {[file exists [file join $project_root $top]]} {
        error "Project already exists: $top; open it in the GUI instead of recreating it"
    }
    open_project $top
    set_top $top
    foreach source {lfw_model.cpp lfw_axi.cpp lfw_model.h} {
        add_files [file join $repo $source] -cflags "-I$repo -ffp-contract=off"
    }
    add_files -tb [file join $repo stress_sim tb_lfw_rtl_stress.cpp] -cflags "-I$repo -ffp-contract=off"
    add_files -tb $fixture
    open_solution -flow_target vivado sol
    set_part xc7z020clg400-1
    create_clock -period 20 -name default
    config_op fadd -impl fabric
    config_op fsub -impl fabric
    close_project
}
exit
