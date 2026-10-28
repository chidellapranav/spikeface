# Run only C simulation and C synthesis in the existing GUI projects.
set repo [file dirname [file normalize [info script]]]
cd [file join $repo gui_projects]
foreach top {lfw_conv2_axi lfw_readout_axi} {
    open_project $top
    set_top $top
    open_solution sol
    csim_design -clean
    csynth_design
    close_project
}
exit
