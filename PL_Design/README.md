# PYNQ-Z2 Vivado project

Use Vivado **2022.2** with Zynq-7000 device support. The target is
`xc7z020clg400-1`, board `tul.com.tw:pynq-z2:part0:1.0`, at 50 MHz.
This design uses the three floating-point LFW HLS IPs.

## Included files

- `lfw_pynq_z2_build/lfw_pynq_z2.xpr`: Vivado project.
- `lfw_pynq_z2_build/lfw_pynq_z2.srcs/`: block design and IP configuration.
- `lfw_pynq_z2_build/lfw_pynq_z2.gen/sources_1/bd/lfw_system/hdl/lfw_system_wrapper.v`: top-level wrapper.
- `lfw_pynq_z2_build/lfw_system.xsa`: existing hardware handoff archive,
  **without a bitstream**. It does not establish successful implementation or timing closure.
- `create_lfw_pl.tcl`: alternative project-generation and implementation flow.

## Prepare the IP repositories

From the repository root, unpack the committed archives:

```bash
unzip -o hls_ip/lfw_conv1/export.zip -d hls_ip/lfw_conv1/ip
unzip -o hls_ip/lfw_conv2/export.zip -d hls_ip/lfw_conv2/ip
unzip -o hls_ip/lfw_readout/export.zip -d hls_ip/lfw_readout/ip
```

Each resulting directory must contain `component.xml` directly inside it.
In Vivado, use **Settings → IP → Repository → Add** to register these paths
(replace `<repo>` with the absolute path of your checkout):

| IP | Repository directory |
| --- | --- |
| `xilinx.com:hls:lfw_conv1_axi:1.0` | `<repo>/hls_ip/lfw_conv1/ip` |
| `xilinx.com:hls:lfw_conv2_axi:1.0` | `<repo>/hls_ip/lfw_conv2/ip` |
| `xilinx.com:hls:lfw_readout_axi:1.0` | `<repo>/hls_ip/lfw_readout/ip` |

The saved XPR uses relative paths to these directories. Built-in Xilinx IPs
(PS7, AXI DMA, SmartConnect, reset, concat and constant) come with Vivado.
The separate `hls_ip/export.zip` is another conv2 archive; use the three
per-IP archives listed above for this setup.

## Install the board preset and open the project

Place the PYNQ-Z2 board files so this file exists:
`<repo>/PL_Design/board_files/pynq-z2/A.0/board.xml`.
The board files are a local prerequisite and are not bundled here.
Before opening the project, register them in the Vivado Tcl Console:

```tcl
set repo /absolute/path/to/spikeface
set_param board.repoPaths [list $repo/PL_Design/board_files]
open_project $repo/PL_Design/lfw_pynq_z2_build/lfw_pynq_z2.xpr
set_property ip_repo_paths [list $repo/hls_ip/lfw_conv1/ip $repo/hls_ip/lfw_conv2/ip $repo/hls_ip/lfw_readout/ip] [current_project]
update_ip_catalog
```

If board files are installed elsewhere, register that directory instead.
Open `lfw_system` under **Sources → Design Sources**, check **Report IP Status**,
validate the block design, and generate output products. The top module is
`lfw_system_wrapper`. Run synthesis, implementation, inspect timing, and generate
the bitstream when required. Generated caches and run outputs are ignored.

## Recreate the project with Tcl

The generator defaults to local HLS GUI export paths. To use the unpacked
archives above instead, run from the repository root:

```bash
export SPIKEFACE_BOARD_REPO="$PWD/PL_Design/board_files"
export SPIKEFACE_CONV1_IP="$PWD/hls_ip/lfw_conv1/ip"
export SPIKEFACE_CONV2_IP="$PWD/hls_ip/lfw_conv2/ip"
export SPIKEFACE_READOUT_IP="$PWD/hls_ip/lfw_readout/ip"
vivado -mode batch -source PL_Design/create_lfw_pl.tcl -tclargs bd /tmp/lfw_pynq_z2_fresh
```

Choose a fresh output directory **outside `PL_Design`**. The script refuses to
overwrite an existing project. Mode `bd` creates and validates the design;
mode `bit` additionally runs implementation, checks timing, generates the
bitstream, and exports hardware. Use a fresh output directory for either mode.

For newly exported GUI IPs, the alternative repository paths are
`<repo>/newarch_HLS/gui_projects/{lfw_conv1_axi,lfw_conv2_axi,lfw_readout_axi}/sol/impl/ip`.
Complete HLS verification and export each IP before using those directories.
