# Bundled PYNQ-Z2 board support

`pynq-z2/A.0/` contains the TUL PYNQ-Z2 board definition:

- `board.xml`: board identity, interfaces, and FPGA device.
- `part0_pins.xml`: board pin mapping.
- `preset.xml`: Zynq processing-system and DDR configuration.

These files were copied unchanged from the owner's working board repository in
the successful Windows Vivado 2022.2 CMake block-design test. They identify TUL
(`tul.com.tw`) as the board vendor and `xc7z020clg400-1` as the FPGA. Original
vendor identifiers are preserved; these files are not authored by SpikeFace.
Their original download revision was not recorded. They should be treated as a
fixed project dependency rather than a claim to track the latest vendor release.

CMake uses this directory automatically. The Vivado Tcl script registers it
before creating the project; no copying into the Vivado installation is needed.
Use `SPIKEFACE_BOARD_REPO` only to intentionally select a different repository.

The vendor XML references an optional `pynq_z2.jpg` GUI thumbnail which was not
present in the working board repository. The three XML files provide the design
and PS preset information; no board illustration is bundled.

Keep the DDR preset unchanged unless the replacement is validated for the
physical board. A successful block-design build does not verify board DDR or
hardware inference. See the root README for build and deployment instructions.
