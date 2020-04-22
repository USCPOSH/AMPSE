** Translated using xdm X.X.X-5a04674 on Jan_22_2020_19_06_36_PM
** from /Users/garng/XDM_work/data-model/data-model/build/xdm_bundle/hspice.xml
** to /Users/garng/XDM_work/data-model/data-model/build/xdm_bundle/xyce.xml

*$testbench
.OPTIONS DEVICE TNOM=25  ; converted options using xdm
.options timeint method=gear
.PARAM 
+ WIDTH={0.2*1/fcomp} INV=180n VIN=0 VCM=550m PERIOD={1/fcomp} DELAY=0 NAND_COMP=180n 
+ FCOMP=100e6 RESET_COMP=180n DIF_PAIR=520n INV_COMP=180n NAND_W=180n LATCH_COMP=180n
.INCLUDE 45nm_MGK.pm

*// Library name: POSH_TI_SAR_Parameterized

*// Cell name: NAND2
*// View name: schematic
.SUBCKT NAND2 A B VDD VSS D
M1 net7 B VSS VSS nmos M=20 L=45n W={nand_w}
M0 D A net7 VSS nmos M=20 L=45n W={nand_w}
M3 D B VDD VDD pmos M=10 L=45n W={26/9*nand_w}
M2 D A VDD VDD pmos M=10 L=45n W={26/9*nand_w}
.ENDS NAND2
*// End of subcircuit definition.
*
*// Library name: POSH_TI_SAR_Parameterized
*// Cell name: inverter
*// View name: schematic
.SUBCKT inverter VDD VIN VOUT VSS
M0 VOUT VIN VSS VSS nmos M=10 L=45n W={inv}
M1 VOUT VIN VDD VDD pmos M=10 L=45n W={26/9*inv}
.ENDS inverter
*// End of subcircuit definition.
*
*// Library name: POSH_TI_SAR_Parameterized
*// Cell name: NAND2_comp
*// View name: schematic
.SUBCKT NAND2_comp A B VDD VSS D
M1 net7 B VSS VSS nmos M=20 L=45n W={nand_comp}
M0 D A net7 VSS nmos M=20 L=45n W={nand_comp}
M3 D B VDD VDD pmos M=10 L=45n W={26/9*nand_comp}
M2 D A VDD VDD pmos M=10 L=45n W={26/9*nand_comp}
.ENDS NAND2_comp
*// End of subcircuit definition.
*
*// Library name: POSH_TI_SAR_Parameterized
*// Cell name: comparator_p
*// View name: schematic
.SUBCKT comparator_p CLKC RDY VDD VINN VINP VOUTN VOUTP VSS
M10 VOUTN net16 VSS VSS nmos M=10 L=45n W={inv_comp}
M3 net16 net9 VSS VSS nmos M=10 L=45n W={latch_comp}
M5 net16 CLKC VSS VSS nmos M=5 L=45n W={reset_comp}
M4 net9 CLKC VSS VSS nmos M=5 L=45n W={reset_comp}
M7 VOUTP net9 VSS VSS nmos M=10 L=45n W={inv_comp}
M6 net9 net16 VSS VSS nmos M=10 L=45n W={latch_comp}
M9 VOUTN net16 VDD VDD pmos M=10 L=45n W={26/9*inv_comp}
M0 net12 CLKC VDD VDD pmos M=10 L=45n W={2*dif_pair}
M2 net16 VINN net12 VDD pmos M=5 L=45n W={dif_pair}
M1 net9 VINP net12 VDD pmos M=5 L=45n W={dif_pair}
M8 VOUTP net9 VDD VDD pmos M=10 L=45n W={26/9*inv_comp}
X1 VOUTP VOUTN VDD VSS RDY NAND2_comp
.ENDS comparator_p
*// End of subcircuit definition.
*
*// Library name: POSH_TI_SAR_Parameterized
*// Cell name: tb_comparator_p
*// View name: schematic
X170 VOUTP VDD1 VDD1 VSS net0130 NAND2
X171 VOUTP VDD1 VDD1 VSS net0131 NAND2
X172 VOUTP VDD1 VDD1 VSS net0132 NAND2
X173 VOUTP VDD1 VDD1 VSS net0133 NAND2
X174 VOUTP VDD1 VDD1 VSS net0134 NAND2
X175 VOUTP VDD1 VDD1 VSS net0135 NAND2
X176 VOUTP VDD1 VDD1 VSS net0136 NAND2
X177 VOUTP VDD1 VDD1 VSS net0137 NAND2
X180 VOUTN VDD1 VDD1 VSS net0140 NAND2
X181 VOUTN VDD1 VDD1 VSS net0141 NAND2
X182 VOUTN VDD1 VDD1 VSS net0142 NAND2
X183 VOUTN VDD1 VDD1 VSS net0143 NAND2
X184 VOUTN VDD1 VDD1 VSS net0144 NAND2
X185 VOUTN VDD1 VDD1 VSS net0145 NAND2
X186 VOUTN VDD1 VDD1 VSS net0146 NAND2
X187 VOUTN VDD1 VDD1 VSS net0147 NAND2
V5 VDD1 0 DC 1.1
V4 VINP 0 DC {vcm+VIN/2}
V3 VINN 0 DC {vcm-VIN/2}
V1 VSS 0 DC 0
V0 VDD 0 DC 1.1
V2 EN 0 PULSE(1.1 0 'delay' 500p 500p 'width' 'period')
X150 VDD1 VOUTN net050 VSS inverter
X151 VDD1 VOUTN net051 VSS inverter
X152 VDD1 VOUTN net052 VSS inverter
X153 VDD1 VOUTN net053 VSS inverter
X154 VDD1 VOUTN net054 VSS inverter
X155 VDD1 VOUTN net055 VSS inverter
X156 VDD1 VOUTN net056 VSS inverter
X157 VDD1 VOUTN net057 VSS inverter
X140 VDD1 VOUTP net060 VSS inverter
X141 VDD1 VOUTP net061 VSS inverter
X142 VDD1 VOUTP net062 VSS inverter
X143 VDD1 VOUTP net063 VSS inverter
X144 VDD1 VOUTP net064 VSS inverter
X145 VDD1 VOUTP net065 VSS inverter
X146 VDD1 VOUTP net066 VSS inverter
X147 VDD1 VOUTP net067 VSS inverter
X130 VDD1 RDY net40 VSS inverter
X131 VDD1 RDY net41 VSS inverter
X132 VDD1 RDY net42 VSS inverter
X133 VDD1 RDY net43 VSS inverter
X134 VDD1 RDY net44 VSS inverter
X135 VDD1 RDY net45 VSS inverter
X136 VDD1 RDY net46 VSS inverter
X137 VDD1 RDY net47 VSS inverter
X0 EN RDY VDD VINN VINP VOUTN VOUTP VSS comparator_p
C0130 net0130 VSS C=1p

C0131 net0131 VSS C=1p
C0132 net0132 VSS C=1p
C0133 net0133 VSS C=1p
C0134 net0134 VSS C=1p
C0135 net0135 VSS C=1p
C0136 net0136 VSS C=1p
C0137 net0137 VSS C=1p
C0140 net0140 VSS C=1p
C0141 net0141 VSS C=1p
C0142 net0142 VSS C=1p
C0143 net0143 VSS C=1p
C0144 net0144 VSS C=1p
C0145 net0145 VSS C=1p
C0146 net0146 VSS C=1p
C0147 net0147 VSS C=1p
C050 net050 VSS C=1p

C051 net051 VSS C=1p
C052 net052 VSS C=1p
C053 net053 VSS C=1p
C054 net054 VSS C=1p
C055 net055 VSS C=1p
C056 net056 VSS C=1p
C057 net057 VSS C=1p
C060 net060 VSS C=1p
C061 net061 VSS C=1p
C062 net062 VSS C=1p
C063 net063 VSS C=1p
C064 net064 VSS C=1p
C065 net065 VSS C=1p
C066 net066 VSS C=1p
C067 net067 VSS C=1p
C40 net40 VSS C=1p
C41 net41 VSS C=1p
C42 net42 VSS C=1p
C43 net43 VSS C=1p
C44 net44 VSS C=1p
C45 net45 VSS C=1p
C46 net46 VSS C=1p
C47 net47 VSS C=1p
.TRAN 0.25e-9 400n 0



.PRINT TRAN FORMAT=PROBE v(voutn) v(voutp) v(rdy)  ; aggregated using xdm
