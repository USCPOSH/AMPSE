$hspice version of comparator

.param C=1.5p dif_pair=530n inv=160n inv_comp=190n latch_comp=177n 
+    nand_comp=191n nand_w=160n nor_w=286n reset_comp=180n switch=1.06u 
+    fs=100x fin=31/64*fs delay=5n width=0.2*1/fs period=1/fs periodx10=period*10
.inc /Users/garng/XDM_work/data-model/netlists/forUSC/Hspice/Library/45nm_MGK.pm

*// Library name: POSH_TI_SAR_Parameterized
*// Cell name: NAND2_comp
*// View name: schematic
.subckt NAND2_comp A B VDD VSS D
    M1 net7 B VSS VSS nmos w=nand_comp l=45n m=20
    M0 D A net7 VSS nmos w=nand_comp l=45n m=20
    M3 D B VDD VDD pmos w=26/9*nand_comp l=45n m=10
    M2 D A VDD VDD pmos w=26/9*nand_comp l=45n m=10
.ends NAND2_comp
*// End of subcircuit definition.

*// Library name: POSH_TI_SAR_Parameterized
*// Cell name: comparator_p
*// View name: schematic
M10 VOUTN net16 VSS VSS nmos w=inv_comp l=45n m=10
M3 net16 net9 VSS VSS nmos w=latch_comp l=45n m=10
M5 net16 CLKC VSS VSS nmos w=reset_comp l=45n m=5
M4 net9 CLKC VSS VSS nmos w=reset_comp l=45n m=5
M7 VOUTP net9 VSS VSS nmos w=inv_comp l=45n m=10
M6 net9 net16 VSS VSS nmos w=latch_comp l=45n m=10
M9 VOUTN net16 VDD VDD pmos w=26/9*inv_comp l=45n m=10
M0 net12 CLKC VDD VDD pmos w=2*dif_pair l=45n m=10
M2 net16 VINN net12 VDD pmos w=dif_pair l=45n m=5
M1 net9 VINP net12 VDD pmos w=dif_pair l=45n m=5
M8 VOUTP net9 VDD VDD pmos w=26/9*inv_comp l=45n m=10
X1 VOUTP VOUTN VDD VSS RDY NAND2_comp

VDD VDD 0 1.1
VSS VSS 0 0
V1 CLKC 0 pulse 1.1 0 'delay' 500p 500p 'width' 'period'
V2 VINN 0 PWL 0 0 30e-9 0 31e-9 1.1 60e-9 1.1
V3 VINP 0 PWL 0 0 10e-9 0 11e-9 1.1 40e-9 1.1 41e-9 0 60e-9 0
C1 VOUTN 0 1p
C2 VOUTP 0 1p
C3 RDY 0 1p

.tran 10e-12 60n start=0 
.print tran v(clkc) v(vinn) v(vinp) v(voutn) v(voutp) v(rdy)
