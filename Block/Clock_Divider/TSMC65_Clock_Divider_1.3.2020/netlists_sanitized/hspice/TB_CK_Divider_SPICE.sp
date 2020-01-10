** Generated for: hspiceD
** Generated on: Jan  9 19:47:59 2020
** Design library name: ClockDivider
** Design cell name: TB_CK_Divider
** Design view name: schematic
.PARAM res=580 vctrl=1 w_int=4u w_ip=1u w_tail=2.5u ain=100m fin=12G 
+	vbias=500m


.PROBE TRAN
+    V(d)
+    I(v0)
.OP all 1n

.TRAN 10e-12 10e-9 START=0.0

.TEMP 25.0
.OPTION
+    ARTIST=2
+    INGOLD=2
+    PARHIER=LOCAL
+    PSF=2
.LIB "TSMC65nm_LIB.sp" TT

** Library name: DAC_2018_May_TSMC65_ss_soumya
** Cell name: divby2_cml_tail
** View name: schematic
.subckt divby2_cml_tail vtail gnd idiv_ctrl<0> idiv_ctrl<1> idiv_ctrl<2> vdd vdiv
m15 vtail vdiv net18 gnd nmos l=240e-9 w=3e-6 multi=2 nf=1 
m14 vtail vdiv net19 gnd nmos l=240e-9 w=3e-6 multi=4 nf=1 
m5 net18 idiv_ctrl<0> gnd gnd nmos l=240e-9 w=3e-6 multi=2 nf=1 
m0 vtail vdiv net20 gnd nmos l=240e-9 w=3e-6 multi=8 nf=1 
m4 net19 idiv_ctrl<1> gnd gnd nmos l=240e-9 w=3e-6 multi=4 nf=1 
m11 vtail gnd gnd gnd nmos l=240e-9 w=3e-6 multi=2 nf=1
m8 vdiv vdiv net013 gnd nmos l=240e-9 w=3e-6 multi=2 nf=1 
m6 net20 idiv_ctrl<2> gnd gnd nmos l=240e-9 w=3e-6 multi=8 nf=1 
m9 vdiv gnd gnd gnd nmos l=240e-9 w=3e-6 multi=2 nf=1 
m12 gnd gnd gnd gnd nmos l=240e-9 w=3e-6 multi=8 nf=1
m7 net013 vdd gnd gnd nmos l=240e-9 w=3e-6 multi=2 nf=1 
.ends divby2_cml_tail
** End of subcircuit definition.

** Library name: DAC_2018_May_TSMC65_ss_soumya
** Cell name: divby2_cml_latch
** View name: schematic
.subckt divby2_cml_latch clk clkb d db p pb vtail gnd rst rstb vdd
m43 gnd gnd gnd gnd nmos l=60e-9 w=2.5e-6 multi=28 nf=1 
m42 gnd gnd gnd gnd nmos l=60e-9 w=2.125e-6 multi=4 nf=1 
m39 gnd gnd gnd gnd nmos l=60e-9 w=2.125e-6 multi=2 nf=1 
m38 gnd gnd gnd gnd nmos l=60e-9 w=2.5e-6 multi=2 nf=1 
m37 net022 gnd gnd gnd nmos l=60e-9 w=2.5e-6 multi=2 nf=1 
m36 vtail gnd gnd gnd nmos l=60e-9 w=2.125e-6 multi=2 nf=1 
m32 p gnd vtail gnd nmos l=60e-9 w=2.125e-6 multi=2 nf=1 
m35 net1 gnd gnd gnd nmos l=60e-9 w=2.125e-6 multi=2 nf=1 
m24 pb rst vtail gnd nmos l=60e-9 w=2.125e-6 multi=2 nf=1 
m25 net022 rstb vtail gnd nmos l=60e-9 w='w_tail*1' multi=4 nf=1 
m34 net04 gnd gnd gnd nmos l=60e-9 w=2.125e-6 multi=4 nf=1 
m17 pb db net04 gnd nmos l=60e-9 w='w_ip*1' multi=17 nf=1 
m5 net04 d p gnd nmos l=60e-9 w='w_ip*1' multi=17 nf=1 
m41 net1 p pb gnd nmos l=60e-9 w='w_ip*1' multi=17 nf=1 
m3 p pb net1 gnd nmos l=60e-9 w='w_ip*1' multi=17 nf=1 
m0 net04 clk net022 gnd nmos l=60e-9 w='w_int*1' multi=5 nf=1 
m33 vtail gnd gnd gnd nmos l=60e-9 w=2.125e-6 multi=2 nf=1 
m30 net1 clkb net022 gnd nmos l=60e-9 w='w_int*1' multi=5 nf=1 
r3  gnd gnd res

**Series configuration of R0
r0_1__dmy0  vdd r0_1__dmy0 gnd  res 
r0_2__dmy0  r0_1__dmy0 r0_2__dmy0  res 
r0_3__dmy0  r0_2__dmy0 r0_3__dmy0  res 
r0_4__dmy0  r0_3__dmy0 r0_4__dmy0  res 
r0_5__dmy0  r0_4__dmy0 r0_5__dmy0  res 
r0_6__dmy0  r0_5__dmy0 r0_6__dmy0  res 
r0_7__dmy0  r0_6__dmy0 r0_7__dmy0  res 
r0_8__dmy0  r0_7__dmy0 pb  res
**End of R0

**Series configuration of R2
r2_1__dmy0  vdd r2_1__dmy0 res 
r2_2__dmy0  r2_1__dmy0 r2_2__dmy0  res 
r2_3__dmy0  r2_2__dmy0 r2_3__dmy0  res 
r2_4__dmy0  r2_3__dmy0 r2_4__dmy0  res 
r2_5__dmy0  r2_4__dmy0 r2_5__dmy0  res
r2_6__dmy0  r2_5__dmy0 r2_6__dmy0  res
r2_7__dmy0  r2_6__dmy0 r2_7__dmy0  res 
r2_8__dmy0  r2_7__dmy0 p  res 
**End of R2

.ends divby2_cml_latch
** End of subcircuit definition.

** Library name: DAC_2018_May_TSMC65_ss_soumya
** Cell name: divby2_cml_DFF
** View name: schematic
.subckt divby2_cml_DFF clk clkb d db p pb vtail<1> vtail<0> gnd rst rstb vdd
xi1 clk clkb p pb db d vtail<0> gnd rst rstb vdd divby2_cml_latch
xi0 clkb clk d db p pb vtail<1> gnd rst rstb vdd divby2_cml_latch
.ends divby2_cml_DFF
** End of subcircuit definition.

** Library name: DAC_2018_May_TSMC65_ss_soumya
** Cell name: divby2_cml
** View name: schematic
.subckt divby2_cml clk clkb d db p pb gnd idiv_ctrl<0> idiv_ctrl<1> idiv_ctrl<2> rst rstb vdd vdiv
xi3 vtail<0> gnd idiv_ctrl<0> idiv_ctrl<1> idiv_ctrl<2> vdd vdiv divby2_cml_tail
xi2 vtail<1> gnd idiv_ctrl<0> idiv_ctrl<1> idiv_ctrl<2> vdd vdiv divby2_cml_tail
m3 gnd vdiv gnd gnd nmos l=800e-9 w=4e-6 m=15 nf=1 
m2 gnd vdiv gnd gnd nmos l=800e-9 w=6e-6 m=25 nf=1 
m1 gnd vdiv gnd gnd nmos l=60e-9 w=4e-6 m=40 nf=1 
m0 gnd vdiv gnd gnd nmos l=60e-9 w=4e-6 m=40 nf=1 
xi4 clk clkb d db p pb vtail<1> vtail<0> gnd rst rstb vdd divby2_cml_DFF
.ends divby2_cml
** End of subcircuit definition.

** Library name: ClockDivider
** Cell name: TB_CK_Divider
** View name: schematic
xi1 clk clkb d db p pb gnd idiv_ctrl<0> idiv_ctrl<1> idiv_ctrl<2> gnd vdd vdd vdiv divby2_cml
v8 net1 0 DC=vbias
v5 vdiv 0 DC=1
v4 idiv_ctrl<0> 0 DC=vctrl
v3 idiv_ctrl<1> 0 DC=1
v2 idiv_ctrl<2> 0 DC=1
v1 gnd 0 DC=0
v0 vdd 0 DC=1
v7 clkb net1 SIN 0 '-ain' fin
v6 clk net1 SIN 0 ain fin
.END
