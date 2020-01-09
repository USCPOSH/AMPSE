** Generated for: hspiceD
** Generated on: Jan  8 13:39:43 2020
** Design library name: CAD_modules
** Design cell name: test_VCO_Dtype2_65
** Design view name: schematic
.PARAM rres=2K fnnn=12 fppp=10 vdd=1 vbias=0.9 wpppn=190n wnnn=400n 
+	lastt=60n


.TRAN 1e-12 lastt START=0.0
.OPTION DELMAX=1e-9


.TEMP 25.0
.OPTION
+    ARTIST=2
+    INGOLD=2
+    PARHIER=LOCAL
+    PSF=2
.LIB "TSMC65nm_LIB.sp" TT

** Library name: CAD_modules
** Cell name: diff2sing_v1
** View name: schematic
.subckt diff2sing_v1 b vdd vss in1 in2 o
m2 net3 b vdd vdd pch_lvt l=60e-9 w=2.08e-6 m=1 nf=4 
m0 net8 in1 net3 vdd pch_lvt l=60e-9 w=2.08e-6 m=1 nf=4 
m1 o in2 net3 vdd pch_lvt l=60e-9 w=2.08e-6 m=1 nf=4 
m4 net8 net8 vss vss nch_lvt l=60e-9 w=1.56e-6 m=1 nf=4 
m3 o net8 vss vss nch_lvt l=60e-9 w=1.56e-6 m=1 nf=4 
.ends diff2sing_v1
** End of subcircuit definition.

** Library name: CAD_modules
** Cell name: VCO_type2_65
** View name: schematic
.subckt VCO_type2_65 vdd vss o<1> o<2> o<3> o<4> o<5> o<6> o<7> o<8> op<1> vbias
m12 op<1> o<8> vss vss nmos l=60e-9 w='wnnn' multi=1 nf=fnnn 
m11 o<8> o<7> vss vss nmos l=60e-9 w='wnnn' multi=1 nf=fnnn 
m10 o<7> o<6> vss vss nmos l=60e-9 w='wnnn' multi=1 nf=fnnn
m4 o<6> o<5> vss vss nmos l=60e-9 w='wnnn' multi=1 nf=fnnn 
m3 o<5> o<4> vss vss nmos l=60e-9 w='wnnn' multi=1 nf=fnnn 
m2 o<4> o<3> vss vss nmos l=60e-9 w='wnnn' multi=1 nf=fnnn 
m1 o<3> o<2> vss vss nmos l=60e-9 w='wnnn' multi=1 nf=fnnn 
m0 o<2> o<1> vss vss nmos l=60e-9 w='wnnn' multi=1 nf=fnnn 
m15 op<1> vbias vdd vdd pmos l=60e-9 w='wpppn' multi=1 nf=fppp 
m14 o<7> vbias vdd vdd pmos l=60e-9 w='wpppn' multi=1 nf=fppp
m13 o<8> vbias vdd vdd pmos l=60e-9 w='wpppn' multi=1 nf=fppp
m9 o<6> vbias vdd vdd pmos l=60e-9 w='wpppn' multi=1 nf=fppp 
m8 o<4> vbias vdd vdd pmos l=60e-9 w='wpppn' multi=1 nf=fppp 
m7 o<5> vbias vdd vdd pmos l=60e-9 w='wpppn' multi=1 nf=fppp 
m6 o<2> vbias vdd vdd pmos l=60e-9 w='wpppn' multi=1 nf=fppp 
m5 o<3> vbias vdd vdd pmos l=60e-9 w='wpppn' multi=1 nf=fppp 
.ends VCO_type2_65
** End of subcircuit definition.

** Library name: CAD_modules
** Cell name: VCO_Dtype2_65
** View name: schematic
.subckt VCO_Dtype2_65 vdd vss on<1> on<2> on<3> on<4> on<5> on<6> on<7> on<8> op<1> op<2> op<3> op<4> op<5> op<6> op<7> op<8> vbias
r1<1> on<1> op<2> rres
r1<2> on<2> op<3> rres
r1<3> on<3> op<4> rres
r1<4> on<4> op<5> rres
r1<5> on<5> op<6> rres
r1<6> on<6> op<7> rres
r1<7> on<7> op<8> rres
r0<1> op<1> on<2> rres
r0<2> op<2> on<3> rres
r0<3> op<3> on<4> rres
r0<4> op<4> on<5> rres
r0<5> op<5> on<6> rres
r0<6> op<6> on<7> rres
r0<7> op<7> on<8> rres
xi1 vdd vss op<1> op<2> op<3> op<4> op<5> op<6> op<7> op<8> on<1> vbias VCO_type2_65
xi0 vdd vss on<1> on<2> on<3> on<4> on<5> on<6> on<7> on<8> op<1> vbias VCO_type2_65
.ends VCO_Dtype2_65
** End of subcircuit definition.

** Library name: CAD_modules
** Cell name: test_VCO_Dtype2_65
** View name: schematic
v1 vss 0 DC=0
v0 vdd 0 DC=vdd
xi6<1> vss vdd vss on<1> op<1> oo<1> diff2sing_v1
xi6<2> vss vdd vss on<2> op<2> oo<2> diff2sing_v1
xi6<3> vss vdd vss on<3> op<3> oo<3> diff2sing_v1
xi6<4> vss vdd vss on<4> op<4> oo<4> diff2sing_v1
xi6<5> vss vdd vss on<5> op<5> oo<5> diff2sing_v1
xi6<6> vss vdd vss on<6> op<6> oo<6> diff2sing_v1
xi6<7> vss vdd vss on<7> op<7> oo<7> diff2sing_v1
xi6<8> vss vdd vss on<8> op<8> oo<8> diff2sing_v1
v3 vbias 0 PWL 0 0 lastt vbias TD=0
c1<1> op<1> vss 2e-15
c1<2> op<2> vss 2e-15
c1<3> op<3> vss 2e-15
c1<4> op<4> vss 2e-15
c1<5> op<5> vss 2e-15
c1<6> op<6> vss 2e-15
c1<7> op<7> vss 2e-15
c1<8> op<8> vss 2e-15
c0<1> on<1> vss 2e-15
c0<2> on<2> vss 2e-15
c0<3> on<3> vss 2e-15
c0<4> on<4> vss 2e-15
c0<5> on<5> vss 2e-15
c0<6> on<6> vss 2e-15
c0<7> on<7> vss 2e-15
c0<8> on<8> vss 2e-15
xi0 vdd vss on<1> on<2> on<3> on<4> on<5> on<6> on<7> on<8> op<1> op<2> op<3> op<4> op<5> op<6> op<7> op<8> vbias VCO_Dtype2_65

.IC  on<1> 0
.MEAS TRAN pwr  AVG I(v1)
.MEAS TRAN_CONT cont_vout1 find v(vbias) when v(oo<1>)=0.8 rise=1
.MEAS TRAN_CONT cont_tout1 when v(oo<1>)=0.8 rise=1

.END
