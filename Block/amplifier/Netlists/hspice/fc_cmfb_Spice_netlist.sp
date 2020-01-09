** Generated for: hspiceD
** Generated on: Jan  8 23:49:10 2020
** Design library name: brbcad
** Design cell name: test_folded_CMFB
** Design view name: schematic
.PARAM lbias=500n fbias=10 lbn=100n lbp=100n lin1=100n lin2=100n ltn=100n 
+	ltp=100n cload=0 vcmo=500m mamp=1 fbn=100 fbp=100 fin1=100 fin2=100 
+	ftn1=100 ftn2=100 ftp1=100 ftp2=100


.PROBE AC
+    V(rout2) VP(rout2)
+    V(rout1) VP(rout1)
+    V(voutp) VP(voutp)
+    V(voutn) VP(voutn)
+    V(cap_cal) VP(cap_cal)
.PROBE NOISE
+    V(rout2) VP(rout2)
+    V(rout1) VP(rout1)
+    V(voutp) VP(voutp)
+    V(voutn) VP(voutn)
+    V(cap_cal) VP(cap_cal)
.AC DEC 10 1e3 1e12

**.NOISE V(voutn) V2 100

.PRINT NOISE ONOISE INOISE

.TEMP 25.0
.OPTION
+    ARTIST=2
+    INGOLD=2
+    PARHIER=LOCAL
+    PSF=2
.INCLUDE "/shares/techfile/PTM/45nm/Techfile_45nm.scs"

** Library name: brbcad
** Cell name: folded3
** View name: schematic
.subckt folded3 vdd vss vbn vbp vin vip voutn voutp vtn vtp
m13 net35 vtn vss vss nmos L=_par0 W=300e-9 M=_par1
m15 voutp vbn net35 vss nmos L=_par2 W=300e-9 M=_par3
m4 net33 vtn vss vss nmos L=_par0 W=600e-9 M=_par4
m3 net24 vip net33 vss nmos L=_par5 W=300e-9 M=_par6
m2 net32 vin net33 vss nmos L=_par5 W=300e-9 M=_par6
m1 net34 vtn vss vss nmos L=_par0 W=300e-9 M=_par1
m0 voutn vbn net34 vss nmos L=_par2 W=300e-9 M=_par3
m14 net35 vip net20 vdd pmos L=_par7 W=600e-9 M=_par8
m12 net20 vtp vdd vdd pmos L=_par9 W=1.2e-6 M=_par10
m11 net34 vin net20 vdd pmos L=_par7 W=600e-9 M=_par8
m17 voutp vbp net24 vdd pmos L=_par11 W=600e-9 M=_par12
m8 voutn vbp net32 vdd pmos L=_par11 W=600e-9 M=_par12
m7 net32 vtp vdd vdd pmos L=_par9 W=600e-9 M=_par13
m16 net24 vtp vdd vdd pmos L=_par9 W=600e-9 M=_par13
.ends folded3
** End of subcircuit definition.

** Library name: brbcad
** Cell name: bias_cascode
** View name: schematic
.subckt bias_cascode vdd vss vbn vbp vcm vtn
m4 net33 vtn vss vss nmos L=_par0 W=600e-9 M=_par1
m2 vbn vcm net33 vss nmos L=_par2 W=300e-9 M=_par3
m1 vbp vtn vss vss nmos L=_par0 W=300e-9 M=_par4
m0 vtn vbn vbp vss nmos L=_par5 W=300e-9 M=_par6
m12 net20 vtn vdd vdd pmos L=_par7 W=1.2e-6 M=_par8
m11 vbp vcm net20 vdd pmos L=_par9 W=600e-9 M=_par10
m8 vtn vbp vbn vdd pmos L=_par11 W=600e-9 M=_par12
m7 vbn vtn vdd vdd pmos L=_par7 W=600e-9 M=_par13
.ends bias_cascode
** End of subcircuit definition.

** Library name: brbcad
** Cell name: differ_amplifier_gain10
** View name: schematic
.subckt differ_amplifier_gain10 vdd vss vin vip voutn voutp
m20 net22 net22 vss vss nmos L=100e-9 W=50e-9 M=20
m15 voutp vip net13 vss nmos L=100e-9 W=50e-9 M=20
m18 net13 net22 vss vss nmos L=100e-9 W=100e-9 M=20
m0 voutn vin net13 vss nmos L=100e-9 W=50e-9 M=20
m19 net22 net22 vdd vdd pmos L=100e-9 W=100e-9 M=20
m17 voutp voutp vdd vdd pmos L=100e-9 W=100e-9 M=20
m8 voutn voutp vdd vdd pmos L=100e-9 W=100e-9 M=20
.ends differ_amplifier_gain10
** End of subcircuit definition.

** Library name: brbcad
** Cell name: test_folded_CMFB
** View name: schematic
xi12 vdd2 vss vbn vbp vcm vcm rout1 rout2 vtn vtn folded3 _par0=ltn _par1='ftn1*mamp' _par2=lbn _par3='fbn*mamp' _par4='ftn2*mamp' _par5=lin2 _par6='fin2*mamp' _par7=lin1 _par8='fin1*mamp' _par9=ltp _par10='ftp2*mamp' _par11=lbp _par12='fbp*mamp' _par13='ftp1*mamp'
xi13 vdd2 vss vbn2 vbp2 cap_cal cap_cal net015 net014 vtn2 vtn2 folded3 _par0=ltn _par1='ftn1*mamp' _par2=lbn _par3='fbn*mamp' _par4='ftn2*mamp' _par5=lin2 _par6='fin2*mamp' _par7=lin1 _par8='fin1*mamp' _par9=ltp _par10='ftp2*mamp' _par11=lbp _par12='fbp*mamp' _par13='ftp1*mamp'
xi5 vdd vss vbn vbp vin vip voutn voutp vtn vtn folded3 _par0=ltn _par1='ftn1*mamp' _par2=lbn _par3='fbn*mamp' _par4='ftn2*mamp' _par5=lin2 _par6='fin2*mamp' _par7=lin1 _par8='fin1*mamp' _par9=ltp _par10='ftp2*mamp' _par11=lbp _par12='fbp*mamp' _par13='ftp1*mamp'
c5 vtn2 vss 1e-6
c4 vbp2 vss 1e-6
c3 vbn2 vss 1e-6
c1 voutp vss cload
c0 voutn vss cload
v16 net050 vss DC=500e-3
v13 net039 vss DC=vcmo
v10 net019 vss DC=500e-3 AC 1
v9 vcm vss DC=500e-3
v8 vdd2 0 DC=1
v14 net031 vss DC=vcmo
v2 net07 vss DC=0 AC 500e-3 0
v1 vss 0 DC=0
v0 vdd 0 DC=1
r8 rout1 net012 1e6
r7 net03 voutn 1e6
r2 cap_cal net019 1e3
r6 voutp net03 1e6
r9 net012 rout2 1e6
e2 vin vcm VCVS net07 vss 1
e1 vip vcm VCVS vss net07 1
e12 outd VSS VCVS voutp voutn 1
e13 outr VSS VCVS rout1 rout2 1
i10 rout1 rout2 AC 1 0
xi26 vdd2 vss vbn2 vbp2 net050 vtn2 bias_cascode _par0=ltn _par1='ftn2*mamp' _par2=lin2 _par3='(2*fin2)*mamp' _par4='(2*ftn1)*mamp' _par5=lbn _par6='(2*fbn)*mamp' _par7=ltp _par8='ftp2*mamp' _par9=lin1 _par10='(2*fin1)*mamp' _par11=lbp _par12='(2*fbp)*mamp' _par13='(2*ftp1)*mamp'
xi11 vdd2 vss vbn vbp vcm vtn bias_cascode _par0=ltn _par1='ftn2*mamp' _par2=lin2 _par3='(2*fin2)*mamp' _par4='(2*ftn1)*mamp' _par5=lbn _par6='(2*fbn)*mamp' _par7=ltp _par8='ftp2*mamp' _par9=lin1 _par10='(2*fin1)*mamp' _par11=lbp _par12='(2*fbp)*mamp' _par13='(2*ftp1)*mamp'
xi16 vdd vss net039 net012 net018 net017 differ_amplifier_gain10
xi23 vdd vss net031 net03 net052 net029 differ_amplifier_gain10
m4 voutn net052 vss vss nmos L=lbias W=300e-9 M=fbias
m5 voutp net052 vss vss nmos L=lbias W=300e-9 M=fbias
m0 rout1 net018 vss vss nmos L=lbias W=300e-9 M=fbias
m1 rout2 net018 vss vss nmos L=lbias W=300e-9 M=fbias
m7 rout2 net018 vdd vdd pmos L=lbias W=600e-9 M=fbias
m3 rout1 net018 vdd vdd pmos L=lbias W=600e-9 M=fbias
m6 voutp net052 vdd vdd pmos L=lbias W=600e-9 M=fbias
m2 voutn net052 vdd vdd pmos L=lbias W=600e-9 M=fbias

**gain

.MEAS AC gain MAX vdb(outd)

**pole1

.MEAS AC pole1 WHEN vp(outd) = -45

**pole2

.MEAS AC pole2 WHEN vp(outd) = -135

**Rout

.MEAS AC ROUT MAX vm(outr)

**GM

.MEAS AC gmax MAX vm(outd)

.MEAS AC GM PARAM = PAR('gmax/rout')

**power

.MEAS DC pwr AVG i(v0)



.MEAS DC cmo AVG V(voutn)

**SWINGP

.MEAS DC vovp PARAM = 'lv10(xi5.m8)'

.MEAS DC vdsp PARAM = 'lx3(xi5.m8)'

.MEAS DC swingp PARAM = PAR('-vdsp-vovp')

**SWINGN

.MEAS DC vovn PARAM = 'lv10(xi5.m0)'

.MEAS DC vdsn PARAM = 'lx3(xi5.m0)'

.MEAS DC swingn PARAM = PAR('vdsn-vovn')



**tail transistors

**SWING of M7

.MEAS DC vov7 PARAM = 'lv10(xi5.m7)'

.MEAS DC vds7 PARAM = 'lx3(xi5.m7)'

.MEAS DC swing7 PARAM = PAR('-vds7-vov7')

**SWING of M1

.MEAS DC vovn1 PARAM = 'lv10(xi5.m1)'

.MEAS DC vdsn1 PARAM = 'lx3(xi5.m1)'

.MEAS DC swingn1 PARAM = PAR('vdsn1-vovn1')



**SWING of M14

.MEAS DC vov14 PARAM = 'lv10(xi5.m14)'

.MEAS DC vds14 PARAM = 'lx3(xi5.m14)'

.MEAS DC swing14 PARAM = PAR('-vds14-vov14')

**SWING of M4

.MEAS DC vovn4 PARAM = 'lv10(xi5.m4)'

.MEAS DC vdsn4 PARAM = 'lx3(xi5.m4)'

.MEAS DC swingn4 PARAM = PAR('vdsn4-vovn4')



**Cin

.MEAS AC Cin3db WHEN vdb(cap_cal) =-3

.MEAS AC Cin PARAM = PAR('1/2/3.1415/1000/Cin3db')

**Cout

.MEAS AC cout3db WHEN vp(outr) = 135

.MEAS AC COUT PARAM = PAR('1/2/3.1415/ROUT/cout3db')



**noise

**.MEAS NOISE invn RMS INOISE **multiply by sqrt of pole1 in python to get the actual irn

**.MEAS NOISE irn PARAM = PAR('pole1*invn')
.END

