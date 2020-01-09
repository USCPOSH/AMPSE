** Generated for: hspiceD
** Generated on: Jan  9 00:00:46 2020
** Design library name: brbcad
** Design cell name: test_class_ab
** Design view name: schematic
.PARAM vincm=493.6m fbias=10 lbias=500n fin=10 fp=10 lin=100n lp=100n 
+	cload=0 vcmo=500m mamp=2


.PROBE AC
+    V(cap_cal) VP(cap_cal)
+    V(rout2) VP(rout2)
+    V(rout1) VP(rout1)
+    V(voutn2) VP(voutn2)
+    V(voutn1) VP(voutn1)
.AC DEC 10 1e3 1e12

.OP all 0

.TEMP 25.0
.OPTION
+    ARTIST=2
+    INGOLD=2
+    PARHIER=LOCAL
+    PSF=2
.INCLUDE "/shares/techfile/PTM/45nm/Techfile_45nm.scs"

** Library name: brbcad
** Cell name: differ_amplifier_gain10
** View name: schematic
.subckt differ_amplifier_gain10 vdd vss vin vip voutp
m20 net22 net22 vss vss nmos L=100e-9 W=300e-9 M=1
m15 net013 vip net13 vss nmos L=100e-9 W=300e-9 M=1
m18 net13 net22 vss vss nmos L=1e-6 W=600e-9 M=1
m0 voutp vin net13 vss nmos L=100e-9 W=300e-9 M=1
m19 net22 net22 vdd vdd pmos L=100e-9 W=600e-9 M=1
m17 net013 net013 vdd vdd pmos L=100e-9 W=600e-9 M=1
m8 voutp net013 vdd vdd pmos L=100e-9 W=600e-9 M=1
.ends differ_amplifier_gain10
** End of subcircuit definition.

** Library name: brbcad
** Cell name: classab
** View name: schematic
.subckt classab vdd vss vin vip voutn voutp
m15 voutp vip vss vss nmos L=_par0 W=300e-9 M=_par1
m0 voutn vin vss vss nmos L=_par0 W=300e-9 M=_par1
m17 voutp voutp vdd vdd pmos L=_par2 W=600e-9 M=_par3
m8 voutn voutp vdd vdd pmos L=_par2 W=600e-9 M=_par3
.ends classab
** End of subcircuit definition.

** Library name: brbcad
** Cell name: test_class_ab
** View name: schematic
i29 rout1 rout2 AC '1A'
c3 voutn2 vss cload
c0 voutn1 vss cload
v18 net056 vss DC=500e-3 AC 1
v14 net034 vss DC=vcmo
v9 vcm vss DC=vincm AC 0
v8 vdd2 0 DC=1
v16 net049 vss DC=vcmo
v2 net07 vss DC=0 AC 500e-3 0
v1 vss 0 DC=0
v0 vdd 0 DC=1
r10 net056 cap_cal 1e3
r6 rout1 net04 1e6
r5 net06 voutn2 1e6
r4 voutn1 net06 1e6
r7 net04 rout2 1e6
e2 vip vcm VCVS net07 vss 1
e1 vin vcm VCVS vss net07 1
e12 outd VSS VCVS voutn1 voutn2 1
e13 outr VSS VCVS rout1 rout2 1
xi23 vdd vss net034 net06 net037 differ_amplifier_gain10
xi26 vdd2 vss net049 net04 net011 differ_amplifier_gain10
xi33 vdd2 vss cap_cal cap_cal net052 net064 classab _par0=lin _par1=fin _par2=lp _par3=fp
xi28 vdd2 vss vcm vcm rout1 net046 classab _par0=lin _par1='fin*mamp' _par2=lp _par3='fp*mamp'
xi27 vdd2 vss vcm vcm rout2 net045 classab _par0=lin _par1='fin*mamp' _par2=lp _par3='fp*mamp'
xi21 vdd vss vip vin voutn2 net053 classab _par0=lin _par1='fin*mamp' _par2=lp _par3='fp*mamp'
xi20 vdd vss vin vip voutn1 net054 classab _par0=lin _par1='fin*mamp' _par2=lp _par3='fp*mamp'
m10 rout1 net011 vdd2 vdd2 pmos L=lbias W=600e-9 M=fbias
m9 rout2 net011 vdd2 vdd2 pmos L=lbias W=600e-9 M=fbias
m2 voutn2 net037 vdd vdd pmos L=lbias W=600e-9 M=fbias
m6 voutn1 net037 vdd vdd pmos L=lbias W=600e-9 M=fbias
m5 voutn1 net037 vss vss nmos L=lbias W=300e-9 M=fbias
m4 voutn2 net037 vss vss nmos L=lbias W=300e-9 M=fbias
m8 rout2 net011 vss vss nmos L=lbias W=300e-9 M=fbias
m7 rout1 net011 vss vss nmos L=lbias W=300e-9 M=fbias

//gain
.MEAS AC gain MAX vdb(outd)
//pole1
.MEAS AC pole1 WHEN vp(outd) = -45
//zero
.MEAS AC zero WHEN vp(outd) = -135
//Rout
.MEAS AC ROUT MAX vm(outr)
//GM
.MEAS AC gmax MAX vm(outd)
.MEAS AC GM PARAM = PAR('gmax/rout')
//power
.MEAS DC pwr AVG I(v0)
//SWINGP
.MEAS DC vovp PARAM = 'lv10(xi20.M8)'
.MEAS DC vdsp PARAM = 'lx3(xi20.M8)'
.MEAS DC swingp PARAM=PAR('-vdsp-vovp')
//SWINGN
.MEAS DC vovn PARAM = 'lv10(xi20.M0)'
.MEAS DC vdsn PARAM = 'lx3(xi20.M0)'
.MEAS DC swingn PARAM=PAR('vdsn-vovn')

.MEAS DC cmo AVG V(rout1)

//Cin
.MEAS AC Cin3db WHEN vdb(cap_cal) =-3
.MEAS AC Cin PARAM=PAR('1/2/3.1415/1000/Cin3db')
//Cout
.MEAS AC cout3db WHEN vp(outr) = 135
.MEAS AC COUT PARAM=PAR('1/2/3.1415/ROUT/cout3db')

.END
