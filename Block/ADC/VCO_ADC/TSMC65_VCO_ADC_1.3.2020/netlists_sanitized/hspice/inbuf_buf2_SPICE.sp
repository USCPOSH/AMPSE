** Generated for: hspiceD
** Generated on: Jan  8 15:54:29 2020
** Design library name: CAD_modules
** Design cell name: test_VCO_analogin_cmlbuffer_v3
** Design view name: schematic
.PARAM resout=12.5k multi=4 fing_in=1 l_ttt=400n fing_ttt=4 vcm=0.25 dvv=0 
+	wpppp=700n fpppp=2



.AC DEC 100 1.0 1e12

.DC dvv -300e-3 300e-3 200e-3

.NOISE V(oxn) v4 0

.PRINT NOISE ONOISE INOISE

.TRAN 10e-12 540e-9 START=0.0
.OPTION DELMAX=100e-9


.TEMP 25.0
.OPTION
+    ARTIST=2
+    INGOLD=2
+    PARHIER=LOCAL
+    PSF=2
.LIB "TSMC65nm_LIB" TT

** Library name: CAD_modules
** Cell name: VCO_analogin_cmlbuffer_v2
** View name: schematic
.subckt VCO_analogin_cmlbuffer_v2 vdd vss on op vin vip
m12 vdd vin net014 vss nmos l=l_ttt w='600e-9' m=1 nf=_par0 
m23 on vin vss vss nmos l=l_ttt w=600e-9 m=_par1 nf=1 
m13 vdd vip net014 vss nmos l=l_ttt w='600e-9' m=1 nf=_par0 
m4 vdd vin on vss nmos l=l_ttt w='600e-9' m=_par1 nf=_par0 
m22 op vip vss vss nmos l=l_ttt w=600e-9 m=_par1 nf=1 
m11 net014 net014 vss vss nmos l=l_ttt w='600e-9' m=2 nf=_par2 
m19 on net014 vss vss nmos l=l_ttt w='600e-9' m=_par1 nf=_par2 
m18 op net014 vss vss nmos l=l_ttt w='600e-9' m=_par1 nf=_par2 
m1 vdd vip op vss nmos l=l_ttt w='600e-9' m=_par1 nf=_par0 
.ends VCO_analogin_cmlbuffer_v2
** End of subcircuit definition.

** Library name: CAD_modules
** Cell name: test_VCO_analogin_cmlbuffer_v3
** View name: schematic
v5 net04 vss DC=vcm AC 1
v4 net03 vss DC=vcm
v3 vdd 0 DC=1
v2 vss 0 DC=0
v0 vcm vss DC=vcm AC 1
m3 vdd oxp net04 vdd pmos l=60e-9 w='wpppp' multi=16 nf=fpppp 
m2 net04 oxn vdd vdd pmos l=60e-9 w='wpppp' multi=16 nf=fpppp
m1 vdd op vdd vdd pmos l=60e-9 w='wpppp' multi=16 nf=fpppp 
m0 vdd on vdd vdd pmos l=60e-9 w='wpppp' multi=16 nf=fpppp
e1 net3 vcm VCVS vss dvv 1
e0 net4 vcm VCVS dvv vss 1
v1 dvv vss AC 500e-3 SIN 0 200e-3 16e6
xi2 vdd vss oxn oxp net03 net03 VCO_analogin_cmlbuffer_v2 _par0=fing_in _par1=multi _par2=fing_ttt
xi0 vdd vss on op net4 net3 VCO_analogin_cmlbuffer_v2 _par0=fing_in _par1=multi _par2=fing_ttt
.END
