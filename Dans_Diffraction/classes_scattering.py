# -*- coding: utf-8 -*-
"""
Scattering Class "classes_scattering.py"
    Scattering functions for Crystal class.

By Dan Porter, PhD
Diamond
2017

Version 1.7
Last updated: 26/05/20

Version History:
10/09/17 0.1    Program created
30/10/17 1.0    Main functions finshed, some testing done
06/01/18 1.1    Renamed classes_scattering.py
31/10/18 1.2    Added print_symmetric_contributions
21/01/19 1.3    Added non-resonant diffraction, corrected resonant diffraction
16/12/19 1.4    Added multiple_scattering code, print_all_reflections updated with units
18/02/20 1.5    Added tensor_scattering code
20/03/20 1.6    Increased powder gauss width from 2fwhm to 6fwhm, added powder averaging
14/04/20 1.6    Added powder_correction
26/05/20 1.7    Removed tensor_scattering

@author: DGPorter
"""

import sys,os
import numpy as np

from . import functions_general as fg
from . import functions_crystallography as fc
from . import multiple_scattering as ms
# from . import tensor_scattering as ts  # Removed V1.7

__version__ = '1.7'
__scattering_types__ = {'xray': ['xray','x','x-ray','thomson','charge'],
                        'neutron': ['neutron','n','nuclear'],
                        'xray magnetic': ['xray magnetic','magnetic xray','spin xray','xray spin'],
                        'neutron magnetic': ['neutron magnetic','magnetic neutron','magnetic'],
                        'xray resonant': ['xray resonant','resonant','resonant xray','rxs']}


class Scattering:
    """
    Simulate diffraction from Crystal
    Useage:
        xtl = Crystal()
        xtl.Scatter.setup_scatter(type='x-ray',energy_keV=8.0)
        xtl.Scatter.intensity([h,k,l]) # Returns intensity
        print(xtl.Scatter.print_all_refelctions()) # Returns formated string of all allowed reflections
        
        Allowed radiation types:
            'xray','neutron','xray magnetic','neutron magnetic','xray resonant'
    """
    
    #------Options-------
    # Standard Options
    _scattering_type = 'xray'  # 'xray','neutron','xray magnetic','neutron magnetic','xray resonant'
    _scattering_specular_direction = [0,0,1]  # reflection
    _scattering_parallel_direction = [0,0,1]  # transmission
    _scattering_theta_offset = 0.0
    _scattering_min_theta = -180.0
    _scattering_max_theta = 180.0
    _scattering_min_twotheta = -180.0
    _scattering_max_twotheta = 180.0

    # Units
    _powder_units = 'tth' # tth (two theta), Q, d
    
    # Complex Structure factor
    _return_structure_factor = False
    
    # Thermal Factors
    _use_isotropic_thermal_factor = True
    _use_anisotropic_thermal_factor = False
    
    # Magnetic Options
    _calclate_magnetic_component = True
    _use_magnetic_form_factor = True
    
    # Polarisation Options
    _polarised = False
    _polarisation = 'sp'
    
    # Radiation energy
    _energy_kev = fg.Cu
    
    # Resonant X-ray Options
    _azimuthal_angle = 0
    _azimuthal_reference = [1,0,0]
    _resonant_approximation_e1e1 = True
    _resonant_approximation_e2e2 = False
    _resonant_approximation_e1e2 = False
    _resonant_approximation_m1m1 = False
    
    def __init__(self, xtl):
        "initialise"
        self.xtl = xtl

        # Initialise the scattering type container
        self.Type = ScatteringTypes(self, __scattering_types__)
    
    def x_ray(self, HKL):
        """
        Calculate the squared structure factor for the given HKL, using x-ray scattering factors
          Scattering.x_ray([1,0,0])
          Scattering.x_ray([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)
        
        Qmag = self.xtl.Cell.Qmag(HKL)
        
        # Get atomic form factors
        ff = fc.xray_scattering_factor(type,Qmag)
        
        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso, Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calcualtion not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)
        
        # Calculate structure factor
        # Broadcasting used on 2D ff
        SF =  np.sum(ff*dw*occ*np.exp(1j*2*np.pi*dot_KR),axis=1)
        #SF = np.zeros(Nref,dtype=np.complex)
        #for ref in range(Nref):
        #    for at in range(Nat): 
        #        SF[ref] += ff[ref,at]*dw[ref,at]*occ[at]*np.exp(1j*2*np.pi*dot_KR[ref,at])
        
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def x_ray_fast(self, HKL):
        """
        Calculate the squared structure factor for the given HKL, using atomic number as scattering length
          Scattering.x_ray_fast([1,0,0])
          Scattering.x_ray_fast([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)
        
        # Get atomic form factors
        ff = fc.atom_properties(type, 'Z')
        
        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            Qmag = self.xtl.Cell.Qmag(HKL)
            dw = fc.debyewaller(uiso,Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
        # Calculate structure factor
        SF =  np.sum(ff*dw*occ*np.exp(1j*2*np.pi*dot_KR),axis=1)
        
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def neutron(self, HKL):
        """
        Calculate the squared structure factor for the given HKL, using neutron scattering length
          Scattering.neutron([1,0,0])
          Scattering.neutron([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)
        
        # Get atomic form factors
        ff = fc.atom_properties(type, 'Coh_b')
        
        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            Qmag = self.xtl.Cell.Qmag(HKL)
            dw = fc.debyewaller(uiso,Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
        # Calculate structure factor
        SF =  np.sum(ff*dw*occ*np.exp(1j*2*np.pi*dot_KR),axis=1)
        
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def magnetic_neutron(self, HKL):
        """
        Calculate the magnetic component of the structure factor for the given HKL, using neutron rules and form factor
          Scattering.magnetic_neutron([1,0,0])
          Scattering.magnetic_neutron([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)
        
        Q = self.xtl.Cell.calculateQ(HKL)
        Qhat = fg.norm(Q).reshape([-1,3])
        Qmag = self.xtl.Cell.Qmag(HKL)
        
        # Get magnetic form factors
        if self._use_magnetic_form_factor:
            ff = fc.magnetic_form_factor(type,Qmag)
        else:
            ff = np.ones([len(HKL), Nat])
        
        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1,1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz)
        moment = momentmag*fg.norm(momentxyz) # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
        # Calculate structure factor
        SF = np.zeros(Nref,dtype=np.complex)
        for n,Qh in enumerate(Qhat):
            SFm = [0.,0.,0.]
            for m,mom in enumerate(moment):
                # Calculate Magnetic part
                QM = mom - np.dot(Qh,mom)*Qh
                
                # Calculate structure factor
                SFm = SFm + ff[n,m]*np.exp(1j*2*np.pi*dot_KR[n,m])*QM
            
            # Calculate polarisation with incident neutron
            if self._polarised:
                SF[n] = np.dot(SFm,self._polarisation_vector_incident)
            else:
                #SF[n] = np.dot(SFm,SFm) # maximum possible
                SF[n] = (np.dot(SFm,[1,0,0]) + np.dot(SFm,[0,1,0]) + np.dot(SFm,[0,0,1]))/3 # average polarisation

        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def xray_magnetic(self, HKL):
        """
        Calculate the non-resonant magnetic component of the structure factor 
        for the given HKL, using x-ray rules and form factor
          Scattering.xray_magnetic([1,0,0])
          Scattering.xray_magnetic([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        
        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
        Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
        No orbital component assumed
        magnetic moments assumed to be in the same reference frame as the polarisation
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)
        
        Qmag = self.xtl.Cell.Qmag(HKL)
        
        # Get magnetic form factors
        if self._use_magnetic_form_factor:
            ff = fc.magnetic_form_factor(type,Qmag)
        else:
            ff = np.ones([len(HKL),Nat])
        
        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1,1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz) # moment direction in cartesian reference frame
        moment = momentmag*fg.norm(momentxyz) # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
        # Calculate structure factor
        SF = np.zeros(Nref,dtype=np.complex)
        for n in range(Nref):
            # Calculate vector structure factor
            SFm = [0.,0.,0.]
            for m,mom in enumerate(moment):
                SFm = SFm + ff[n,m]*np.exp(1j*2*np.pi*dot_KR[n,m])*mom
            
            # Calculate polarisation with incident x-ray
            # The reference frame of the x-ray and the crystal are assumed to be the same
            # i.e. pol=[1,0,0] || mom=[1,0,0] || (1,0,0)
            if self._polarised:
                SF[n] = np.dot(SFm,self._polarisation_vector_incident)
            else:
                #SF[n] = np.dot(SFm,SFm) # maximum possible
                SF[n] = (np.dot(SFm,[1,0,0]) + np.dot(SFm,[0,1,0]) + np.dot(SFm,[0,0,1]))/3 # average polarisation
        
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def xray_resonant(self, HKL, energy_kev=None, polarisation='sp', F0=1, F1=1, F2=1, azim_zero=[1,0,0], PSI=[0], disp=False):
        """
        Calculate structure factors using resonant scattering factors in the dipolar approximation
          I = Scattering.xray_resonant(HKL,energy_kev,polarisation,F0,F1,F2)
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
            energy_kev = x-ray energy in keV
            polarisation = x-ray polarisation: 'ss',{'sp'},'ps','pp'
            F0/1/2 = Resonance factor Flm
            azim_zero = [h,k,l] vector parallel to the origin of the azimuth
            psi = azimuthal angle defining the scattering plane
        
        Uses the E1E1 resonant x-ray scattering amplitude:
            fxr_n = (ef.ei)*F0 -i(ef X ei).z_n*F1 + (ef.z_n)(ei.z_n)F2
        
        Where ei and ef are the initial and final polarisation states, respectively,
        and z_n is a unit vector in the direction of the magnetic moment of the nth ion.
        The polarisation states are determined to be one of the natural synchrotron 
        states, where sigma (s) is perpendicular to the scattering plane and pi (p) is
        parallel to it.
                ( s-s  s-p )
                ( p-s  p-p )
        
        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (15)
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        
        PSI = np.asarray(PSI,dtype=np.float).reshape([-1])
        Npsi = len(PSI)
        
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)
        
        Qmag = self.xtl.Cell.Qmag(HKL)
        
        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso,Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
        SF = np.zeros([Nref,Npsi],dtype=np.complex)
        for psival in range(Npsi):
            # Get resonant form factor
            fxres = self.xray_resonant_scattering_factor(HKL,energy_kev,polarisation,F0,F1,F2,azim_zero,PSI[psival],disp=disp)
            
            # Calculate structure factor
            # Broadcasting used on 2D fxres
            SF[:,psival] =  np.sum(fxres*dw*occ*np.exp(1j*2*np.pi*dot_KR),axis=1)
            
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def xray_resonant_scattering_factor(self, HKL, energy_kev=None, polarisation='sp', F0=1, F1=1, F2=1, azim_zero=[1,0,0], psi=0, disp=False):
        """
        Calcualte fxres, the resonant x-ray scattering factor
          fxres = Scattering.xray_resonant_scattering_factor(HKL,energy_kev,polarisation,F0,F1,F2,azim_zero,psi)
        energy_kev = x-ray energy in keV
            polarisation = x-ray polarisation: 'ss',{'sp'},'ps','pp'
            F0/1/2 = Resonance factor Flm
            azim_zero = [h,k,l] vector parallel to the origin of the azimuth {[1,0,0]}
            psi = azimuthal angle defining the scattering plane {0}
        
        Uses the E1E1 resonant x-ray scattering amplitude:
            fxr_n = (ef.ei)*F0 -i(ef X ei).z_n*F1 + (ef.z_n)(ei.z_n)F2
        
        Where ei and ef are the initial and final polarisation states, respectively,
        and z_n is a unit vector in the direction of the magnetic moment of the nth ion.
        The polarisation states are determined to be one of the natural synchrotron 
        states, where sigma (s) is perpendicular to the scattering plane and pi (p) is
        parallel to it.
                ( s-s  s-p )
                ( p-s  p-p )
        
        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (15)
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        Qmag = self.xtl.Cell.Qmag(HKL)
        tth = fc.cal2theta(Qmag, energy_kev)
        
        mxmymz = self.xtl.Structure.mxmymz()
        Nat = len(mxmymz)
        
        fxres = np.zeros([Nref,Nat],dtype=np.complex)
        for ref in range(Nref):
            # Resonant scattering factor
            # Electric Dipole transition at 3d L edge
            #F0,F1,F2 = 1,1,1 # Flm (form factor?)
            z1,z2,z3 = self.scatteringcomponents(mxmymz, HKL[ref], azim_zero, psi).T
            tthr = np.deg2rad(tth[ref])/2.0

            polarisation = polarisation.replace('-', '').replace(' ', '')
            if polarisation in ['sigmasigma', 'sigsig', 'ss']:    # Sigma-Sigma
                f0 = 1*np.ones(Nat)
                f1 = 0*np.ones(Nat)
                f2 = z2**2
            elif polarisation in ['sigmapi', 'sigpi', 'sp']:  # Sigma-Pi
                f0 = 0 * np.ones(Nat)
                f1 = z3 * np.sin(tthr) + z1 * np.cos(tthr)
                f2 = z2 * (z1 * np.sin(tthr) + z3 * np.cos(tthr))
            elif polarisation in ['pisigma', 'pisig', 'ps']:  # Pi-Sigma
                f0 = 0*np.ones(Nat)
                f1 = z1*np.cos(tthr) - z3*np.sin(tthr)
                f2 = -z2*(z1*np.sin(tthr)-z3*np.cos(tthr))
            elif polarisation in ['pipi', 'pp']:  # Pi-Pi
                f0 = np.cos(2*tthr)*np.ones(Nat)
                f1 = -z2*np.sin(2*tthr)
                f2 = -(np.cos(tthr)**2)*(z1**2*np.tan(tthr)**2 + z3**2)
            else:
                raise ValueError('Incorrect polarisation. pol should be e.g. ''ss'' or ''sp''')
            fxres[ref,:] = F0*f0 -1j*F1*f1 + F2*f2
            if disp:
                print('( h, k, l)   TTH  (    mx,    my,    mz)  (    z1,    z2,    z3)')
                fmt = '(%2.0f,%2.0f,%2.0f) %6.2f  (%6.3f,%6.3f,%6.3f)  (%6.3f,%6.3f,%6.3f)  f0=%8.4f  f1=%8.4f  f2=%8.4f fxres= (%8.4f + %8.4fi)'
                for at in range(Nat):
                    vals = (HKL[ref,0],HKL[ref,1],HKL[ref,2],tth[ref],
                            mxmymz[at,0],mxmymz[at,1],mxmymz[at,2],
                            z1[at],z2[at],z3[at],
                            f0[at],f1[at],f2[at],
                            fxres[ref,at].real,fxres[ref,at].imag)
                    print(fmt%vals)
        return fxres

    def xray_nonresonant_magnetic(self, HKL, energy_kev=None, azim_zero=[1, 0, 0], psi=0, polarisation='s-p', disp=False):
        """
        Calculate the non-resonant magnetic component of the structure factor
        for the given HKL, using x-ray rules and form factor
          Scattering.xray_magnetic([1,0,0])
          Scattering.xray_magnetic([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.

        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
        Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
        No orbital component assumed
        magnetic moments assumed to be in the same reference frame as the polarisation
        """

        HKL = np.asarray(np.rint(HKL), dtype=np.float).reshape([-1, 3])

        uvw, type, label, occ, uiso, mxmymz = self.xtl.Structure.get()

        kin, kout, ein, eout = self.scatteringvectors(HKL, energy_kev, azim_zero, psi, polarisation)

        Qmag = self.xtl.Cell.Qmag(HKL)

        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso, Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1

        # Get magnetic form factors
        if self._use_magnetic_form_factor:
            ff = fc.magnetic_form_factor(type, Qmag)
        else:
            ff = np.ones([len(HKL), len(uvw)])

        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1, 1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz)  # moment direction in cartesian reference frame
        moment = momentmag * fg.norm(momentxyz)  # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.

        # Magnetic form factor
        # f_non-res_mag = i.r0.(hw/mc^2).fD.[.5.L.A + S.B] #equ 2 Hill+McMorrow 1996
        # ignore orbital moment L
        B = np.zeros([len(HKL), 3])
        for n in range(len(HKL)):
            #print(n,HKL[n],kin[n],kout[n],ein[n],eout[n])
            B[n, :] = np.cross(eout[n], ein[n]) + \
                np.cross(kout[n], eout[n]) * np.dot(kout[n], ein[n]) - \
                np.cross(kin[n], ein[n]) * np.dot(kin[n], eout[n]) - \
                np.cross(np.cross(kout[n], eout[n]), np.cross(kin[n], ein[n]))
        fspin = 1j * ff * np.dot(moment, B.T).T

        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)

        # Calculate structure factor
        SF = np.sum(fspin * dw * occ * np.exp(1j * 2 * np.pi * dot_KR), axis=1)

        if disp:
            for n in range(len(HKL)):
                pr_h = '(%2.0f,%2.0f,%2.0f)'%tuple(HKL[n,:])
                pr_b = '(%6.2g,%6.2g,%6.2g)'%tuple(B[n,:])
                print('psi=%3.0f  %3d hkl=%10s B=%22s'%(psi, n, pr_h, pr_b))
                ctot = 0j
                for m in range(len(moment)):
                    if np.sum(moment[m,:]**2) < 0.01: continue
                    pr_m = '(%6.2g,%6.2g,%6.2g)'%tuple(moment[m,:])
                    dot = np.dot(moment[m],B[n,:])
                    pdot = '%6.3f'%dot
                    phase = np.exp(1j * 2 * np.pi * dot_KR[n, m])
                    prph = '%5.2f+i%5.2f'%(np.real(phase),np.imag(phase))
                    prod = dot*phase
                    pprd = '%5.2f+i%5.2f'%(np.real(prod),np.imag(prod))
                    ctot += prod
                    ptot = '%5.2f+i%5.2f'%(np.real(ctot),np.imag(ctot))
                    print('\t%3d mom=%22s dot(mom,B)=%6s   exp(ik.r)=%12s   sum=%12s   tot=%12s'%(m,pr_m,pdot,prph,pprd,ptot))

        SF = SF / self.xtl.scale

        if self._return_structure_factor: return SF

        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)

    def xray_resonant_magnetic(self, HKL, energy_kev=None, azim_zero=[1, 0, 0], psi=0, polarisation='s-p', F0=0, F1=1, F2=0, disp=True):
        """
        Calculate the non-resonant magnetic component of the structure factor
        for the given HKL, using x-ray rules and form factor
          Scattering.xray_magnetic([1,0,0])
          Scattering.xray_magnetic([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.

        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
        Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
        No orbital component assumed
        magnetic moments assumed to be in the same reference frame as the polarisation
        """

        HKL = np.asarray(np.rint(HKL), dtype=np.float).reshape([-1, 3])

        uvw, type, label, occ, uiso, mxmymz = self.xtl.Structure.get()

        kin, kout, ein, eout = self.scatteringvectors(HKL, energy_kev, azim_zero, psi, polarisation)

        Qmag = self.xtl.Cell.Qmag(HKL)

        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso, Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1

        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1, 1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz)  # moment direction in cartesian reference frame
        moment = momentmag * fg.norm(momentxyz)  # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.
        fe1e1 = np.zeros([len(HKL), len(uvw)], dtype=np.complex)
        for ref in range(len(HKL)):
            # Magnetic form factor
            # f_res_mag = [(e'.e)F0 - i(e'xe).Z*F1 + (e'.Z)*(e.Z)*F2] #equ 7 Hill+McMorrow 1996
            f0 = np.dot(eout[ref], ein[ref])
            f1 = np.dot(np.cross(eout[ref], ein[ref]), moment.T)
            f2 = np.dot(eout[ref], moment.T) * np.dot(ein[ref], moment.T)
            fe1e1[ref, :] = f0*F0 - 1j*f1*F1 + f2*F2

        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)

        if disp:
            for ref in range(len(HKL)):
                pr_h = '(%2.0f,%2.0f,%2.0f)'%tuple(HKL[ref,:])
                print('psi=%3.0f  %3d hkl=%10s'%(psi, ref, pr_h))
                ctot = 0j
                for m in range(len(moment)):
                    if np.sum(moment[m,:]**2) < 0.01: continue
                    pr_m = '(%6.2g,%6.2g,%6.2g)'%tuple(moment[m,:])
                    f0 = np.dot(eout[ref], ein[ref])
                    f1 = np.dot(np.cross(eout[ref], ein[ref]), moment[m,:])
                    f2 = np.dot(eout[ref], moment[m,:]) * np.dot(ein[ref], moment[m,:])
                    dot = f0*F0 - 1j*f1*F1 + f2*F2
                    phase = np.exp(1j * 2 * np.pi * dot_KR[ref, m])
                    prph = '%5.2f+i%5.2f'%(np.real(phase),np.imag(phase))
                    prod = dot*phase
                    pprd = '%5.2f+i%5.2f'%(np.real(prod),np.imag(prod))
                    ctot += prod
                    ptot = '%5.2f+i%5.2f'%(np.real(ctot),np.imag(ctot))
                    print('\t%3d mom=%22s f0=%5.2f f1=%5.2f f2=%5.2f   exp(ik.r)=%12s   sum=%12s   tot=%12s'%(m,pr_m,f0,f1,f2,prph,pprd,ptot))

        # Calculate structure factor
        SF = np.sum(fe1e1 * dw * occ * np.exp(1j * 2 * np.pi * dot_KR), axis=1)

        SF = SF / self.xtl.scale

        if self._return_structure_factor: return SF

        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)

    def scatteringvectors(self, hkl, energy_kev=None, azim_zero=[1, 0, 0], psi=0, polarisation='s-p'):
        """
        Determine the scattering and polarisation vectors of a reflection based on energy, azimuth and polarisation.
        :param xtl: Crystal Class
        :param hkl: [n,3] array of reflections
        :param energy_kev: x-ray scattering energy in keV
        :param azim_zero: [1,3] direction along which the azimuthal zero angle is determind
        :param psi: float angle in degrees about the azimuth
        :param polarisation: polarisation with respect to the scattering plane, options:
                    'ss' : sigma-sigma polarisation
                    'sp' : sigma-pi polarisation
                    'ps' : pi-sigma polarisation
                    'pp' : pi-pi polarisation
                or: polarisation: float polarisation angle of scattered vector in degrees
        :return: kin, kout, ein, eout
        Returned values are [n,3] arrays
            kin : [n,3] array of incident wavevectors
            kout: [n,3] array of scattered wavevectors
            ein : [n,3] array of incident polarisation
            eout: [n,3] array of scattered polarisation

        The basis is chosen such that Q defines the scattering plane, sigma and pi directions are normal to this plane.
        Q is defined as Q = kout - kin, with kout +ve along the projection of azim_zero
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        # Define coordinate system I,J,Q (U1,U2,U3)
        Ihat, Jhat, Qhat = self.scatteringbasis(hkl, azim_zero, psi)
        Ihat = Ihat.reshape([-1, 3])
        Jhat = Jhat.reshape([-1, 3])
        Qhat = Qhat.reshape([-1, 3])

        # Determine wavevectors
        bragg = self.xtl.Cell.tth(hkl, energy_kev) / 2.
        rb = np.deg2rad(bragg).reshape([-1, 1])
        kin = np.cos(rb) * Ihat - np.sin(rb) * Qhat
        kout = np.cos(rb) * Ihat + np.sin(rb) * Qhat
        esig = Jhat  # sigma polarisation (in or out)
        piin = np.cross(kin, esig)  # pi polarisation in
        piout = np.cross(kout, esig)  # pi polarisation out

        # Polarisation
        try:
            # polarisation = 'ss' or 's-s'
            polarisation = polarisation.replace('-', '').replace(' ', '')
        except AttributeError:
            # polarisation = angle in deg from sigma' to pi'
            ein = 1.0 * esig
            pol = np.deg2rad(polarisation)
            eout = np.cos(pol)*esig + np.sin(pol)*piout
        if polarisation in ['sigmasigma', 'sigsig', 'ss']:
            ein = 1.0 * esig
            eout = 1.0 * esig
        elif polarisation in ['sigmapi', 'sigpi', 'sp']:
            ein = 1.0 * esig
            eout = 1.0 * piout
        elif polarisation in ['pisigma', 'pisig', 'ps']:
            ein = 1.0 * piin
            eout = 1.0 * esig
        elif polarisation in ['pipi', 'pp']:
            ein = 1.0 * piin
            eout = 1.0 * piout
        return kin, kout, ein, eout

    def scatteringcomponents(self, mxmymz, hkl, azim_zero=[1,0,0], psi=0):
        """
        Transform magnetic vector into components within the scattering plane
            ***warning - may not be correct for non-cubic systems***
        """

        # Define coordinate system I,J,Q (U1,U2,U3)
        U = self.scatteringbasis(hkl, azim_zero, psi)
        
        # Determine components of the magnetic vector
        z1z2z3 = np.dot(mxmymz, U.T) # [mxmymz.I, mxmymz.J, mxmymz.Q]
        return fg.norm(z1z2z3)

    def scatteringbasis(self, hkl, azim_zero=[1, 0, 0], psi=0):
        """
        Determine the scattering and polarisation vectors of a reflection based on energy, azimuth and polarisation.
        :param hkl: [n,3] array of reflections
        :param azim_zero: [1,3] direction along which the azimuthal zero angle is determind
        :param psi: float azimuthal angle about U3 in degrees
        :return: U1, U2, U3
        The basis is chosen such that Q defines the scattering plane, the sigma direction is normal to this plane,
        the pi direction is always within this plane.
        The azimuthal angle defines a rotation about the Q axis in a clockwise mannor, matching I16.
        At an azimuth of 0degrees, U1 is perpendicular to Q, along the direction of azim_zero.
        """

        # Define coordinate system I,J,Q (U1,U2,U3)
        # See FDMNES User's Guide p20 'II-11) Anomalous or resonant diffraction'
        # U1 || projection of azim_zero
        # U2 _|_ U1,U3
        # U3 || Q = kf-ki
        azim_zero = fg.norm(self.xtl.Cell.calculateQ(azim_zero)) # put in orthogonal basis
        Qhat = fg.norm(self.xtl.Cell.calculateQ(hkl)).reshape([-1,3])  # || Q
        AxQ = fg.norm(np.cross(azim_zero, Qhat))
        Ihat = fg.norm(np.cross(Qhat, AxQ)).reshape([-1,3])  # || to projection of azim_zero
        Jhat = fg.norm(np.cross(Qhat, Ihat)).reshape([-1,3])  # _|_ to I and Q

        # Rotate psi about Qhat
        rpsi = np.deg2rad(psi)
        # -ve sin makes clockwise rotation
        # This was checked on 21/1/19 vs CRO paper + sergio's calculations and seems to agree with experiment,
        # however we never did an azimuthal scan of the (103) which would have distinguished this completely.
        Ihat_psi = fg.norm(np.cos(rpsi) * Ihat - np.sin(rpsi) * Jhat)
        Jhat_psi = fg.norm(np.cross(Qhat, Ihat_psi))
        return np.vstack([Ihat_psi, Jhat_psi, Qhat])

    def print_scattering_coordinates(self,hkl,azim_zero=[1,0,0],psi=0):
        """
        Transform magnetic vector into components within the scattering plane
            ***warning - may not be correct for non-cubic systems***
        """
        
        # Define coordinate system I,J,Q (U1,U2,U3)
        Qhat = fg.norm(self.xtl.Cell.calculateQ(hkl)) # || Q
        AxQ = fg.norm(np.cross(azim_zero,Qhat))
        Ihat = fg.norm(np.cross(Qhat,AxQ)) # || to azim_zero
        Jhat = fg.norm(np.cross(Qhat,Ihat)) # -| to I and Q
        
        # Rotate coordinate system by azimuth
        Ihat_psi = fg.norm(np.cos(np.deg2rad(psi))*Ihat + np.sin(np.deg2rad(psi))*Jhat)
        Jhat_psi = fg.norm(np.cross(Qhat,Ihat_psi))
        
        # Determine components of the magnetic vector
        U=np.vstack([Ihat_psi,Jhat_psi,Qhat])
        print('U1 = (%5.2f,%5.2f,%5.2f)'%(U[0,0],U[0,1],U[0,2]))
        print('U2 = (%5.2f,%5.2f,%5.2f)'%(U[1,0],U[1,1],U[1,2]))
        print('U3 = (%5.2f,%5.2f,%5.2f)'%(U[2,0],U[2,1],U[2,2]))
    
    def print_intensity(self, HKL):
        """
        Print intensities calcualted in different ways
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Qmag =  self.xtl.Cell.Qmag(HKL)
        srt = np.argsort(Qmag)
        HKL = HKL[srt,:]
        
        IN=self.neutron(HKL)
        IX=self.x_ray(HKL)
        INM=self.magnetic_neutron(HKL)*1e4
        IXM=self.xray_magnetic(HKL)*1e4
        IXRss=self.xray_resonant(HKL, None, 'ss')
        IXRsp=self.xray_resonant(HKL, None, 'sp')
        IXRps=self.xray_resonant(HKL, None, 'ps')
        IXRpp=self.xray_resonant(HKL, None, 'pp')
        
        fmt = '(%2.0f,%2.0f,%2.0f)  %8.1f  %8.1f  %8.2f  %8.2f  ss=%8.2f  sp=%8.2f  ps=%8.2f  pp=%8.2f'
        print('( h, k, l)   Neutron      xray   Magn. N  Magn. XR   sig-sig    sig-pi    pi-sig     pi-pi')
        for n in range(len(HKL)):
            vals=(HKL[n][0],HKL[n][1],HKL[n][2],IN[n],IX[n],INM[n],IXM[n],IXRss[n],IXRsp[n],IXRps[n],IXRpp[n])
            print(fmt%vals)
    
    def intensity(self, HKL, scattering_type=None):
        """
        Calculate the squared structure factor for the given HKL
          Crystal.intensity([1,0,0])
          Crystal.intensity([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        
        Notes:
        - Uses x-ray atomic form factors, calculated from approximated tables in the ITC
        - This may be a little slow for large numbers of reflections, as it is not currently
         possible to use accelerated calculation methods in Jython.
        - Debye-Waller factor (atomic displacement) is applied for isotropic ADPs
        - Crystal.scale is used to scale the complex structure factor, so the intensity is  
         reduced by (Crystal.scale)^2
        - Testing against structure factors calculated by Vesta.exe is very close, though there
          are some discrepancies, probably due to the method of calculation of the form factor.
        """

        if scattering_type is None:
            scattering_type = self._scattering_type
        scattering_type = scattering_type.lower()

        # Break up long lists of HKLs
        n_arrays = np.ceil(len(HKL)*len(self.xtl.Structure.u)/10000.)
        hkl_array = np.array_split(HKL, n_arrays)

        intensity = []
        for _hkl in hkl_array:
            if scattering_type in ['xray','x','x-ray','thomson','charge']:
                intensity += self.x_ray(_hkl).tolist()
            elif scattering_type in ['neutron','n','nuclear']:
                intensity += self.neutron(_hkl).tolist()
            elif scattering_type in ['xray magnetic','magnetic xray','spin xray','xray spin']:
                intensity += list(self.xray_magnetic(_hkl)*1e4)
            elif scattering_type in ['neutron magnetic','magnetic neutron','magnetic']:
                intensity += list(self.magnetic_neutron(_hkl)*1e4)
            elif scattering_type in ['xray resonant','resonant','resonant xray','rxs']:
                intensity += self.xray_resonant(_hkl).tolist()
            elif scattering_type in ['xray resonant magnetic', 'xray magnetic resonant',
                                     'resonant magnetic', 'magnetic resonant']:
                intensity += self.xray_resonant_magnetic(
                    _hkl,
                    self._energy_kev,
                    self._azimuthal_reference,
                    self._azimuthal_angle,
                    self._polarisation,
                    F0=0, F1=1, F2=0).tolist()
            elif scattering_type in ['xray nonresonant magnetic', 'xray magnetic nonresonant',
                                     'nonresonant magnetic', 'magnetic nonresonant',
                                     'xray non-resonant magnetic', 'xray magnetic non-resonant',
                                     'non-resonant magnetic', 'magnetic non-resonant']:
                intensity += self.xray_resonant_magnetic(
                    _hkl,
                    self._energy_kev,
                    self._azimuthal_reference,
                    self._azimuthal_angle,
                    self._polarisation).tolist()
            else:
                print('Scattering type not defined')
        return np.array(intensity)
    
    def hkl(self, HKL, energy_kev=None):
        " Calculate the two-theta and intensity of the given HKL, display the result"
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        inten = self.intensity(HKL)
        
        print('Energy = %6.3f keV' % energy_kev)
        print('( h, k, l) TwoTheta  Intensity')
        for n in range(len(tth)):
            print('(%2.0f,%2.0f,%2.0f) %8.2f  %9.2f' % (HKL[n,0],HKL[n,1],HKL[n,2],tth[n],inten[n]))
    
    def hkl_reflection(self, HKL, energy_kev=None):
        " Calculate the theta, two-theta and intensity of the given HKL in reflection geometry, display the result"
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        theta = self.xtl.Cell.theta_reflection(HKL, energy_kev, self._scattering_specular_direction, self._scattering_theta_offset)
        inten = self.intensity(HKL)
        
        print('Energy = %6.3f keV' % energy_kev)
        print('Specular Direction = (%1.0g,%1.0g,%1.0g)' %
              (self._scattering_specular_direction[0],
               self._scattering_specular_direction[1],
               self._scattering_specular_direction[2]))
        print('( h, k, l)    Theta TwoTheta  Intensity')
        for n in range(len(tth)):
            print('(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f' %
                  (HKL[n, 0], HKL[n, 1], HKL[n, 2], theta[n], tth[n], inten[n]))
    
    def hkl_transmission(self,HKL,energy_kev=None):
        " Calculate the theta, two-theta and intensity of the given HKL in transmission geometry, display the result"
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        theta = self.xtl.Cell.theta_transmission(HKL, energy_kev, self._scattering_specular_direction,self._scattering_theta_offset)
        inten = self.intensity(HKL)
        
        print('Energy = %6.3f keV' % energy_kev)
        print('Direction parallel to beam  = (%1.0g,%1.0g,%1.0g)' %(self._scattering_parallel_direction[0],self._scattering_parallel_direction[1],self._scattering_parallel_direction[2]))
        print('( h, k, l)    Theta TwoTheta  Intensity')
        for n in range(len(tth)):
            print('(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f' % (HKL[n,0],HKL[n,1],HKL[n,2],theta[n],tth[n],inten[n]))
    
    def setup_scatter(self,type=None,energy_kev=None,wavelength_a=None, powder_units=None,
                      specular=None,parallel=None,theta_offset=None,
                      min_theta=None,max_theta=None,min_twotheta=None,max_twotheta=None):
        """
        Simple way to set scattering parameters, each parameter is internal to xtl (self)
        
        type        : self._scattering type               :  'xray','neutron','xray magnetic','neutron magnetic','xray resonant'
        energy_kev  : self._energy_kev                    :  radiation energy in keV
        wavelength_a: self._wavelength_a                  :  radiation wavelength in Angstrom
        powder_units: self._powder_units                  :  units to use when displaying/ plotting ['twotheta', 'd',' 'q']
        min_twotheta: self._scattering_min_two_theta      :  minimum detector (two-theta) angle
        max_twotheta: self._scattering_max_two_theta      :  maximum detector (two-theta) angle
        min_theta   : self._scattering_min_theta          :  minimum sample angle = -opening angle
        max_theta   : self._scattering_max_theta          :  maximum sample angle = opening angle
        theta_offset: self._scattering_theta_offset       :  sample offset angle
        specular    : self._scattering_specular_direction : [h,k,l] : reflections normal to sample surface
        parallel    : self._scattering_parallel_direction : [h,k,l] : reflections normal to sample surface
        """
        
        if type is not None:
            self._scattering_type = type
        
        if energy_kev is not None:
            self._energy_kev = energy_kev
        
        if wavelength_a is not None:
            self._energy_kev = fc.wave2energy(wavelength_a)

        if powder_units is not None:
            self._powder_units = powder_units
        
        if specular is not None:
            self._scattering_specular_direction = specular
        
        if parallel is not None:
            self._scattering_parallel_direction = parallel
        
        if theta_offset is not None:
            self._scattering_theta_offset = theta_offset
        
        if min_theta is not None:
            self._scattering_min_theta = min_theta
        
        if max_theta is not None:
            self._scattering_max_theta = max_theta
        
        if min_twotheta is not None:
            self._scattering_min_twotheta = min_twotheta
        
        if max_twotheta is not None:
            self._scattering_max_twotheta = max_twotheta
        
        print('Scattering Options:')
        print('                            Type : %s'%(self._scattering_type))
        print('                  Default Energy : %6.3f keV'%(self._energy_kev))
        print('                    Powder Units : %s'%(self._powder_units))
        print('  Specular Direction (reflection): (%2.0f,%2.0f,%2.0f)'%(self._scattering_specular_direction[0],self._scattering_specular_direction[1],self._scattering_specular_direction[2]))
        print('Parallel Direction (transmission): (%2.0f,%2.0f,%2.0f)'%(self._scattering_parallel_direction[0],self._scattering_parallel_direction[1],self._scattering_parallel_direction[2]))
        print('                   Sample Offset : %5.2f'%(self._scattering_theta_offset))
        print('             Minimum Theta angle : %5.2f'%(self._scattering_min_theta))
        print('             Maximum Theta angle : %5.2f'%(self._scattering_max_theta))
        print('         Minimum Two-Theta angle : %5.2f'%(self._scattering_min_twotheta))
        print('         Maximum Two-Theta angle : %5.2f'%(self._scattering_max_twotheta))

    def generate_powder(self, q_max=8, peak_width=0.01, background=0, powder_average=True):
        """
        Generates array of intensities along a spaced grid, equivalent to a powder pattern.
          Q,I = generate_powder(energy_kev=8.0,peak_width=0.05,background=0)
            q_max = maximum Q, in A-1
            peak_width = width of convolution, in A-1
            background = average of normal background
            powder_average = True*/False, apply the powder averaging correction
          Returns:
            Q = [1000x1] array of wave-vector values
            I = [1000x1] array of intensity values

        Note: To get two-theta values use:
            tth = fc.cal2theta(Q, energy_kev)
        Note: To get d-spacing values use:
            dspace = fc.caldspace(Q)
        """

        # Get reflections
        hmax, kmax, lmax = fc.maxHKL(q_max, self.xtl.Cell.UVstar())
        HKL = fc.genHKL([hmax, -hmax], [kmax, -kmax], [lmax, -lmax])
        HKL = self.xtl.Cell.sort_hkl(HKL)  # required for labels
        Qmag = self.xtl.Cell.Qmag(HKL)
        HKL = HKL[Qmag < q_max, :]
        Qmag = self.xtl.Cell.Qmag(HKL)
        # Qmag = Qmag[Qmag<q_max]

        # Calculate intensities
        I = self.intensity(HKL)

        if powder_average:
            # Apply powder averging correction, I0/|Q|**2
            I = I/(Qmag+0.001)**2

        # create plotting mesh
        pixels = int(2000 * q_max)  # reduce this to make convolution faster
        pixel_size = q_max / (1.0 * pixels)
        peak_width_pixels = peak_width / (1.0 * pixel_size)
        mesh = np.zeros([pixels])
        mesh_q = np.linspace(0, q_max, pixels)

        # add reflections to background
        pixel_coord = Qmag / (1.0 * q_max)
        pixel_coord = (pixel_coord * (pixels - 1)).astype(int)

        for n in range(1, len(I)):
            mesh[pixel_coord[n]] = mesh[pixel_coord[n]] + I[n]

        # Convolve with a gaussian (if >0 or not None)
        if peak_width:
            gauss_x = np.arange(-3*peak_width_pixels, 3*peak_width_pixels + 1)  # gaussian width = 2*FWHM
            G = fg.gauss(gauss_x, None, height=1, cen=0, fwhm=peak_width_pixels, bkg=0)
            mesh = np.convolve(mesh, G, mode='same')

            # Add background (if >0 or not None)
        if background:
            bkg = np.random.normal(background, np.sqrt(background), [pixels])
            mesh = mesh + bkg
        return mesh_q, mesh

    def powder_correction(self, HKL, intensities, symmetric_multiplyer=True, powder_average=True):
        """
        Averages symmetric reflections and applies symmetry multipliers and 1/q^2 correction
        Ic = I0*C
        :param HKL: [nx3] array of [h,k,l] reflections
        :param intensities: [nx1] array of reflection intensities
        :return: [mx3], [mx1] arrays of averaged, corrected reflections + intensity
        """
        # Average symmetric reflections
        rhkl, rinten = self.xtl.Symmetry.average_symmetric_intensity(HKL, intensities)

        if symmetric_multiplyer:
            multiplyer = self.xtl.Symmetry.reflection_multiplyer(rhkl)
            rinten = rinten*multiplyer

        if powder_average:
            q = self.xtl.Cell.Qmag(rhkl)
            rinten = rinten/(q+0.001)**2

        return rhkl, rinten

    def print_all_reflections(self, energy_kev=None, print_symmetric=False,
                              min_intensity=0.01, max_intensity=None, units=None):
        """
        Prints a list of all allowed reflections at this energy
            energy_kev = energy in keV
            print_symmetric = False*/True : omits reflections with the same intensity at the same angle
            min_intensity = None/ 0.01 : omits reflections less than this (remove extinctions)
            max_intensity = None/ 0.01 : omits reflections greater than this (show extinctions only)
            units = None/ 'twotheta'/ 'q'/ 'dspace' : choose scattering angle units to display
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        if min_intensity is None: min_intensity = -1
        if max_intensity is None: max_intensity = np.inf
        
        hkl = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        if not print_symmetric:
            hkl = self.xtl.Symmetry.remove_symmetric_reflections(hkl)
        hkl = self.xtl.Cell.sort_hkl(hkl)

        tth = self.xtl.Cell.tth(hkl, energy_kev)
        inrange = np.all([tth < self._scattering_max_twotheta, tth > self._scattering_min_twotheta], axis=0)
        hkl = hkl[inrange, :]
        tth = tth[inrange]
        #inten = np.sqrt(self.intensity(hkl)) # structure factor
        inten = self.intensity(hkl)

        if units is None:
            units = self._powder_units
        units = units.lower()
        if units in ['d', 'dspc', 'dspace', 'd space', 'd-space', 'dspacing', 'd spacing', 'd-spacing']:
            unit_str = 'd-spacing'
            unit = fc.caldspace(tth, energy_kev)
        elif units in ['q', 'wavevector']:
            unit_str = 'Q'
            unit = fc.calqmag(tth, energy_kev)
        else:
            unit_str = 'TwoTheta'
            unit = tth
        
        fmt = '(%3.0f,%3.0f,%3.0f) %10.2f  %9.2f\n'
        outstr = ''
        
        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Radiation: %s\n' % self._scattering_type
        outstr += '( h, k, l)    %10s  Intensity\n' % unit_str
        #outstr+= fmt % (hkl[0,0],hkl[0,1],hkl[0,2],tth[0],inten[0]) # hkl(0,0,0)
        count = 0
        for n in range(1, len(tth)):
            if inten[n] < min_intensity: continue
            if inten[n] > max_intensity: continue
            count += 1
            outstr += fmt % (hkl[n,0], hkl[n,1], hkl[n,2],unit[n],inten[n])
        outstr += 'Reflections: %1.0f\n' % count
        return outstr
    
    def print_ref_reflections(self, energy_kev=None, min_intensity=0.01, max_intensity=None):
        """
        Prints a list of all allowed reflections at this energy in the reflection geometry
            energy = energy in keV
            min_intensity = None/ 0.01 : omits reflections less than this (remove extinctions)
            max_intensity = None/ 0.01 : omits reflections greater than this (show extinctions only)
                       |
                    // |\
                    \\ |
                     \\|___/____
                      \\   \
                       \\__
        
        Note, to change min/max theta values or specular direciton, change the following attributres of 
        the crystal object:
            self._scattering_max_two_theta   :  maximum detector (two-theta) angle
            self._scattering_min_theta       :  minimum sample angle = -opening angle
            self._scattering_max_theta       :  maximum sample angle = opening angle
            self._scattering_theta_offset    :  sample offset angle
            self._scattering_specular_direction : [h,k,l] : reflections normal to sample surface
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        if min_intensity == None: min_intensity=-1
        if max_intensity == None: max_intensity=np.inf
        
        HKL = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        HKL = self.xtl.Cell.sort_hkl(HKL)
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        HKL = HKL[tth>self._scattering_min_twotheta,:]
        tth = tth[tth>self._scattering_min_twotheta]
        theta = self.xtl.Cell.theta_reflection(HKL, energy_kev, self._scattering_specular_direction,self._scattering_theta_offset)
        #inten = np.sqrt(self.intensity(HKL)) # structure factor
        inten = self.intensity(HKL)
        
        p1=(theta>self._scattering_min_theta) * (theta<self._scattering_max_theta)
        p2=(tth>(theta+self._scattering_min_theta)) * (tth<(theta+self._scattering_max_theta))
        pos_theta = p1*p2
        
        fmt = '(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f\n'
        outstr = '' 
        
        outstr+= 'Energy = %6.3f keV\n' % energy_kev
        outstr+= 'Radiation: %s\n' % self._scattering_type
        outstr+= 'Specular Direction = (%1.0g,%1.0g,%1.0g)\n' %(self._scattering_specular_direction[0],self._scattering_specular_direction[1],self._scattering_specular_direction[2])
        outstr+= '( h, k, l) TwoTheta    Theta  Intensity\n'
        #outstr+= fmt % (HKL[0,0],HKL[0,1],HKL[0,2],tth[0],theta[0],inten[0]) # hkl(0,0,0)
        count = 0
        for n in range(1,len(tth)):
            if inten[n] < min_intensity: continue
            if inten[n] > max_intensity: continue
            if not pos_theta[n]: continue
            #if not print_symmetric and np.abs(tth[n]-tth[n-1]) < 0.01: continue # only works if sorted
            count += 1
            outstr+= fmt % (HKL[n,0],HKL[n,1],HKL[n,2],tth[n],theta[n],inten[n])
        outstr+= 'Reflections: %1.0f\n'%count
        return outstr
    
    def print_tran_reflections(self,energy_kev=None, min_intensity=0.01,max_intensity=None):
        """
        Prints a list of all allowed reflections at this energy in the transmission geometry
            energy = energy in keV
            min_intensity = None/ 0.01 : omits reflections less than this (remove extinctions)
            max_intensity = None/ 0.01 : omits reflections greater than this (show extinctions only)
                   \ /      
             --<-- || --<-- 
                  / \       
        
        Note, to change min/max theta values or specular direciton, change the following attributres of 
        the crystal object:
            self._scattering_max_two_theta   :  maximum detector (two-theta) angle
            self._scattering_min_theta       :  minimum sample angle = -opening angle
            self._scattering_max_theta       :  maximum sample angle = opening angle
            self._scattering_theta_offset    :  sample offset angle
            self._scattering_parallel_direction : [h,k,l] : reflections normal to sample surface, parallel to beam at theta = 0
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        if min_intensity == None: min_intensity=-1
        if max_intensity == None: max_intensity=np.inf
        
        HKL = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        HKL = self.xtl.Cell.sort_hkl(HKL)
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        HKL = HKL[tth>self._scattering_min_twotheta,:]
        tth = tth[tth>self._scattering_min_twotheta]
        theta = self.xtl.Cell.theta_transmission(HKL, energy_kev, self._scattering_parallel_direction)
        #inten = np.sqrt(self.intensity(HKL)) # structure factor
        inten = self.intensity(HKL)
        
        p1=(theta>self._scattering_min_theta) * (theta<self._scattering_max_theta) 
        p2=(tth>(theta+self._scattering_min_theta)) * (tth<(theta+self._scattering_max_theta))
        pos_theta = p1*p2
        
        fmt = '(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f\n' 
        outstr = '' 
        
        outstr+= 'Energy = %6.3f keV\n' % energy_kev
        outstr+= 'Radiation: %s\n' % self._scattering_type
        outstr+= 'Direction parallel to beam  = (%1.0g,%1.0g,%1.0g)\n' %(self._scattering_parallel_direction[0],self._scattering_parallel_direction[1],self._scattering_parallel_direction[2])
        outstr+= '( h, k, l) TwoTheta    Theta  Intensity\n'
        #outstr+= fmt % (HKL[0,0],HKL[0,1],HKL[0,2],tth[0],theta[0],inten[0]) # hkl(0,0,0)
        count = 0
        for n in range(1,len(tth)):
            if inten[n] < min_intensity: continue
            if inten[n] > max_intensity: continue
            if not pos_theta[n]: continue
            #if not print_symmetric and np.abs(tth[n]-tth[n-1]) < 0.01: continue # only works if sorted
            count += 1
            outstr+= fmt % (HKL[n,0],HKL[n,1],HKL[n,2],tth[n],theta[n],inten[n])
        outstr+=('Reflections: %1.0f\n'%count)
        return outstr
    
    def print_symmetric_reflections(self,HKL):
        "Prints equivalent reflections"
        
        symHKL = self.xtl.Symmetry.symmetric_reflections(HKL)
        Nsyms = len(symHKL)
        outstr = ''
        outstr+= 'Equivalent reflections: %d\n' % Nsyms
        for n in range(Nsyms):
            outstr+= '(%5.3g,%5.3g,%5.3g)\n' % (symHKL[n,0],symHKL[n,1],symHKL[n,2])
        return outstr
    
    def print_atomic_contributions(self,HKL):
        """
        Prints the atomic contributions to the structure factor
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        
        # Calculate the full intensity
        I = self.intensity(HKL)
        
        # Calculate the structure factors of the symmetric atomic sites
        base_label = self.xtl.Atoms.label
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        
        Qmag = self.xtl.Cell.Qmag(HKL)
        
        # Get atomic form factors
        ff = fc.xray_scattering_factor(type,Qmag)
        
        # Get Debye-Waller factor
        dw = fc.debyewaller(uiso,Qmag)
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
        # Calculate structure factor
        SF =  ff*dw*occ*np.exp(1j*2*np.pi*dot_KR)
        
        # Sum structure factors of each base label in atoms
        SFbase = np.zeros([len(HKL),len(base_label)],dtype=np.complex128)
        for n in range(len(base_label)):
            label_idx = label == base_label[n]
            SFbase[:,n] = np.sum(SF[:,label_idx],axis=1)
        
        # Get the real part of the structure factor
        #SFtot = np.sqrt(np.real(SF * np.conj(SF)))
        SFrel = np.real(SFbase)
        SFimg = np.imag(SFbase)
        
        # Generate the results
        outstr = ''
        outstr+= '( h, k, l) Intensity' + ' '.join(['%12s    '%x for x in base_label])+'\n'
        for n in range(Nref):
            ss = ' '.join(['%6.1f + i%-6.1f' % (x,y) for x,y in zip(SFrel[n],SFimg[n])])
            outstr+= '(%2.0f,%2.0f,%2.0f) %9.2f    %s\n' % (HKL[n,0],HKL[n,1],HKL[n,2],I[n],ss)
        return outstr

    def print_symmetry_contributions(self,HKL):
        """
        Prints the symmetry contributions to the structure factor for each atomic site
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        Nref = len(HKL)
        
        # Calculate the full intensity
        I = self.intensity(HKL)
        
        # Calculate the structure factors of the symmetric atomic sites
        buvw,btype,base_label,bocc,buiso,bmxmymz = self.xtl.Atoms.get()
        operations = np.hstack([self.xtl.Symmetry.symmetric_coordinate_operations(buvw[n]) for n in range(len(buvw))])
        rotations = np.hstack([self.xtl.Symmetry.symmetric_coordinate_operations(buvw[n],True)[1] for n in range(len(buvw))])

        # Calculate the structure factors
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        Qmag = self.xtl.Cell.Qmag(HKL)
        ff = fc.xray_scattering_factor(type,Qmag)
        dw = fc.debyewaller(uiso,Qmag)
        dot_KR = np.dot(HKL,uvw.T)
        phase = np.exp(1j*2*np.pi*dot_KR)
        sf = ff*dw*occ*phase
        
        # Generate the results
        outstr = ''
        for n in range(Nref):
            ss = '\n'
            all_phase = 0j
            all_sf = 0j
            for lab in base_label:
                label_idx = np.argwhere(label == lab)
                ss += '  %s\n'%lab
                tot_phase = 0j
                tot_sf = 0j
                for a in label_idx:
                    uvwstr = '(%-7.3g,%-7.3g,%-7.3g)'%(uvw[a,0],uvw[a,1],uvw[a,2])
                    phstr = fg.complex2str(phase[n,a])
                    sfstr = fg.complex2str(sf[n,a])
                    val= (a,uvwstr,operations[a[0]],rotations[a[0]],phstr,sfstr)
                    ss += '    %3d %s %25s %20s  %s  %s\n'%val
                    tot_phase += phase[n,a]
                    tot_sf += sf[n,a]
                ss += '%74sTotal:  %s  %s\n'%(' ',fg.complex2str(tot_phase),fg.complex2str(tot_sf))
                all_phase += tot_phase
                all_sf += tot_sf
            ss += '%62s Reflection Total:  %s  %s\n'%(' ',fg.complex2str(all_phase),fg.complex2str(all_sf))
            outstr+= '(%2.0f,%2.0f,%2.0f) I = %9.2f    %s\n' % (HKL[n,0],HKL[n,1],HKL[n,2],I[n],ss)
        return outstr
    
    def find_close_reflections(self,HKL,energy_kev=None,max_twotheta=2,max_angle=10):
        """
        Find and print list of reflections close to the given one
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        all_HKL = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        all_HKL = self.xtl.Cell.sort_hkl(all_HKL)
        
        all_tth = self.xtl.Cell.tth(all_HKL,energy_kev)
        tth1 = self.xtl.Cell.tth(HKL, energy_kev)
        tth_dif = np.abs(all_tth-tth1)
        
        all_Q = self.xtl.Cell.calculateQ(all_HKL)
        Q1 = self.xtl.Cell.calculateQ(HKL)
        all_angles = np.abs([fg.ang(Q1,Q2,'deg') for Q2 in all_Q])
        
        selected = (tth_dif < max_twotheta)*(all_angles < max_angle)
        sel_HKL = all_HKL[selected,:]
        sel_tth = all_tth[selected]
        sel_angles = all_angles[selected]
        sel_intensity = self.intensity(sel_HKL)
        
        # Generate Results
        fmt = '(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f\n'
        outstr = '' 
        
        outstr+= 'Energy = %6.3f keV\n' % energy_kev
        outstr+= 'Close to Reflection (%2.0f,%2.0f,%2.0f)\n' %(HKL[0],HKL[1],HKL[2])
        outstr+= '( h, k, l) TwoTheta    Angle  Intensity\n'
        count = 0
        for n in range(0,len(sel_HKL)):
            count += 1
            outstr+= fmt % (sel_HKL[n,0],sel_HKL[n,1],sel_HKL[n,2],sel_tth[n],sel_angles[n],sel_intensity[n])
        outstr+= 'Reflections: %1.0f\n'%count
        return outstr

    def multiple_scattering(self, hkl, azir=[0, 0, 1], pv=[1, 0], energy_range=[7.8, 8.2], numsteps=60,
                            full=False, pv1=False, pv2=False, sfonly=True, pv1xsf1=False):
        """
        Run multiple scattering code, plot result.

        mslist = xtl.Scatter.multiple_scattering([h,k,l], energy_range=[7.8, 8.2])

        See multiple_scattering.py for more details. Code created by Dr Gareth Nisbet, DLS
        :param hkl: [h,k,l] principle reflection
        :param azir: [h,k,l] reference of azimuthal 0 angle
        :param pv: [s,p] polarisation vector
        :param energy_range: [min, max] energy range in keV
        :param numsteps: int: number of calculation steps from energy min to max
        :param full: True/False: calculation type: full
        :param pv1: True/False: calculation type: pv1
        :param pv2: True/False: calculation type: pv2
        :param sfonly: True/False: calculation type: sfonly *default
        :param pv1xsf1: True/False: calculation type: pv1xsf1?
        :return: array
        """
        return ms.run_calcms(self.xtl, hkl, azir, pv, energy_range, numsteps,
                             full=full, pv1=pv1, pv2=pv2, sfonly=sfonly, pv1xsf1=pv1xsf1)

    def ms_azimuth(self, hkl, energy_kev, azir=[0, 0, 1], pv=[1, 0], numsteps=3, peak_width=0.1,
                   full=False, pv1=False, pv2=False, sfonly=True, pv1xsf1=False):
        """
        Returns an azimuthal dependence at a particular energy

        :param xtl: Crystal structure from Dans_Diffraction
        :param hkl: [h,k,l] principle reflection
        :param energy_kev: calculation energy
        :param azir: [h,k,l] reference of azimuthal 0 angle
        :param pv: [s,p] polarisation vector
        :param numsteps: int: number of calculation steps from energy min to max
        :param peak_width: convolution width
        :param full: True/False: calculation type: full
        :param pv1: True/False: calculation type: pv1
        :param pv2: True/False: calculation type: pv2
        :param sfonly: True/False: calculation type: sfonly *default
        :param pv1xsf1: True/False: calculation type: pv1xsf1?
        :return: None
        """

        energy_range = [energy_kev-0.001, energy_kev+0.001]
        mslist = self.xtl.Scatter.multiple_scattering(hkl, azir, pv, energy_range, numsteps,
                                                      full=full, pv1=pv1, pv2=pv2, sfonly=sfonly, pv1xsf1=pv1xsf1)

        if pv1 + pv2 + sfonly + full + pv1xsf1 != 0:
            azimuth = np.concatenate(mslist[:, [3, 4]])
            intensity = np.concatenate(mslist[:, [-1, -1]])
        else:
            azimuth = np.concatenate(mslist[:, [3, 4]])
            intensity = np.ones(azimuth)

        # create plotting mesh
        peak_width_pixels = 10
        pixels = int(360*(peak_width_pixels/ peak_width))
        mesh_azi = np.linspace(-180, 180, pixels)
        mesh = np.zeros(mesh_azi.shape)

        # add reflections to background
        pixel_coord = (azimuth-180) / 360.
        pixel_coord = (pixel_coord * pixels).astype(int)

        for n in range(len(intensity)):
            mesh[pixel_coord[n]] = mesh[pixel_coord[n]] + intensity[n]

        # Convolve with a gaussian (if >0 or not None)
        if peak_width:
            gauss_x = np.arange(-3 * peak_width_pixels, 3 * peak_width_pixels + 1)  # gaussian width = 2*FWHM
            G = fg.gauss(gauss_x, None, height=1, cen=0, fwhm=peak_width_pixels, bkg=0)
            mesh = np.convolve(mesh, G, mode='same')
        return mesh_azi, mesh

    '''  Removed tensor scattering 26/05/20 V1.7
    def tensor_scattering(self, atom_label, hkl, energy_kev=None, azir=[0, 0, 1], psideg=0, process='E1E1',
                          rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None):
        """
        Return tensor scattering intensities
          ss, sp, ps, pp = tensor_scattering('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param hkl: list/array, [h,k,l] reflection to calculate
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param psideg: float/array, azimuthal angle/ range
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :return: ss, sp, ps, pp intensity values
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        # Calculate tensor scattering
        ss, sp, ps, pp = ts.CalculateIntensityInPolarizationChannels(
            process, B, sitevec, sglist, lam, hkl, azir, psideg,
            K=rank, Time=time, Parity=parity, mk=mk, lk=lk, sk=sk
        )
        return ss, sp, ps, pp

    def tensor_scattering_stokes(self, atom_label, hkl, energy_kev=None, azir=[0, 0, 1], psideg=0, stokes=0,
                                 pol_theta=45, process='E1E1', rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None):
        """
        Return tensor scattering intensities for non-standard polarisation
          pol = tensor_scattering_stokes('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90, stokes=45)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param hkl: list/array, [h,k,l] reflection to calculate
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param psideg: float, azimuthal angle
        :param stokes: float/array, rotation of polarisation analyser (0=sigma), degrees
        :param pol_theta: float, scattering angle of polarisation analyser, degrees
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :return: array of intensity values
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        # Calculate tensor scattering
        stokesvec_swl = [0, 0, 1]
        pol = ts.CalculateIntensityFromPolarizationAnalyser(
            process, B, sitevec, sglist, lam, hkl, azir, psideg, stokes, pol_theta,
            stokesvec_swl, rank, time, parity, mk, lk, sk
        )
        return pol

    def print_tensor_scattering(self, atom_label, hkl, energy_kev=None, azir=[0, 0, 1], psideg=0, process='E1E1',
                                rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None):
        """
        Return tensor scattering intensities
          ss, sp, ps, pp = tensor_scattering('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param hkl: list/array, [h,k,l] reflection to calculate
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param psideg: float, azimuthal angle
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :return: str
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        # Calculate tensor scattering
        ss, sp, ps, pp = ts.CalculateIntensityInPolarizationChannels(
            process, B, sitevec, sglist, lam, hkl, azir, psideg,
            K=rank, Time=time, Parity=parity, mk=mk, lk=lk, sk=sk
        )
        outstr1 = ts.tensorproperties(sitevec, sglist, hkl, Parity=parity, Time=time)
        outstr2 = ts.print_tensors(B, sitevec, sglist, hkl, K=rank, Parity=parity, Time=time)
        outstr3 = "\nScattering Tensor:\n\n    [ss, sp] = [%5.2f, %5.2f]\n    [ps, pp]   [%5.2f, %5.2f]"
        outstr3 = outstr3 % (ss, sp, ps, pp)
        return outstr1 + outstr2 + outstr3

    def print_tensor_scattering_refs(self, atom_label, energy_kev=None, azir=[0, 0, 1], psideg=0, process='E1E1',
                                     rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None,
                                     print_symmetric=False, units=None):
        """
        Return tensor scattering intensities for all reflections at given azimuth and energy
          ss, sp, ps, pp = tensor_scattering('Ru1', 2.838, [0,1,0], psideg=90)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param psideg: float, azimuthal angle
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :param print_symmetric: False*/True : omits reflections with the same intensity at the same angle
        :param units: None/ 'twotheta'/ 'q'/ 'dspace' : choose scattering angle units to display
        :return: str
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        hkl = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        if not print_symmetric:
            hkl = self.xtl.Symmetry.remove_symmetric_reflections(hkl)
        hkl = self.xtl.Cell.sort_hkl(hkl)

        tth = self.xtl.Cell.tth(hkl, energy_kev)
        inrange = np.all([tth < self._scattering_max_twotheta, tth > self._scattering_min_twotheta], axis=0)
        hkl = hkl[inrange, :]
        tth = tth[inrange]

        if units is None:
            units = self._powder_units
        units = units.lower()
        if units in ['d', 'dspc', 'dspace', 'd space', 'd-space', 'dspacing', 'd spacing', 'd-spacing']:
            unit_str = 'd-spacing'
            unit = fc.caldspace(tth, energy_kev)
        elif units in ['q', 'wavevector']:
            unit_str = 'Q'
            unit = fc.calqmag(tth, energy_kev)
        else:
            unit_str = 'TwoTheta'
            unit = tth

        fmt = '(%3.0f,%3.0f,%3.0f) %10.2f  %11.2f %11.2f %11.2f %11.2f\n'
        outstr = 'Tensor Scattering %s\n' % self.xtl.name
        outstr += 'Process: %s, site: %s\n' % (process, atom_label)
        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Psi_0 = (%.2g,%.2g,%.2g)  Psi = %3.3g\n' % (azir[0], azir[1], azir[2], psideg)
        outstr += '( h, k, l)    %10s  Sigma-Sigma    Sigma-Pi    Pi-Sigma       Pi-Pi\n' % unit_str

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        for n in range(1, len(tth)):
            ss, sp, ps, pp = ts.CalculateIntensityInPolarizationChannels(
                process, B, sitevec, sglist, lam, hkl[n, :], azir, psideg,
                K=rank, Time=time, Parity=parity, mk=mk, lk=lk, sk=sk
            )
            outstr += fmt % (hkl[n, 0], hkl[n, 1], hkl[n, 2], unit[n], ss, sp, ps, pp)
        outstr += 'Reflections: %1.0f\n' % len(tth)
        return outstr

    def print_tensor_scattering_refs_max(self, atom_label, energy_kev=None, azir=[0, 0, 1], process='E1E1',
                                     rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None,
                                     print_symmetric=False, units=None):
        """
        Return tensor scattering intensities for all reflections at given energy at maximum intensity psi
          ss, sp, ps, pp = tensor_scattering('Ru1', 2.838, [0,1,0], psideg=90)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :param print_symmetric: False*/True : omits reflections with the same intensity at the same angle
        :param units: None/ 'twotheta'/ 'q'/ 'dspace' : choose scattering angle units to display
        :return: str
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        hkl = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        if not print_symmetric:
            hkl = self.xtl.Symmetry.remove_symmetric_reflections(hkl)
        hkl = self.xtl.Cell.sort_hkl(hkl)

        tth = self.xtl.Cell.tth(hkl, energy_kev)
        inrange = np.all([tth < self._scattering_max_twotheta, tth > self._scattering_min_twotheta], axis=0)
        hkl = hkl[inrange, :]
        tth = tth[inrange]
        # Caluclate structure factor **2
        inten = self.intensity(hkl)

        if units is None:
            units = self._powder_units
        units = units.lower()
        if units in ['d', 'dspc', 'dspace', 'd space', 'd-space', 'dspacing', 'd spacing', 'd-spacing']:
            unit_str = 'd-spacing'
            unit = fc.caldspace(tth, energy_kev)
        elif units in ['q', 'wavevector']:
            unit_str = 'Q'
            unit = fc.calqmag(tth, energy_kev)
        else:
            unit_str = 'TwoTheta'
            unit = tth

        fmt = '(%3.0f,%3.0f,%3.0f) %10.2f %10.2f  %5.2f (%3.0f) %5.2f (%3.0f) %5.2f (%3.0f) %5.2f (%3.0f)\n'
        outstr = 'Tensor Scattering %s\n' % self.xtl.name
        outstr += 'Process: %s, site: %s\n' % (process, atom_label)
        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Psi_0 = (%.2g,%.2g,%.2g)\n' % (azir[0], azir[1], azir[2])
        outstr += '( h, k, l)    %10s         I0  Sigma-Sigma    Sigma-Pi    Pi-Sigma       Pi-Pi\n' % unit_str

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        psideg = np.arange(0, 361, 5)
        for n in range(1, len(tth)):
            ss, sp, ps, pp = ts.CalculateIntensityInPolarizationChannels(
                process, B, sitevec, sglist, lam, hkl[n, :], azir, psideg,
                K=rank, Time=time, Parity=parity, mk=mk, lk=lk, sk=sk
            )
            iss = np.argmax(ss)
            isp = np.argmax(sp)
            ips = np.argmax(ps)
            ipp = np.argmax(pp)
            ss = ss[iss]
            sp = sp[isp]
            ps = ps[ips]
            pp = pp[ipp]
            psi_ss = psideg[iss]
            psi_sp = psideg[isp]
            psi_ps = psideg[ips]
            psi_pp = psideg[ipp]
            outstr += fmt % (hkl[n, 0], hkl[n, 1], hkl[n, 2], unit[n], inten[n],
                             ss, psi_ss, sp, psi_sp, ps, psi_ps, pp, psi_pp)
        outstr += 'Reflections: %1.0f\n' % len(tth)
        return outstr
    '''

class ScatteringTypes:
    """
    Container for available scattering types
    """
    def __init__(self, parent, typedict):
        self.parent = parent
        self.typedict = typedict

        typenames = typedict.keys()

        for name in typenames:
            attrname = name.replace(' ','_')
            setattr(self, attrname, self.ScatteringType(parent, name))

    def __call__(self, type):
        """
        Calling this container with a scattering type will set the scattering type in the parent class
        :param type: str scattering type
        :return: None
        """
        for typename in self.typedict:
            if type.lower() in self.typedict[typename]:
                self.parent._scattering_type = typename
                print('Changed scattering type to: %s'%typename)

    def i16(self):
        """
        Set max/min angles consistent with reflection geometry on beamline I16
        :return: None
        """
        self.parent.setup_scatter(type='xray', energy_kev=8.0, min_theta=-20, max_theta=140, min_twotheta=0, max_twotheta=140)

    def wish(self):
        """
        Set max/min angles consistent with reflection geometry on beamline I16
        :return: None
        """
        self.parent.setup_scatter(type='neutron', wavelength_a=1.5, min_theta=-180, max_theta=180, min_twotheta=-180, max_twotheta=180)

    class ScatteringType:
        """
        Container for scattering type switcher
        """
        def __init__(self, parent, typename):
            self.parent = parent
            self.typename = typename

        def __call__(self):
            self.parent._scattering_type = self.typename
