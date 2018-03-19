# -*- coding: utf-8 -*-
"""
Scattering Class "classes_scattering.py"
    Scattering functions for Crystal class.

By Dan Porter, PhD
Diamond
2017

Version 1.1
Last updated: 06/01/18

Version History:
10/09/17 0.1    Program created
30/10/17 1.0    Main functions finshed, some testing done
06/01/18 1.1    Renamed classes_scattering.py

@author: DGPorter
"""

import sys,os
import numpy as np

from Dans_Diffraction import functions_general as fg
from Dans_Diffraction import functions_crystallography as fc

class Scattering:
    """
    Simulate diffraction from Crystal
    Useage:
        xtl = Crystal()
        xtl.Scatter.setup_scatter(type='x-ray',energy_keV=8.0)
        xtl.intensity([h,k,l]) # Returns intensity
        xtl.print_all_refelctions() # Returns formated string of all allowed reflections
        
        Allowed radiation types:
            'xray','neutron','xray magnetic','neutron magnetic','xray resonant'
    """
    
    #------Options-------
    # Standard Options
    _scattering_type = 'xray' # 'xray','neutron','xray magnetic','neutron magnetic','xray resonant'
    _scattering_specular_direction = [0,0,1] # reflection
    _scattering_parallel_direction = [0,0,1] # transmission
    _scattering_theta_offset = 0.0
    _scattering_min_theta = -180.0
    _scattering_max_theta = 180.0
    _scattering_min_twotheta = -180.0
    _scattering_max_twotheta = 180.0
    
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
    _polarisation_vector_incident = [1,0,0]
    _polarisation_vector_emitted = [1,0,0]
    
    # Radiation energy
    _energy_kev = fg.Cu
    
    # Resonant X-ray Options
    _azimuthal_angle = 0
    _azimuthal_reference = [1,0,0]
    _resonant_approximation_e1e1 = True
    _resonant_approximation_e2e2 = False
    _resonant_approximation_e1e2 = False
    _resonant_approximation_m1m1 = False
    
    def __init__(self,xtl):
        "initialise"
        self.xtl = xtl
    
    def x_ray(self,HKL):
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
            dw = fc.debyewaller(uiso,Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calcualtion not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
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
    
    def x_ray_fast(self,HKL):
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
    
    def neutron(self,HKL):
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
    
    def magnetic_neutron(self,HKL):
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
            ff = np.ones([len(HKL),Nat])
        
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
    
    def xray_magnetic(self,HKL):
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
    
    def xray_resonant(self,HKL,energy_kev=None,polarisation='sp',F0=1,F1=1,F2=1,azim_zero=[1,0,0],PSI=[0],disp=False):
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
    
    def xray_resonant_scattering_factor(self,HKL,energy_kev=None,polarisation='sp',F0=1,F1=1,F2=1,azim_zero=[1,0,0],psi=0,disp=False):
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
            tthr = np.deg2rad(tth[ref])
            
            if polarisation == 'ss':    # Sigma-Sigma
                f0 = 1*np.ones(Nat)
                f1 = 0*np.ones(Nat)
                f2 = z2**2
            elif polarisation == 'sp':  # Sigma-Pi
                f0 = 0*np.ones(Nat)
                f1 = z1*np.cos(tthr) + z3*np.sin(tthr)
                f2 = -z2*(z1*np.sin(tthr)-z3*np.cos(tthr))
            elif polarisation == 'ps':  # Pi-Sigma
                f0 = 0*np.ones(Nat)
                f1 = z3*np.sin(tthr)-z1*np.cos(tthr)
                f2 = z2*(z1*np.sin(tthr)+z3*np.cos(tthr))
            elif polarisation == 'pp':  # Pi-Pi
                f0 = np.cos(2*tthr)*np.ones(Nat)
                f1 = -z2*np.sin(2*tthr)
                f2 = -(np.cos(tthr)**2)*(z1**2*np.tan(tthr)**2 + z3**2)
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
    
    def scatteringcomponents(self,mxmymz,hkl,azim_zero=[1,0,0],psi=0):
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
        z1z2z3 = np.dot(mxmymz,U.T) # [mxmymz.I, mxmymz.J, mxmymz.Q]
        return fg.norm(z1z2z3)
    
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
    
    def print_intensity(self,HKL):
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
        print('( h, k, l)   Neutron      xray   Magn. N  Magn. XR')
        for n in range(len(HKL)):
            vals=(HKL[n][0],HKL[n][1],HKL[n][2],IN[n],IX[n],INM[n],IXM[n],IXRss[n],IXRsp[n],IXRps[n],IXRpp[n])
            print(fmt%vals)
    
    def intensity(self,HKL):
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
        
        if self._scattering_type.lower() in ['xray','x','x-ray','thomson','charge']:
            return self.x_ray(HKL)
        elif self._scattering_type.lower() in ['neutron','n','nuclear']:
            return self.neutron(HKL)
        elif self._scattering_type.lower() in ['xray magnetic','magnetic xray','spin xray','xray spin']:
            return self.xray_magnetic(HKL)*1e4
        elif self._scattering_type.lower() in ['neutron magnetic','magnetic neutron','magnetic']:
            return self.magnetic_neutron(HKL)*1e4
        elif self._scattering_type.lower() in ['xray resonant','resonant','resonant xray','rxs']:
            return self.xray_resonant(HKL)
        else:
            print('Scattering type not defined')
    
    def hkl(self,HKL,energy_kev=None):
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
    
    def hkl_reflection(self,HKL,energy_kev=None):
        " Calculate the theta, two-theta and intensity of the given HKL in reflection geometry, display the result"
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=np.float).reshape([-1,3])
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        theta = self.xtl.Cell.theta_reflection(HKL, energy_kev, self._scattering_specular_direction,self._scattering_theta_offset)
        inten = self.intensity(HKL)
        
        print('Energy = %6.3f keV' % energy_kev)
        print('Specular Direction = (%1.0g,%1.0g,%1.0g)' %(self._scattering_specular_direction[0],self._scattering_specular_direction[1],self._scattering_specular_direction[2]))
        print('( h, k, l)    Theta TwoTheta  Intensity')
        for n in range(len(tth)):
            print('(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f' % (HKL[n,0],HKL[n,1],HKL[n,2],theta[n],tth[n],inten[n]))
    
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
    
    def setup_scatter(self,type=None,energy_kev=None,wavelength_a=None,
                      specular=None,parallel=None,theta_offset=None,
                      min_theta=None,max_theta=None,min_twotheta=None,max_twotheta=None):
        """
        Simple way to set scattering parameters, each parameter is internal to xtl (self)
        
        type        : self._scattering type               : 'xray','neutron','xray magnetic','neutron magnetic','xray resonant'
        energy_kev  : self._energy_kev                    : radiation energy in keV
        wavelength_a: self._wavelength_a                  : radiation wavelength in Angstrom
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
        
        if specular is not None:
            self._scattering_specular_direction = specular
        
        if parallel is not None:
            self._scattering_parallel_direction
        
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
        print('  Specular Direction (reflection): (%2.0f,%2.0f,%2.0f)'%(self._scattering_specular_direction[0],self._scattering_specular_direction[1],self._scattering_specular_direction[2]))
        print('Parallel Direction (transmission): (%2.0f,%2.0f,%2.0f)'%(self._scattering_parallel_direction[0],self._scattering_parallel_direction[1],self._scattering_parallel_direction[2]))
        print('                   Sample Offset : %5.2f'%(self._scattering_theta_offset))
        print('             Minimum Theta angle : %5.2f'%(self._scattering_min_theta))
        print('             Maximum Theta angle : %5.2f'%(self._scattering_max_theta))
        print('         Minimum Two-Theta angle : %5.2f'%(self._scattering_min_twotheta))
        print('         Maximum Two-Theta angle : %5.2f'%(self._scattering_max_twotheta))
    
    def print_all_reflections(self,energy_kev=None,print_symmetric=False,min_intensity=0.01,max_intensity=None):
        """
        Prints a list of all allowed reflections at this energy
            energy_kev = energy in keV
            print_symmetric = False*/True : omits reflections with the same intensity at the same angle
            min_intensity = None/ 0.01 : omits reflections less than this (remove extinctions)
            max_intensity = None/ 0.01 : omits reflections greater than this (show extinctions only)
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        if min_intensity == None: min_intensity=-1
        if max_intensity == None: max_intensity=np.inf
        
        HKL = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        HKL = self.xtl.Cell.sort_hkl(HKL)
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        #inten = np.sqrt(self.intensity(HKL)) # structure factor
        inten = self.intensity(HKL)
        
        fmt = '(%3.0f,%3.0f,%3.0f) %8.2f  %9.2f\n' 
        outstr = ''
        
        outstr+= 'Energy = %6.3f keV\n' % energy_kev
        outstr+= 'Radiation: %s\n' % self._scattering_type
        outstr+= '( h, k, l)    TwoTheta  Intensity\n'
        #outstr+= fmt % (HKL[0,0],HKL[0,1],HKL[0,2],tth[0],inten[0]) # hkl(0,0,0)
        count = 0
        for n in range(1,len(tth)):
            if inten[n] < min_intensity: continue
            if inten[n] > max_intensity: continue
            if not print_symmetric and np.abs(tth[n]-tth[n-1]) < 0.01: continue # only works if sorted
            count += 1
            outstr+= fmt % (HKL[n,0],HKL[n,1],HKL[n,2],tth[n],inten[n])
        outstr+= 'Reflections: %1.0f\n'%count
        return outstr
    
    def print_ref_reflections(self,energy_kev=None,min_intensity=0.01,max_intensity=None):
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
    