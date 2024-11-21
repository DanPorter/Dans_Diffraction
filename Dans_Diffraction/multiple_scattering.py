"""
Multiple Scattering code, By Dr Gareth Nisbet
For more information see:
Nisbet, A. G. A., Beutier, G., Fabrizi, F., Moser, B. & Collins, S. P. (2015). Acta Cryst. A71, 20-25.
http://dx.doi.org/10.5281/zenodo.12866

Example:
    xtl = dif.Crystal('Diamond.cif')
    mslist = run_calcms(xtl, [0,0,3], [0,1,0], [1,0], [2.83, 2.85], plot=True)

Created from python package "calcms"
Version 1.2
21/11/2024
 -------------------------------------------
 Copyright 2014 Diamond Light Source Ltd.123

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 Dr Gareth Nisbet, gareth.nisbet@diamond.ac.uk Tel: +44 1235 778786
 www.diamond.ac.uk
 Diamond Light Source, Chilton, Didcot, Oxon, OX11 0DE, U.K.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools


__version__ = '1.2'


def run_calcms(xtl, hkl, azir=[0, 0, 1], pv=[1, 0], energy_range=[7.8, 8.2], numsteps=60,
               full=False, pv1=False, pv2=False, sfonly=True, pv1xsf1=False):
    """
    Run the multiple scattering code

    mslist = run_calcms(xtl, [0,0,1])

    :param xtl: Crystal structure from Dans_Diffraction
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

    # ===============================================================================
    #                         DMS Calculation
    # ===============================================================================

    mslist = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]

    # ================= Generate Reflist from Cif ===================================
    sf, reflist, lattice, structure = loadcif(xtl, energy_range[-1])
    refindex = ~np.isnan(Vfind(reflist, np.round(hkl) - reflist).vindex())
    sf = sf[refindex]
    reflist = reflist[refindex]
    sf2 = sf[Vfind(reflist, np.round(hkl) - reflist).vindex()]
    loopnum = 1

    # ------------------------------------------------------------------------------
    if pv1 + pv2 + sfonly + full + pv1xsf1 > 1:
        print('Choose only one intensity option')
        print('full=%s, pv1=%s, pv2=%s, sfonly=%s, pv1xsf1=%s' % (full, pv1, pv2, sfonly, pv1xsf1))
        return None
    elif pv1 + pv2 + sfonly + full + pv1xsf1 == 0:
        print('Geometry Only')
        mslist = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]

    for enval in np.linspace(energy_range[0], energy_range[1], numsteps):
        print(str(loopnum) + ' of ' + str(numsteps))

        # ===========================================================================
        # SF0*Gauss*SF1*SF2*PV2
        # ===========================================================================
        if full:
            #print('full calculation: SF1*SF2*PV2')
            ms = Calcms(lattice, hkl, hkl, reflist, enval, azir, sf, sf2)  # [:,[3,4,5]]
            polfull = ms.polfull(pv)
            mslist = np.concatenate((mslist, ms.polfull(pv)), 0)
        # ===========================================================================
        # PV1 only
        # ===========================================================================
        elif pv1:
            #print('pv1 calculation: PV1')
            ms = Calcms(lattice, hkl, hkl, reflist, enval, azir, sf, sf2)
            mslist = np.concatenate((mslist, ms.pol1only(pv)), 0)
        # ===========================================================================
        # PV2 only
        # ===========================================================================
        elif pv2:
            #print('pv2 calculation: PV2')
            ms = Calcms(lattice, hkl, hkl, reflist, enval, azir, sf, sf2)
            mslist = np.concatenate((mslist, ms.pol2only(pv)), 0)
        # ===========================================================================
        # SF only
        # ===========================================================================
        elif sfonly:
            #print('sfonly calculation: SF1*SF2')
            ms = Calcms(lattice, hkl, hkl, reflist, enval, azir, sf, sf2)
            mslist = np.concatenate((mslist, ms.sfonly()), 0)

        # ===========================================================================
        # SF only
        # ===========================================================================
        elif pv1xsf1:
            #print('pv1xsf1 calculation: SF1*PV1')
            ms = Calcms(lattice, hkl, hkl, reflist, enval, azir, sf)
            mslist = np.concatenate((mslist, ms.pv1xsf1(pv)), 0)
        # ===========================================================================
        # Geometry only - no structure factors
        # ===========================================================================
        else:
            print('Geometry Only')
            ms = Calcms(lattice, hkl, hkl, reflist, enval, azir)
            mslist = np.concatenate((mslist, ms.geometry()), 0)

        loopnum = loopnum + 1

    keepindex = np.where([~np.isnan(mslist).any(1)])[1]
    mslist = np.array(mslist[keepindex, :])
    return mslist


########################################################################################################################
###############################################  Ancillary Functions  ##################################################
########################################################################################################################

def loadcif(xtl, energy_kev):
    """
    New loadcif from Dans_Diffraction
    returns:
        intensity: Structure factor^2. I = sf x sf*
        reflist: array of [h,k,l] reflections
        lattice: [a,b,c,alpha,beta,gamma]
        sf: complex structure factors
    """
    lattice = xtl.Cell.lp()
    reflist = xtl.Cell.all_hkl(energy_kev)
    reflist = xtl.Cell.sort_hkl(reflist)
    reflist = reflist[1:]
    old_sf = xtl.Scatter._return_structure_factor
    xtl.Scatter._return_structure_factor = True
    sf = xtl.Scatter.intensity(reflist)  # complex structure factor
    xtl.Scatter._return_structure_factor = old_sf
    intensity = np.real(sf * np.conj(sf))
    print('MS Reflections: %d' % len(reflist))
    return intensity, reflist, lattice, sf


class Bmatrix(object):
    """ Convert to Cartesian coordinate system. Returns the Bmatrix and the metric tensors in direct and reciprocal spaces"""

    def __init__(self, lattice):
        self.lattice = lattice
        lattice = self.lattice
        a = lattice[0]
        b = lattice[1]
        c = lattice[2]
        alph = lattice[3]
        bet = lattice[4]
        gamm = lattice[5]
        alpha1 = alph * np.pi / 180.0
        alpha2 = bet * np.pi / 180.0
        alpha3 = gamm * np.pi / 180.0
        beta1 = np.arccos((np.cos(alpha2) * np.cos(alpha3) - np.cos(alpha1)) / (np.sin(alpha2) * np.sin(alpha3)))
        beta2 = np.arccos((np.cos(alpha1) * np.cos(alpha3) - np.cos(alpha2)) / (np.sin(alpha1) * np.sin(alpha3)))
        beta3 = np.arccos((np.cos(alpha1) * np.cos(alpha2) - np.cos(alpha3)) / (np.sin(alpha1) * np.sin(alpha2)))
        b1 = 1. / (a * np.sin(alpha2) * np.sin(beta3))
        b2 = 1. / (b * np.sin(alpha3) * np.sin(beta1))
        b3 = 1. / (c * np.sin(alpha1) * np.sin(beta2))
        c1 = b1 * b2 * np.cos(beta3)
        c2 = b1 * b3 * np.cos(beta2)
        c3 = b2 * b3 * np.cos(beta1)
        self.bmatrix = np.matrix([[b1, b2 * np.cos(beta3), b3 * np.cos(beta2)],
                                  [0.0, b2 * np.sin(beta3), -b3 * np.sin(beta2) * np.cos(alpha1)], [0.0, 0.0, 1. / c]])

    def bm(self):
        return self.bmatrix

    def ibm(self):
        return self.bmatrix.I

    def mt(self):
        return self.bmatrix.I * self.bmatrix.transpose().I

    def rmt(self):
        mt = self.bmatrix.I * self.bmatrix.transpose().I
        return mt.I


class Rotxyz(object):
    """Example p = Rotxyz(initial_vector, vectorrotateabout, angle)"""

    def __init__(self, u, angle):
        self.u = u
        self.angle = angle
        u = np.matrix(self.u) / np.linalg.norm(np.matrix(self.u))
        e11 = u[0, 0] ** 2 + (1 - u[0, 0] ** 2) * np.cos(angle * np.pi / 180.0)
        e12 = u[0, 0] * u[0, 1] * (1 - np.cos(angle * np.pi / 180.0)) - u[0, 2] * np.sin(angle * np.pi / 180.0)
        e13 = u[0, 0] * u[0, 2] * (1 - np.cos(angle * np.pi / 180.0)) + u[0, 1] * np.sin(angle * np.pi / 180.0)
        e21 = u[0, 0] * u[0, 1] * (1 - np.cos(angle * np.pi / 180.0)) + u[0, 2] * np.sin(angle * np.pi / 180.0)
        e22 = u[0, 1] ** 2 + (1 - u[0, 1] ** 2) * np.cos(angle * np.pi / 180.0)
        e23 = u[0, 1] * u[0, 2] * (1 - np.cos(angle * np.pi / 180.0)) - u[0, 0] * np.sin(angle * np.pi / 180.0)
        e31 = u[0, 0] * u[0, 2] * (1 - np.cos(angle * np.pi / 180.0)) - u[0, 1] * np.sin(angle * np.pi / 180.0)
        e32 = u[0, 1] * u[0, 2] * (1 - np.cos(angle * np.pi / 180.0)) + u[0, 0] * np.sin(angle * np.pi / 180.0)
        e33 = u[0, 2] ** 2 + (1 - u[0, 2] ** 2) * np.cos(angle * np.pi / 180.0)
        self.rotmat = np.matrix([[e11, e12, e13], [e21, e22, e23], [e31, e32, e33]])

    def rmat(self):
        return self.rotmat


class Dhkl(object):
    """calculate d-spacing for reflection from reciprocal metric tensor
    d = Dhkl(lattice,HKL)
    lattice = [a b c alpha beta gamma] (angles in degrees)
    HKL: list of hkl. size(HKL) = n x 3 or 3 x n
    !!! if size(HKL) is 3 x 3, HKL must be in the form:
    HKL = [h1 k1 l1 ; h2 k2 l2 ; h3 k3 l3]
    """

    def __init__(self, lattice, hkl):
        self.lattice = lattice
        self.hkl = np.matrix(hkl)

    def d(self):
        hkl = self.hkl
        if np.shape(hkl)[0] == 3 and np.shape(hkl)[1] != 3:
            hkl = hkl.transpose()
            T = 1
        else:
            T = 0
        G = Bmatrix(self.lattice).mt()
        d = 1. / np.sqrt(np.diagonal(hkl * (G.I * hkl.transpose())))
        # d = 1/np.sqrt(hkl*G.I*hkl.T)
        if T == 1:
            d = d.transpose()
        return d


class Interplanarangle(object):
    def __init__(self, lattice, hkl1, hkl2):
        """ calculates interplanar angles in degrees for reflections using the metric tensor
        Example Interplanarangle(lattice,hkl,hkl2) where hkl and hkl2 must have the same column length
        Interplanarangle([3,3,3,90,90,120],[[1,2,3],[1,2,3]],[[1,1,3],[1,2,3]])
        """
        self.lattice = lattice
        if len(hkl1) != len(hkl2):
            hkl1 = np.zeros((len(hkl2), 3)) + hkl1
        self.hkl1 = np.matrix(hkl1)
        self.hkl2 = np.matrix(hkl2)

    def ang(self):
        G = Bmatrix(self.lattice).mt()
        dhkl1 = Dhkl(self.lattice, self.hkl1).d()
        dhkl2 = Dhkl(self.lattice, self.hkl2).d()
        term1 = np.diagonal(self.hkl1 * (G.I * self.hkl2.transpose()))
        return np.arccos(np.multiply((term1 * dhkl1), dhkl2)) * 180 / np.pi


class Bragg(object):
    def __init__(self, lattice, hkl, energy):
        """ returns Bragg angle of a reflection
        theta = Bragg(lattice,hkl,energy)
        """
        self.lattice = lattice
        self.hkl = hkl
        self.energy = energy

    def th(self):
        keV2A = 12.3984187
        wl = keV2A / self.energy
        d = Dhkl(self.lattice, self.hkl).d()
        #        if wl/2.0/d <= 1:
        theta = 180 / np.pi * np.arcsin(wl / 2.0 / d)
        #        else:
        #            theta = np.nan;
        return theta


class Hklgen(object):
    def __init__(self, depth):
        self.depth = depth

    def v(self):
        depth = self.depth
        reflist = np.zeros((((2 * depth) + 1) ** 3) * 3).reshape(((((2 * depth) + 1) ** 3) * 3) / 3, 3)
        list1 = [x + 1 for x in range(-depth - 1, depth)]
        clist = itertools.cycle(list1)
        for hh in range(depth, (((2 * depth) + 1) ** 3) - depth, (2 * depth + 1)):  # 2 times depth +1
            reflist[[hh + x + 1 for x in range(-depth - 1, depth)], 0] = [x + 1 for x in range(-depth - 1, depth)]
        for kk in range(depth, (((2 * depth) + 1) ** 3) - depth, (2 * depth + 1)):  # 2 times depth +1
            reflist[[kk + x + 1 for x in range(-depth - 1, depth)], 1] = clist.next()
        for kk in range(depth, (((2 * depth) + 1) ** 3) - depth, (2 * depth + 1)):  # 2 times depth +1
            reflist[[kk + x + 1 for x in range(-depth - 1, depth)], 2] = clist.next()
        reflist[:, 2].sort()
        return reflist.astype(int)


class Vfind(object):
    def __init__(self, vlist, v):
        #         result1=list(np.where(vlist-v==0)[0])
        #         self.refindex=[x if result1.count(x) >= 3 else np.nan for x in result1]
        v = np.array(v)
        refindex2 = []
        for i1 in range(v.shape[0]):
            result1 = list(np.where(vlist - v[i1, :] == 0)[0])
            try:
                refindex = [x for x in result1 if result1.count(x) >= 3][0]
            except:
                refindex = np.nan
            refindex2.append(refindex)
        self.refindex = refindex2

    def vindex(self):
        return self.refindex


########################################################################################################################
#####################################################  Calcms  #########################################################
########################################################################################################################


class Calcms(object):
    def __init__(self, lattice, hkl, hklint, hkl2, energy, azir, F=[], F2=[]):
        self.F = np.matrix(F)
        self.F2 = np.matrix(F2)
        self.lattice = lattice
        self.hkl = np.matrix(hkl)
        self.hkl2 = np.matrix(hkl2)
        self.hkl3 = hklint - self.hkl2
        self.energy = energy
        self.azir = np.matrix(azir)
        bm = Bmatrix(self.lattice).bm()
        #   Convert primary hkl and reduced hkl2 list to orthogonal coordinate system
        hklnotlist = (bm * self.hkl.transpose()).transpose()
        self.hklrlv = hklnotlist
        azir2 = (bm * self.azir.transpose()).transpose()
        zref = (bm * np.matrix([0, 0, 1]).transpose()).transpose()
        #   Determin transformation to align primary reflection to the z direction
        alignangle = Interplanarangle(self.lattice, [0, 0, 1], self.hkl).ang()
        realvecthkl = (bm * self.hkl2.transpose()).transpose()
        realvecthkl3 = (bm * self.hkl3.transpose()).transpose()
        rotvect = np.cross(zref, hklnotlist)
        if np.abs(rotvect[0][0]) + np.abs(rotvect[0][1]) + np.abs(rotvect[0][2]) >= 0.0001:
            realvecthkl = realvecthkl * Rotxyz(rotvect, alignangle[0]).rmat()
            self.tvprime = hklnotlist * Rotxyz(rotvect, alignangle[0]).rmat()
        else:
            self.tvprime = hklnotlist
        # Build Ewald Sphere
        # brag1 = np.empty(self.hkl2.shape[0]) * 0 + 1.0 * Bragg(self.lattice, self.hkl, self.energy).th()
        hkl_bragg = Bragg(self.lattice, self.hkl, self.energy).th()
        brag1 = hkl_bragg * np.ones(len(self.hkl2))  # Dan 26/7/2024
        self.brag1 = brag1
        keV2A = 12.398
        ko = (self.energy / keV2A)
        self.ko = ko
        #   height dependent radius of ewald slice in the hk plane
        # Dan: rewl is a (nxn) matrix, not sure if this is intentional...
        rewl = ko * np.cos((np.arcsin(
            ((ko * np.sin(-brag1 * np.pi / 180.0)) + (realvecthkl[:, 2])) / ko) * 180.0 / np.pi) * np.pi / 180.0)
        # Dan: this fix below results in the calculation looking wrong...
        # height = np.array(realvecthkl[:, 2]).reshape(-1)
        # rewl = ko * np.cos((np.arcsin(
        #     ((ko * np.sin(-brag1 * np.pi / 180.0)) + height) / ko) * 180.0 / np.pi) * np.pi / 180.0)
        rhk = np.sqrt(np.square(realvecthkl[:, 0]) + np.square(realvecthkl[:, 1]))
        #   Origin of intersecting circle
        orighk = np.empty(self.hkl2.shape[0]) * 0 + (ko * np.cos(brag1[0] * np.pi / 180.))
        ####################### MS Calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if np.abs(rotvect[0][0]) + np.abs(rotvect[0][1]) + np.abs(rotvect[0][2]) > 0.001:
            azir2 = azir2 * Rotxyz(rotvect, alignangle[0]).rmat()
        azirangle = np.arctan2(azir2[0, 0], azir2[0, 1]) * 180.0 / np.pi
        rhkangle = np.arctan2((realvecthkl[:, 0]), (realvecthkl[:, 1])) * 180.0 / np.pi
        yhkintercept = np.divide(np.square(orighk) - np.square(rhk) + np.square(rewl), (2.0 * orighk)) - orighk
        xintercept = np.sqrt(np.square(rewl) - np.square(
            np.divide((np.square(orighk) - np.square(rhk) + np.square(rewl)), 2.0 * orighk)))
        #        realindex1 = np.where(yhkintercept.imag!=0)
        #        realindex2 = np.where(xintercept.imag!=0)
        #        realindex=[realindex1,realindex2]
        interceptangle1 = np.arctan2(xintercept, yhkintercept) * 180.0 / np.pi
        interceptangle2 = np.arctan2(-xintercept, yhkintercept) * 180.0 / np.pi  # with respect to the real space origin
        #        self.ewpsi1=np.arctan2(xintercept,yhkintercept-orighk)*180.0/np.pi
        self.ewpsi1 = interceptangle1 + rhkangle
        self.ewpsi2 = interceptangle2 + rhkangle
        psirotate = (interceptangle1 + azirangle - rhkangle)
        psirotate2 = (interceptangle2 + azirangle - rhkangle)
        self.interceptangle1 = interceptangle1 - rhkangle
        self.interceptangle2 = interceptangle2 - rhkangle
        self.rhkangle = rhkangle
        ########## return hkl back to original coordinate system ##############
        psi1 = (np.mod(psirotate + 180.0, 360.0) - 180.0)
        psi1 = psi1[:, 0]
        psi2 = (np.mod(psirotate2 + 180.0, 360.0) - 180.0)
        psi2 = psi2[:, 0]
        brag1 = np.matrix(brag1).transpose()
        braga = np.array(brag1)[0]
        self.kov1 = np.array((Rotxyz([1, 0, 0], -np.array(braga)[0]).rmat() * np.matrix([[0, self.ko, 0]]).T).T)
        self.psi1 = psi1
        self.psi2 = psi2
        self.bragg1 = brag1
        energyl = np.matrix(np.ones(psi1.shape[0]) * energy).T
        if len(F) == 0:
            self.fullarray = np.array(np.concatenate((hkl2, psi1, psi2, brag1, energyl), 1))
        else:
            self.fullarray = np.array(np.concatenate((hkl2, psi1, psi2, brag1, (self.F).T, energyl), 1))
        self.realvecthkl = realvecthkl
        self.realvecthkl3 = realvecthkl3
        self.ko = ko

    def tv(self):
        return self.realvecthkl

    def tvt(self):
        return self.realvecthkl3

    def rhkangle(self):
        return self.rhkangle

    def prlv(self):
        return self.hklrlv

    def kov(self):
        return self.kov1

    def ko(self):
        return self.ko

    def psi(self):
        return np.concatenate((self.psi1[:, 0], self.psi2[:, 0]), 1)
        # return self.psi1, self.psi2

    def ewpsi(self):
        return self.ewpsi1, self.ewpsi2

    def bragg(self):
        return np.array(self.bragg1)

    def full(self):
        """ returns hkl2,psi1,psi2,brag1,energ """
        return self.fullarray

    def trv(self):
        """ returns transformed and rotated vectors. """
        trvarray = np.array(
            [Rotxyz([0, 0, 1], np.array(self.ewpsi1[i1, :])[0][0]).rmat() * self.realvecthkl[i1, :].T for i1 in
             range(self.ewpsi1.shape[0])])
        trvarray2 = np.array(
            [Rotxyz([0, 0, 1], np.array(self.ewpsi2[i1, :])[0][0]).rmat() * self.realvecthkl[i1, :].T for i1 in
             range(self.ewpsi2.shape[0])])
        return np.matrix(np.squeeze(trvarray)), np.matrix(np.squeeze(trvarray2))

    def trvt(self):
        """ returns transformed and rotated tertiary vectors. """
        trvarrayt = np.array(
            [Rotxyz([0, 0, 1], np.array(self.ewpsi1[i1, :])[0][0]).rmat() * self.realvecthkl3[i1, :].T for i1 in
             range(self.ewpsi1.shape[0])])
        trvarray2t = np.array(
            [Rotxyz([0, 0, 1], np.array(self.ewpsi2[i1, :])[0][0]).rmat() * self.realvecthkl3[i1, :].T for i1 in
             range(self.ewpsi2.shape[0])])
        return np.matrix(np.squeeze(trvarrayt)), np.matrix(np.squeeze(trvarray2t))

    def bvects(self):
        """ returns secondary beam vectors """
        return self.trv()[0] + self.kov1, self.trv()[1] + self.kov1

    def bvects2(self):
        """ returns tertiary beam vectors """
        return self.trvt()[0] + self.bvects()[0], self.trvt()[1] + self.bvects()[1]

    def angs(self):
        """ Angles between ko and beam vectors """
        norms1 = np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        angs1 = np.arccos(
            (np.matrix(-self.kov()) * np.matrix(self.bvects()[0]).T) /
            (np.linalg.norm(self.kov()) * norms1)) * 180.0 / np.pi
        norms2 = np.apply_along_axis(np.linalg.norm, 1, self.bvects()[1])
        angs2 = np.arccos(
            (np.matrix(-self.kov()) * np.matrix(self.bvects()[1]).T) /
            (np.linalg.norm(self.kov()) * norms2)) * 180.0 / np.pi
        return angs1, angs2

    def psiplaneang(self):
        """ Angle required to rotate k1 about ko onto the secondary scattering plane """
        v1 = np.matrix([[1, 0, 0]])  # determines slice direction of interplanerangle function
        norms1 = np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        nbv = (self.bvects()[0].T / norms1).T  # normalized beam vectors
        v2 = np.cross(-self.kov(), nbv)
        psiangs = Interplanarangle([1, 1, 1, 90, 90, 90], v1, v2).ang()
        return psiangs

    def psiplaneang2(self):
        """ Angle required to rotate k2 about k1 onto the tertiary scattering plane """
        norms1 = np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        norms2 = np.apply_along_axis(np.linalg.norm, 1, self.bvects2()[0])
        nbv1 = np.cross(-self.kov(), (self.bvects()[0].T / norms1).T)
        nbv2 = np.cross((self.bvects()[0].T / norms1).T, (self.bvects2()[0].T / norms2).T)
        psiangs2 = Interplanarangle([1, 1, 1, 90, 90, 90], nbv1, nbv2).ang()
        return psiangs2

    def pol(self, polv):
        """ returns hkl2, sig, pi, pfactor   """
        refs = self.fullarray[:, [0, 1, 2]]
        brags = Bragg(self.lattice, refs, self.energy).th()
        psiang = self.psiplaneang()
        pmtmpv = np.array(np.squeeze([(np.matrix([[1, 0], [0, np.cos(2 * brags[i1] * np.pi / 180.0)]]) *
                                       np.matrix(
                                           [[np.cos(psiang[i1] * np.pi / 180.0), np.sin(psiang[i1] * np.pi / 180.0)],
                                            [-np.sin(psiang[i1] * np.pi / 180.0),
                                             np.cos(psiang[i1] * np.pi / 180.0)]]) * np.matrix(polv).T).T
                                      for i1 in range(brags.shape[0])]))
        sums = np.matrix(np.sum((pmtmpv) ** 2, 1)).T
        return np.concatenate((pmtmpv, sums), 1)

    def pol2(self, polv):
        """ returns hkl3, sig, pi, pfactor   """
        refs = self.fullarray[:, [0, 1, 2]]
        brags = Bragg(self.lattice, refs, self.energy).th()
        brags2 = Bragg(self.lattice, self.hkl - refs, self.energy).th()
        psiang = self.psiplaneang()
        psiang2 = self.psiplaneang2()
        pmtmpv2 = np.array(np.squeeze([(np.matrix([[1, 0], [0, np.cos(2 * brags2[i1] * np.pi / 180.0)]]) *
                                        np.matrix(
                                            [[np.cos(psiang2[i1] * np.pi / 180.0), np.sin(psiang2[i1] * np.pi / 180.0)],
                                             [-np.sin(psiang2[i1] * np.pi / 180.0),
                                              np.cos(psiang2[i1] * np.pi / 180.0)]]) *
                                        np.matrix([[1, 0], [0, np.cos(2 * brags[i1] * np.pi / 180.0)]]) *
                                        np.matrix(
                                            [[np.cos(psiang[i1] * np.pi / 180.0), np.sin(psiang[i1] * np.pi / 180.0)],
                                             [-np.sin(psiang[i1] * np.pi / 180.0),
                                              np.cos(psiang[i1] * np.pi / 180.0)]]) * np.matrix(polv).T).T
                                       for i1 in range(brags2.shape[0])]))
        sums2 = np.matrix(np.sum((pmtmpv2) ** 2, 1)).T
        return np.concatenate((pmtmpv2, sums2), 1)

    def pv1xsf1(self, polv):
        ampT = np.array(self.F.T) * np.array(self.pol(polv)[:, -1])
        return np.concatenate((self.full(), ampT), 1)

    def geometry(self):
        return self.full()

    def polfull(self, polv):
        ampT = np.array(self.F.T) * np.array(self.F2.T) * np.array(self.pol2(polv)[:, -1])
        return np.concatenate((self.full(), ampT), 1)

    def polfull2(self, polv):
        """ returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F   """
        return np.concatenate((self.full(), self.pol(polv)), 1)

    def sfonly(self):
        ampT = np.array(self.F.T) * np.array(self.F2.T)
        return np.concatenate((self.full(), ampT), 1)

    def pol1only(self, polv):
        ampT = self.pol(polv)[:, -1]
        return np.concatenate((self.full(), ampT), 1)

    def pol2only(self, polv):
        ampT = self.pol2(polv)[:, -1]
        return np.concatenate((self.full(), ampT), 1)

    def SF(self):
        return self.F

    def SF2(self):
        return self.F2

    def pxf(self, polv):
        return np.array(self.SF()) * np.array(self.pol(polv)[:, -1]).T

    def ov(self):
        """ returns original vector list. """
        return self.hkl2

    def orig(self):
        """ Returns reciprocal space origin. """
        return np.matrix(
            [0, self.ko * np.cos(self.brag1[0] * np.pi / 180.0), -self.ko * np.sin(self.brag1[0] * np.pi / 180.0)])

    def kv(self):
        return self.ko

    def tvp(self):
        return self.tvprime

    def ppsi(self):
        return np.concatenate((self.interceptangle1, self.interceptangle2), 1)

    def th(self):
        return np.arcsin((self.kov() + self.trv())[0][:, 2] / self.ko) * 180 / np.pi


# ===============================================================================
#         Gareth Nisbet Diamond Light Source - 29 Nov 2013
# ===============================================================================
