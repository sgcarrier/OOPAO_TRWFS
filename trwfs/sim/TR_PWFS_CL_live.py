# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""

import time
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Pyramid import Pyramid
from OOPAO.TR_Pyramid import TR_Pyramid
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix, interactionMatrix_withTR
from OOPAO.tools.displayTools import cl_plot, displayMap
# %% -----------------------     read parameter file   ----------------------------------
from trwfs.parameter_files.parameterFile_TR_PWFS_general import initializeParameterFile

param = initializeParameterFile()

# %%
plt.ion()

# %% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution=param['resolution'], \
                diameter=param['diameter'], \
                samplingTime=param['samplingTime'], \
                centralObstruction=param['centralObstruction'])

# %% -----------------------     NGS   ----------------------------------
# create the Source object
ngs = Source(optBand=param['opticalBand'], \
             magnitude=param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs * tel

tel.computePSF(zeroPaddingFactor=6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),
           extent=[tel.xPSF_arcsec[0], tel.xPSF_arcsec[1], tel.xPSF_arcsec[0], tel.xPSF_arcsec[1]])
plt.clim([-1, 3])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

src = Source(optBand='K', \
             magnitude=param['magnitude'])

# %% -----------------------     ATMOSPHERE   ----------------------------------

# create the Atmosphere object
atm = Atmosphere(telescope=tel, \
                 r0=param['r0'], \
                 L0=param['L0'], \
                 windSpeed=param['windSpeed'], \
                 fractionalR0=param['fractionnalR0'], \
                 windDirection=param['windDirection'], \
                 altitude=param['altitude'])
# initialize atmosphere
atm.initializeAtmosphere(tel)

atm.update()

plt.figure()
plt.imshow(atm.OPD * 1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

tel + atm
tel.computePSF(8)
plt.figure()
plt.imshow((np.log10(tel.PSF)), extent=[tel.xPSF_arcsec[0], tel.xPSF_arcsec[1], tel.xPSF_arcsec[0], tel.xPSF_arcsec[1]])
plt.clim([-1, 3])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()
tel.print_optical_path()
tel - atm
tel.print_optical_path()

# %%
plt.close('all')
tel.OPD_no_pupil = atm.OPD * 1000

plt.figure()
plt.imshow(tel.OPD * 1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

plt.figure()
plt.imshow(tel.OPD_no_pupil * 1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()
# %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm = DeformableMirror(telescope=tel, \
                      nSubap=20, \
                      mechCoupling=param['mechanicalCoupling'], \
                      misReg=misReg)

plt.figure()
plt.plot(dm.coordinates[:, 0], dm.coordinates[:, 1], 'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

tel * dm
# %% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel - atm

wfs = TR_Pyramid(nSubap=param['nSubaperture'],
              telescope=tel,
              modulation=param["modulation"],
              lightRatio=param['lightThreshold'],
              n_pix_separation=4,
              psfCentering=param['psfCentering'],
              postProcessing='fullFrame',
              nTheta_user_defined = param['nTheta_user_defined'])

# %% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

M2C_KL = compute_KL_basis(tel, atm, dm)

# %% ZERNIKE Polynomials
# create Zernike Object
Z = Zernike(tel, 300)
# compute polynomials for given telescope
Z.computeZernike(tel)

# mode to command matrix to project Zernike Polynomials on DM
M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical, :])) @ Z.modes

# show the first 10 zernikes
dm.coefs = M2C_zernike[:, :100]
tel * dm
# displayMap(tel.OPD)

# %% to manually measure the interaction matrix
from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *

bases = tel.OPD
bases = np.transpose(bases, axes=[2, 0, 1])
# numRemFramesPerBase = findOptimalFrames(tel, wfs, nTheta_user_defined= param["nTheta_user_defined"], bases=bases)
numRemFramesPerBase = np.zeros((300))
numRemFramesPerBase[0:100] = [11, 11, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                                  3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0]

# print(numRemFramesPerBase)
# amplitude of the modes in m
stroke = 1e-9
# Modal Interaction Matrix

# %%
M2C_zonal = np.eye(dm.nValidAct)
# zonal interaction matrix
calib_zonal = interactionMatrix_withTR(ngs=ngs, \
                                atm=atm, \
                                tel=tel, \
                                dm=dm, \
                                wfs=wfs, \
                                M2C=M2C_zonal, \
                                stroke=stroke, \
                                nMeasurements=100, \
                                noise='off',
                                custom_frames  = True,
                                custom_remove_frames = numRemFramesPerBase[0:100])

remFrames = np.unique(numRemFramesPerBase)
plt.figure()
plt.plot(np.std(calib_zonal.D, axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

# %%

# Modal interaction matrix
calib_zernike = CalibrationVault(calib_zonal.D @ M2C_zernike)

plt.figure()
plt.plot(np.sqrt(np.diag(calib_zernike.D.T @ calib_zernike.D)) / calib_zernike.D.shape[0] / ngs.fluxMap.sum())
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

# %%

tel.resetOPD()
# initialize DM commands
dm.coefs = 0
ngs * tel * dm * wfs
tel + atm

# dm.coefs[100] = -1

tel.computePSF(4)
plt.close('all')

# These are the calibration data used to close the loop
calib_CL = calib_zernike
M2C_CL = M2C_zernike

# combine telescope with atmosphere
tel + atm

# initialize DM commands
dm.coefs = 0
ngs * tel * dm * wfs

plt.show()

param['nLoop'] = 500
# allocate memory to save data
SR = np.zeros(param['nLoop'])
total = np.zeros(param['nLoop'])
residual = np.zeros(param['nLoop'])
# wfsSignal = np.arange(0, wfs.nSignal) * 0
wfsSignal = np.zeros(wfs.pyramidSignalCube.shape)
a_est = np.zeros((300))
SE_PSF = []
LE_PSF = np.log10(tel.PSF_norma_zoom)
SE_PSF_K = []

plot_obj = cl_plot(
    list_fig=[atm.OPD, tel.mean_removed_OPD, wfs.cam.frame, np.log10(wfs.get_modulation_frame(radius=30)),
              [[0, 0], [0, 0]], [dm.coordinates[:, 0], np.flip(dm.coordinates[:, 1]), dm.coefs],
              np.log10(tel.PSF_norma_zoom), np.log10(tel.PSF_norma_zoom)], \
    type_fig=['imshow', 'imshow', 'imshow', 'imshow', 'plot', 'scatter', 'imshow', 'imshow'], \
    list_title=['Turbulence OPD [m]', 'Residual OPD [m]', 'WFS Detector Plane', 'WFS Focal Plane', None, None, None,
                None], \
    list_lim=[None, None, None, [-3, 0], None, None, [-4, 0], [-5, 0]], \
    list_label=[None, None, None, None, ['Time [ms]', 'WFE [nm]'], ['DM Commands', ''],
                ['Short Exposure I Band PSF', ''], ['Long Exposure K Band PSF', '']], \
    n_subplot=[4, 2], \
    list_display_axis=[None, None, None, None, True, None, None, None], \
    list_ratio=[[0.95, 0.95, 0.1], [1, 1, 1, 1]], s=20)
# loop parameters
gainCL = 0.2
wfs.cam.photonNoise = True
display = True

reconstructor = M2C_CL @ calib_CL.M

for i in range(param['nLoop']):
    a = time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    # save turbulent phase
    turbPhase = tel.src.phase
    # propagate to the WFS with the CL commands applied
    ngs * tel * dm * wfs

    for j_mode in range(len(a_est)):
        d_idx = np.argwhere(remFrames == numRemFramesPerBase[j_mode])[0][0]
        a_est[j_mode] = (calib_CL.M @ wfsSignal[d_idx, :])[j_mode]

    correction = np.matmul(M2C_CL, a_est)

    dm.coefs = dm.coefs - gainCL * correction

    # dm.coefs = dm.coefs - gainCL * np.matmul(reconstructor, wfsSignal)
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal = wfs.signalCube
    b = time.time()
    print('Elapsed time: ' + str(b - a) + ' s')
    # update displays if required
    if display == True:
        tel.computePSF(4)
        SE_PSF.append(np.log10(tel.PSF_norma_zoom))

        if i > 15:
            src * tel
            tel.computePSF(4)
            SE_PSF_K.append(np.log10(tel.PSF_norma_zoom))

            LE_PSF = np.mean(SE_PSF_K, axis=0)

        cl_plot(list_fig=[atm.OPD, tel.mean_removed_OPD, wfs.cam.frame, np.log10(wfs.get_modulation_frame(radius=30)),
                          [np.arange(i + 1), residual[:i + 1]], dm.coefs, (SE_PSF[-1]), LE_PSF],
                plt_obj=plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break

    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    OPD = tel.OPD[np.where(tel.pupil > 0)]

    print('Loop' + str(i) + '/' + str(param['nLoop']) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' + str(
        residual[i]) + '\n')

# %%
plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')

