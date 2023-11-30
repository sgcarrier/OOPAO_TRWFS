import pyfftw #import before numpy!
import numpy as np
import imageio, os

# commom modules

import matplotlib.pyplot as plt
import time
import pickle
from pathlib import Path
plt.ion()


from OOPAO.Atmosphere       import Atmosphere

from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration  import MisRegistration
from OOPAO.Telescope        import Telescope
from OOPAO.Pyramid        import Pyramid
from OOPAO.Source           import Source
from OOPAO.phaseStats       import spectrum
# calibration modules
import matplotlib.cm as cm
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.calibration.ao_calibration import ao_calibration
# display modules
from OOPAO.tools.displayTools           import displayMap
from OOPAO.calibration.InteractionMatrix import InteractionMatrix_weightedFrames

from trwfs.tools.PWFS_CMOS_NOISE_PROOF_may2022_tools import *
from tqdm import trange
from OOPAO.TRM_Pyramid import TRM_Pyramid
from OOPAO.TR_Pyramid import TR_Pyramid
from pypet import Environment, cartesian_product
from trwfs.parameter_files.parameterFile_TR_PWFS_general import initializeParameterFile

class PWFS_CL():

    def __init__(self, minimum_data_only=True):
        self.DATA = {}
        self.minimum_data_only = minimum_data_only

    def setup(self, traj, enable_custom_frames, ao_calib_file=None):

        self.param = traj.parameters.f_to_dict(fast_access=True, short_names=True)
        # %% -----------------------     TELESCOPE   ----------------------------------

        # create the Telescope object
        self.tel = Telescope(resolution=self.param['resolution'], \
                        diameter=self.param['diameter'], \
                        samplingTime=self.param['samplingTime'], \
                        centralObstruction=self.param['centralObstruction'])

        # %% -----------------------     NGS   ----------------------------------
        # create the Source object
        self.ngs = Source(optBand=self.param['opticalBand'], \
                     magnitude=self.param['magnitude'])

        # combine the NGS to the telescope using '*' operator:
        self.ngs * self.tel

        self.tel.computePSF(zeroPaddingFactor=6)

        # %% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object
        self.atm = Atmosphere(telescope=self.tel, \
                         r0=self.param['r0'], \
                         L0=self.param['L0'], \
                         windSpeed=self.param['windSpeed'], \
                         fractionalR0=self.param['fractionnalR0'], \
                         windDirection=self.param['windDirection'], \
                         altitude=self.param['altitude'])
        # initialize atmosphere
        self.atm.initializeAtmosphere(self.tel)

        self.atm.update()

        self.tel + self.atm
        self.tel.computePSF(8)

        # %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        misReg = MisRegistration(self.param)
        # if no coordonates specified, create a cartesian dm
        self.dm = DeformableMirror(telescope=self.tel, \
                              nSubap=self.param['nSubaperture'], \
                              mechCoupling=self.param['mechanicalCoupling'], \
                              misReg=misReg)


        # %% -----------------------     Modal Basis   ----------------------------------
        # compute the modal basis
        # foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used
        # filename_M2C    = None  # name of the filename, if None a default name is used
        # # KL Modal basis
        # self.M2C_KL = compute_M2C(telescope=self.tel, \
        #                      atmosphere=self.atm, \
        #                      deformableMirror=self.dm, \
        #                      param=self.param, \
        #                      nameFolder=None, \
        #                      nameFile=None, \
        #                      remove_piston=True, \
        #                      HHtName=Noneref_cube = self.wfs.cam.cube[:, self.wfs.validSignal], \
        #                      baseName=None, \
        #                      mem_available=8.1e9, \
        #                      minimF=False, \
        #                      nmo=600, \
        #                      ortho_spm=True, \
        #                      SZ=np.int64(2 * self.tel.OPD.shape[0]), \
        #                      nZer=3, \
        #                      NDIVL=1)
        #
        # # show the first 10 KL
        # self.dm.coefs = self.M2C_KL[:, :10]
        # self.tel * self.dm
        # displayMap(self.tel.OPD)

        if not self.minimum_data_only:
            traj.f_add_result('dm_OPD', self.dm.OPD, comment="OPD of the DM, with the first 10 modes")
            traj.f_add_result('tel_OPD', self.tel.OPD, comment="OPD of the telescope, with the first 10 modes")



        # bases = generateBases(300, self.tel.resolution, 'KL', display=False, scale=False)
        # bases_flat = np.reshape(bases, (300, self.tel.resolution**2)).T
        # #self.dm.coefs = bases_flat[:, :10]
        # self.tel * self.dm
        # displayMap(self.tel.OPD)

        # plt.show(block=True)


        self.M2C_KL = compute_M2C(param=self.param, \
                          telescope=self.tel, \
                          atmosphere=self.atm, \
                          deformableMirror=self.dm, \
                          nameFolder=None, \
                          nameFile=None, \
                          remove_piston=True)

        # %% -----------------------     PYRAMID WFS   ----------------------------------
        # weights_file = "weights_file.pickle"
        # if ((weights_file) and (os.path.exists(weights_file))):
        #     self.weightsPerFramePerMode = pickle.load(open(weights_file, "rb"))
        # else:
        #     self.weightsPerFramePerMode = calcWeightsPerFramePerBaseWithDM(param, M2C)
        #     print(self.weightsPerFramePerMode.shape)
        #     if self.weightsPerFramePerMode is not None:
        #         pickle.dump(self.weightsPerFramePerMode, open(weights_file, "wb"))


        #self.weightsPerFramePerMode = np.insert(self.weightsPerFramePerMode, 0, np.ones((self.param["nTheta_user_defined"])))
        # traj.f_add_result("weightsPerFramePerMode", self.weightsPerFramePerMode,
        #                   comment="Weights for every frame for everymode")
        # make sure tel and atm are separated to initialize the PWFS
        self.tel - self.atm


        self.wfs = TR_Pyramid(nSubap=self.param['nSubaperture'], \
                              telescope=self.tel, \
                              modulation=self.param['modulation'], \
                              lightRatio=self.param['lightThreshold'], \
                              pupilSeparationRatio=self.param['pupilSeparationRatio'], \
                              calibModulation=self.param['calibrationModulation'], \
                              psfCentering=self.param['psfCentering'], \
                              edgePixel=self.param['edgePixel'], \
                              extraModulationFactor=self.param['extraModulationFactor'], \
                              postProcessing=self.param['postProcessing'],
                              nTheta_user_defined   = self.param['nTheta_user_defined'])


        # %%
        self.tel * self.wfs

        subArea = (self.wfs.telescope.D / self.wfs.nSubap) ** 2 # taille d'une sous-ouverture en m2
        traj.f_add_result('subArea', subArea, comment="Size of a sub aperture in m2")
        photons_per_subArea = self.wfs.telescope.src.nPhoton * self.wfs.telescope.samplingTime * subArea
        traj.f_add_result('photons_per_subArea', photons_per_subArea, comment="Photons per sub aperture")

        # plt.figure()
        # plt.imshow(self.wfs.cam.frame)
        # plt.colorbar()
        # plt.show(block=True)

        #refCube = self.wfs.cam.cube

        # %% -----------------------     Time-resolved frames   ----------------------------------

        # self.weightsPerFramePerMode = np.ones((306, 48))
        # if traj.enable_custom_frames == True:
        #     self.weightsPerFramePerMode[0:100, :] = findWeightsPerFramePerMode(self.tel,
        #                                                                       self.wfs,
        #                                                                       nTheta_user_defined=param["nTheta_user_defined"],
        #                                                                       numBases=100)

        self.dm.coefs = self.M2C_KL[:, 0:10] * 1e-9
        self.tel * self.dm * self.wfs
        displayMap(self.tel.OPD)

        self.dm.coefs = 0
        self.wfs.modulation = param["modulation"]
        self.tel * self.dm * self.wfs
        im = plt.imshow(self.wfs.referenceSignal_2D)
        im1_cbar = plt.colorbar(im)


        self.tel - self.atm
        self.wfs.cam.photonNoise = False
        nModes = self.M2C_KL.shape[1]
        stroke = 1 * 1e-9

        i_cube = np.zeros((self.wfs.nTheta, np.sum(self.wfs.validSignal), nModes))
        # ngs.nPhoton = 10000000000
        self.wfs.modulation = param['modulation']
        self.dm.coefs = 0
        self.ngs * self.tel * self.dm * self.wfs
        self.ref_cube = self.wfs.cam.cube[:, self.wfs.validSignal]

        for i in range(nModes):

            # t = np.sqrt(np.mean((ref_cube/np.sum(ref_cube))**2, axis=1))
            # ref_cube_c[:,i] = np.sqrt(np.mean((ref_cube/np.sum(ref_cube))**2, axis=1))/stroke
            self.dm.coefs = self.M2C_KL[:, i] * stroke
            self.tel * self.dm * self.wfs
            push = self.wfs.cam.cube[:, self.wfs.validSignal]
            push_signal = push / np.sum(push) - \
                          self.ref_cube / np.sum(self.ref_cube)

            # push_cube_c[:,i] = np.sqrt(np.mean((push/np.sum(push))**2, axis=1))/stroke
            if i == 0:
                push_cube_c = np.mean(push_signal, axis=1) / stroke

            self.dm.coefs = -self.M2C_KL[:, i] * stroke
            self.tel * self.dm * self.wfs
            pull = self.wfs.cam.cube[:, self.wfs.validSignal]
            pull_signal = pull / np.sum(pull) - \
                          self.ref_cube / np.sum(self.ref_cube)
            if i == 0:
                pull_cube_c = np.mean(pull_signal, axis=1) / stroke

            i_cube[:, :, i] = (0.5 * (push_signal - pull_signal) / stroke)

        self.weighting_cube = np.zeros((self.wfs.nTheta, nModes))
        for i in range(nModes):
            # weighting_cube[:,i] = (np.std(i_cube[:, :, i], axis=1))
            avg_val = np.mean(i_cube[:, :, i])
            # avg_val = 0
            self.weighting_cube[:, i] = ((np.mean((i_cube[:, :, i] - avg_val) ** 2, axis=1))) ** 2
            # weighting_cube[:,i] = -(weighting_cube[:,i] - np.mean(weighting_cube[:,i]))
            self.weighting_cube[:, i] = self.weighting_cube[:, i] / np.max(np.abs(self.weighting_cube[:, i]))

            # weighting_cube[:,i] = np.arctan(weighting_cube[:,i])

        # plt.figure(figsize=(30, 20))
        # im = plt.imshow(self.weighting_cube, cmap=cm.Greys)
        # im1_cbar = plt.colorbar(im)
        # plt.ylabel("Modulation Frame")
        # plt.xlabel("KL mode")


        self.framesRemPerFace = [11, 11, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                            3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1]
        binary_weight_cube = np.ones((self.wfs.nTheta, nModes))
        for f in range(len(self.framesRemPerFace)):
            idx = calcEquidistantFrameIndices(self.framesRemPerFace[f], self.wfs.nTheta)
            binary_weight_cube[idx, f] = 0

        # plt.figure(figsize=(30, 20))
        # im = plt.imshow(binary_weight_cube, cmap=cm.Greys)
        # im1_cbar = plt.colorbar(im)
        # plt.ylabel("Modulation Frame")
        # plt.xlabel("KL mode")


        # %% -----------------------     INTERACTION MATRIX   ----------------------------------

        self.tel - self.atm
        self.wfs.cam.photonNoise = False
        nModes = self.M2C_KL.shape[1]
        stroke = 1 * 1e-9

        def getInterationMatrixModulated(weights, mod=5):
            imat = np.zeros((np.sum(self.wfs.validSignal), nModes))
            self.wfs.modulation = mod
            self.dm.coefs = 0
            self.tel * self.dm * self.wfs
            ref_cube = self.wfs.cam.cube[:, self.wfs.validSignal]

            for m in range(nModes):
                self.dm.coefs = self.M2C_KL[:, m] * stroke
                self.tel * self.dm * self.wfs

                push = self.wfs.cam.cube[:, self.wfs.validSignal]
                push_signal = np.sum(push * weights[:, np.newaxis, m], axis=0) / np.sum(
                    push * weights[:, np.newaxis, m]) - \
                              np.sum(ref_cube * weights[:, np.newaxis, m], axis=0) / np.sum(
                    ref_cube * weights[:, np.newaxis, m])

                self.dm.coefs = -self.M2C_KL[:, m] * stroke
                self.tel * self.dm * self.wfs

                pull = self.wfs.cam.cube[:, self.wfs.validSignal]
                pull_signal = np.sum(pull * weights[:, np.newaxis, m], axis=0) / np.sum(
                    pull * weights[:, np.newaxis, m]) - \
                              np.sum(ref_cube * weights[:, np.newaxis, m], axis=0) / np.sum(
                    ref_cube * weights[:, np.newaxis, m])

                imat[:, m] = (0.5 * (push_signal - pull_signal) / stroke)
                # imat[:,m] /=np.std(imat[:,m] )
            return imat

        def getInterationMatrixUnmodulated():
            imat = np.zeros((np.sum(self.wfs.validSignal), nModes))
            self.wfs.modulation = 0
            self.dm.coefs = 0
            self.tel * self.dm * self.wfs
            ref_frame = self.wfs.cam.frame[self.wfs.validSignal]

            for m in range(nModes):
                self.dm.coefs = self.M2C_KL[:, m] * stroke
                self.tel * self.dm * self.wfs

                push = self.wfs.cam.frame[self.wfs.validSignal]
                push_signal = push / np.sum(push) - \
                              ref_frame / np.sum(ref_frame)

                self.dm.coefs = -self.M2C_KL[:, m] * stroke
                self.tel * self.dm * self.wfs

                pull = self.wfs.cam.frame[self.wfs.validSignal]
                pull_signal = pull / np.sum(pull) - \
                              ref_frame / np.sum(ref_frame)

                imat[:, m] = (0.5 * (push_signal - pull_signal) / stroke)
                # imat[:,m] /=np.std(imat[:,m] )

            return imat

        self.I_mat_modulated = getInterationMatrixModulated(np.ones((self.wfs.nTheta, nModes)), mod=param['modulation'])
        self.I_mat_unmodulated = getInterationMatrixUnmodulated()
        self.I_mat_weighted = getInterationMatrixModulated(self.weighting_cube, mod=param['modulation'])
        self.I_mat_binary = getInterationMatrixModulated(binary_weight_cube, mod=param['modulation'])

        self.I_mat_modulated_inv = inv(self.I_mat_modulated.T @ self.I_mat_modulated) @ self.I_mat_modulated.T
        self.I_mat_unmodulated_inv = inv(self.I_mat_unmodulated.T @ self.I_mat_unmodulated) @ self.I_mat_unmodulated.T
        self.I_mat_weighted_inv = inv(self.I_mat_weighted.T @ self.I_mat_weighted) @ self.I_mat_weighted.T
        self.I_mat_binary_inv = inv(self.I_mat_binary.T @ self.I_mat_binary) @ self.I_mat_binary.T

        # radian_in_1m = (2 * np.pi) / self.ngs.wavelength
        # plt.figure(figsize=(30, 10))
        # plt.plot(np.sqrt(self.wfs.nSubap ** 2) * np.sqrt(np.sum((I_mat_modulated) ** 2, axis=0)) / radian_in_1m, marker="x",
        #          label="Modulated")
        # plt.plot(np.sqrt(self.wfs.nSubap ** 2) * np.sqrt(
        #     np.sum((self.I_mat_unmodulated - np.mean(self.I_mat_modulated, axis=0)) ** 2, axis=0)) / radian_in_1m, marker="o",
        #          label="Unmodulated")
        # plt.plot(np.sqrt(self.wfs.nSubap ** 2) * np.sqrt(np.sum((I_mat_weighted) ** 2, axis=0)) / radian_in_1m, marker="o",
        #          label="TR WFS - Weighted")
        # plt.plot(np.sqrt(self.wfs.nSubap ** 2) * np.sqrt(np.sum((I_mat_binary) ** 2, axis=0)) / radian_in_1m, marker="o",
        #          label="TR WFS - Binary")
        # plt.xlabel("KL modes")
        # plt.ylabel("Sensitivity (std of interaction matrix)")
        # plt.title(f"Sensitivity with lambda={param['modulation']}, SubAp={self.wfs.nSubap}")
        # plt.legend()


        #
        # show the modes projected on the dm, cropped by the pupil and normalized by their maximum value
        # displayMap(self.tel.OPD, norma=True)
        # plt.title('Basis projected on the DM')

        #KL_dm = np.reshape(self.tel.OPD, [self.tel.resolution ** 2, self.tel.OPD.shape[2]])

        #covMat = (KL_dm.T @ KL_dm) / self.tel.resolution ** 2

        # plt.figure()
        # plt.imshow(covMat)
        # plt.title('Orthogonality')
        # plt.show()
        #
        # plt.figure()
        # plt.plot(np.round(np.std(np.squeeze(KL_dm[self.tel.pupilLogical, :]), axis=0), 5))
        # plt.title('KL mode normalization projected on the DM')
        # plt.show()

        # These are the calibration data used to close the loop
        if traj.enable_custom_frames:
            self.calib_CL = self.I_mat_weighted_inv
        else:
            if param['modulation'] == 0:
                self.calib_CL = self.I_mat_unmodulated_inv
            else:
                self.calib_CL = self.I_mat_modulated_inv

        self.M2C_CL = self.M2C_KL

        # combine telescope with atmosphere
        self.tel + self.atm

        # initialize DM commands
        self.dm.coefs = 0
        self.ngs * self.tel * self.dm * self.wfs


    def run_closed_loop(self, traj):

            #numRemFramesPerBase[0:100] = [11, 11, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #numRemFramesPerBase[0:100] = [9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 9, 3, 5, 0, 5, 3, 3, 5, 5, 3, 7, 5, 3, 3, 1, 0, 3, 3, 1, 0, 7, 3, 0, 1, 3, 3, 0, 0, 3, 1, 1, 1, 0, 1, 3, 0, 5, 0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 3, 0, 1, 0, 0, 1, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        # numRemFramesPerBase[0:100] = [41, 41, 35, 35, 39, 31, 31, 31, 31, 27, 27, 25, 27, 23, 23, 23, 19, 19, 19, 19,
        #                               19, 19, 9, 13, 13, 9, 17, 13, 13, 7, 7, 7, 7, 1, 1, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        #                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                               0]




        #for mag in tqdm(mags):
        #    for loop_gain in tqdm(loop_gains):
                #with silence():


        #with silence():
        self.setup(traj, traj.enable_custom_frames, traj.ao_calib_file)


        #print(f"Settings: Mag={mag}, Gain={loop_gain}")
        SR = np.zeros(self.param['nLoop'])
        SR_H = np.zeros(self.param['nLoop'])
        total = np.zeros(self.param['nLoop'])
        residual = np.zeros(self.param['nLoop'])
        OPD = np.zeros((self.param['nLoop'], self.tel.OPD.shape[0], self.tel.OPD.shape[1]))
        wfsSignal_record = np.zeros((self.param['nLoop'], ))
        if traj.enable_custom_frames:
            wfsSignal = np.zeros(self.wfs.pyramidSignalCube.shape)

        else:
            wfsSignal = np.zeros(self.wfs.pyramidSignal.shape)

        wfsSignal_record = np.zeros((self.param['nLoop'],)+wfsSignal.shape)
        #wfsSignal = np.arange(0, self.wfs.nSignal) * 0

        a_est = np.zeros((self.M2C_CL.shape[1]))
        wfs_to_use = None

        # loop parameters
        gainCL = self.param['gainCL']
        self.wfs.cam.photonNoise = self.param['photonNoise']


        if not self.minimum_data_only:
            traj.f_add_result("InteractionMatrix", self.calib_CL.M, comment="Interaction matrix")
            traj.f_add_result("M2C", self.M2C_CL, comment="Modes 2 Commands (M2C) for the DM")

        #for i in trange(self.param['nLoop'], position=0, leave=True):
        for i in trange(self.param['nLoop'], position=0, leave=True):
            #a = time.time()
            # update phase screens => overwrite tel.OPD and consequently tel.src.phase
            self.atm.update()
            # save phase variance
            total[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9
            #total[i] = np.sqrt(np.mean(self.tel.OPD[np.where(self.tel.pupil > 0)])**2)
            # save turbulent phase
            turbPhase = self.tel.src.phase

            # propagate to the WFS with the CL commands applied
            self.tel * self.dm * self.wfs

            # save the DM OPD shape
            dmOPD = self.tel.pupil * self.dm.OPD * 2 * np.pi / self.ngs.wavelength


            weights = self.weighting_cube


            if traj.enable_custom_frames:
                for j_mode in range(len(a_est)):
                    t_signal = self.wfs.cam.cube[:, self.wfs.validSignal]
                    t_t_signal = np.sum(t_signal * weights[:, np.newaxis, j_mode], axis=0) / np.sum(
                        t_signal * weights[:, np.newaxis, j_mode]) - \
                                 np.sum(self.ref_cube * weights[:, np.newaxis, j_mode], axis=0) / np.sum(
                        self.ref_cube * weights[:, np.newaxis, j_mode])
                    # weighted_signal = wfsSignal * self.weightsPerFramePerMode[j_mode, :, np.newaxis]
                    a_est[j_mode] = (self.calib_CL @ t_t_signal)[j_mode]
            else:
                a_est = self.calib_CL @ wfsSignal

            correction = np.matmul(self.M2C_CL, a_est)

            self.dm.coefs = self.dm.coefs - gainCL * correction
            # store the slopes after computing the commands => 2 frames delay
            if traj.enable_custom_frames:
                wfsSignal = self.wfs.pyramidSignalCube
            else:
                wfsSignal = self.wfs.pyramidSignal


            SR[i] = np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil == 1)]))
            phase_wave = self.tel.src.phase[np.where(self.tel.pupil == 1)] * self.tel.src.wavelength / 2 / np.pi
            rms = np.std(phase_wave)
            test = np.exp(-4*(np.pi**2)*(rms**2) / ( (0.790e-6)**2))
            wavelenth = 1.654e-6 #H band
            SR_H[i] = np.exp(-4*(np.pi**2)*(rms**2) / (wavelenth**2))
            residual[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9
            OPD[i, :, :] = self.tel.OPD
            if traj.enable_custom_frames:
                wfsSignal_record[i,:,:] = wfsSignal
            else:
                wfsSignal_record[i,:] = wfsSignal



        traj.f_add_result('SR', SR, comment="Strehl Ratio")
        traj.f_add_result('SR_H', SR_H, comment="Strehl Ratio in the H band")
        traj.f_add_result('Residual', residual, comment="Residual")
        traj.f_add_result('Turbulence', total, comment="Turbulence")
        if not self.minimum_data_only:
            traj.f_add_result('wfsSignal', wfsSignal_record, comment="wfsSignal during closed loop")
        #traj.f_add_result('OPD', OPD, comment="OPD")


def dict_to_trajectory(d, traj):
    for k,v in d.items():
        traj.f_add_parameter(k, v, comment=k)



if __name__ == "__main__":
    PC = PWFS_CL(minimum_data_only=True)

    param = initializeParameterFile()

    # Create an environment that handles running our simulation
    env = Environment(trajectory='run_loops', filename='/home/simonc/Documents/Programming/OOPAO_TRWFS/trwfs/sum/res/quick_test5.hdf5',
                      file_title='testing',
                      comment='Trying out the weighted frames approach',
                      large_overview_tables=True,
                      log_config='DEFAULT',
                      log_stdout=True,
                      overwrite_file=True)

    # Get the trajectory from the environment
    traj = env.trajectory

    dict_to_trajectory(param, traj)


    # Add  parameters
    #traj.f_add_parameter('param', param, comment='param')
    traj.f_add_parameter('enable_custom_frames', True, comment='enable_custom_frames')
    traj.f_add_parameter('ao_calib_file', "", comment='ao_calib_file')

    # Explore the parameters with a cartesian product

    traj.f_explore(cartesian_product({'magnitude': [10.0, 11.0, 12.0, 13.0, 14.0],
                                      'gainCL': [0.1, 0.2, 0.3, 0.4, 0.5],
                                      'nModes': [600],
                                      'nTheta_user_defined': [48,48, 48],
                                      'modulation': [0, 5, 5],
                                      'enable_custom_frames': [False, False, True],
                                      'ao_calib_file': ["no_mod.pickle","mod.pickle","weighted_mod.pickle"]},
                                     ('magnitude', 'gainCL', 'nModes', ('nTheta_user_defined', 'modulation', 'enable_custom_frames', 'ao_calib_file'))))

    # traj.f_explore(cartesian_product({'magnitude': [10, 11,12, 13, 14, 15],
    #                                   'gainCL': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    #                                   'enable_custom_frames': [True, False],
    #                                   'ao_calib_file': ["ao_calib_file_custom_48F_mag0_s13.pickle", "ao_calib_file_normal_48F_mag0_s13.pickle"]},
    #                                  ('magnitude', 'gainCL', ('enable_custom_frames', 'ao_calib_file'))))


    # traj.f_explore(cartesian_product({'magnitude': [14, 15],
    #                                   'gainCL': [0.1, 0.2],
    #                                   'enable_custom_frames': [True, False]}))

    # Run the simulation with all parameter combinations
    env.run(PC.run_closed_loop)

    # Let's check that all runs are completed!
    assert traj.f_is_completed()

    # Finally disable logging and close all log-files
    env.disable_logging()

    # PC.run_loops(param=param,
    #              mags=mags,
    #             loop_gains=loop_gains,
    #             enable_custom_frames=True,
    #             iterations=1000,
    #             filename="mag_and_gain_vs_residual_data_custom_fixed_0L50_testing4.pickle",
    #             ao_calib_file=None)
    # #
    # param['modulation'] = 0
    # PC.run_loops(param=param,
    #              mags=mags,
    #              loop_gains=loop_gains,
    #              enable_custom_frames=False,
    #              iterations=1000,
    #              filename="mag_and_gain_vs_residual_data_normal_fixed_0L50_testing4.pickle",
    #              ao_calib_file=None)
