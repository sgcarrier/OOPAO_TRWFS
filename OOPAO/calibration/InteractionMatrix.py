# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:26:22 2020

@author: cheritie
"""
import numpy as np
import time
from .CalibrationVault import CalibrationVault


def InteractionMatrix(ngs, atm, tel, dm, wfs, M2C, stroke, phaseOffset=0, nMeasurements=50, noise='off', invert=True,
                      print_time=True):
    if wfs.tag == 'pyramid' and wfs.gpu_available:
        nMeasurements = 1
        print('Pyramid with GPU detected => using single mode measurement to increase speed.')
    #    disabled noise functionality from WFS
    if noise == 'off':
        wfs.cam.photonNoise = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')

        # separate tel from ATM
    tel.isPaired = False
    ngs * tel
    try:
        nModes = M2C.shape[1]
    except:
        nModes = 1
    intMat = np.zeros([wfs.nSignal, nModes])
    nCycle = int(np.ceil(nModes / nMeasurements))
    nExtra = int(nModes % nMeasurements)
    if nMeasurements > nModes:
        nMeasurements = nModes

    if np.ndim(phaseOffset) == 2:
        if nMeasurements != 1:
            phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

    for i in range(nCycle):
        if nModes > 1:
            if i == nCycle - 1:
                if nExtra != 0:
                    intMatCommands = np.squeeze(M2C[:, -nExtra:])
                    try:
                        phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, intMatCommands.shape[-1]))
                    except:
                        phaseBuffer = phaseOffset
                else:
                    intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
            else:
                intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
        else:
            intMatCommands = np.squeeze(M2C)

        a = time.time()
        #        push
        dm.coefs = intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        sp = wfs.signal

        #       pull
        dm.coefs = -intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        sm = wfs.signal
        if i == nCycle - 1:
            if nExtra != 0:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    if nExtra == 1:
                        intMat[:, -nExtra] = np.squeeze(0.5 * (sp - sm) / stroke)
                    else:
                        intMat[:, -nExtra:] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    intMat[:, -nMeasurements:] = np.squeeze(0.5 * (sp - sm) / stroke)


        else:
            if nMeasurements == 1:
                intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                intMat[:, i * nMeasurements:((i + 1) * nMeasurements)] = np.squeeze(0.5 * (sp - sm) / stroke)
        intMat = np.squeeze(intMat)

        if print_time:
            print(str((i + 1) * nMeasurements) + '/' + str(nModes))
            b = time.time()
            print('Time elapsed: ' + str(b - a) + ' s')

    out = CalibrationVault(intMat, invert=invert)

    return out


def interactionMatrix_withTR(ngs, atm, tel, dm, wfs, M2C, stroke, phaseOffset=0, nMeasurements=50, noise='off', invert=True,
                      print_time=True, custom_frames=True, custom_remove_frames=[]):
    #    disabled noise functionality from WFS
    if noise == 'off':
        wfs.cam.photonNoise = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')

        # separate tel from ATM
    tel.isPaired = False
    ngs * tel
    try:
        nModes = M2C.shape[1]
    except:
        nModes = 1
    intMat = np.zeros([wfs.nSignal, nModes])
    nCycle = int(np.ceil(nModes / nMeasurements))
    nExtra = int(nModes % nMeasurements)
    if nMeasurements > nModes:
        nMeasurements = nModes

    if np.ndim(phaseOffset) == 2:
        if nMeasurements != 1:
            phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

    for i in range(nCycle):
        if nModes > 1:
            if i == nCycle - 1:
                if nExtra != 0:
                    intMatCommands = np.squeeze(M2C[:, -nExtra:])
                    try:
                        phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, intMatCommands.shape[-1]))
                    except:
                        phaseBuffer = phaseOffset
                else:
                    intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
            else:
                intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
        else:
            intMatCommands = np.squeeze(M2C)

        a = time.time()
        #        push
        dm.coefs = intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        if (wfs.modulation == 0):
            sp = wfs.signal
        else:
            sp = wfs.signalCube[0,:]

        #custom_remove_frames = np.zeros((nMeasurements))
        # custom_remove_frames[0:2] = 9
        # custom_remove_frames[2:4] = 9
        # custom_remove_frames[4:10] = 7
        # custom_remove_frames[10:16] = 5
        # custom_remove_frames[16:21] = 3
        # custom_remove_frames[21:27] = 1
        #custom_remove_frames = np.array([9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 9, 3, 5, 0, 5, 3, 3, 5, 5, 3, 7, 5, 3, 3, 1, 0, 3, 3, 1, 0, 7, 3, 0, 1, 3, 3, 0, 0, 3, 1, 1, 1, 0, 1, 3, 0, 5, 0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 3, 0, 1, 0, 0, 1, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        #custom_remove_frames = np.array([11, 11, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #custom_remove_frames = np.array([41, 41, 35, 35, 39, 31, 31, 31, 31, 27, 27, 25, 27, 23, 23, 23, 19, 19, 19, 19, 19, 19, 9, 13, 13, 9, 17, 13, 13, 7, 7, 7, 7, 1, 1, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #remFrames = np.unique(custom_remove_frames)
        remFrames = wfs.remFrames

        # if custom_frames:
        #     if i == 0:
        #         for j in range(nMeasurements):
        #             d_idx = np.argwhere(remFrames == custom_remove_frames[j])[0][0]
        #             sp[:, j] = wfs.signalCube[d_idx,:,j]

        if custom_frames:
            for j in range(nMeasurements):
                idx = (i * nMeasurements) + j
                if idx < 100:
                    d_idx = np.argwhere(remFrames == custom_remove_frames[idx])[0][0]
                    sp[:] = wfs.signalCube[d_idx,:]

        #       pull
        dm.coefs = -intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        if (wfs.modulation == 0):
            sm = wfs.signal
        else:
            sm = wfs.signalCube[0,:]

        if custom_frames:
            for j in range(nMeasurements):
                idx = (i * nMeasurements) + j
                if idx < 100:
                    d_idx = np.argwhere(remFrames == custom_remove_frames[idx])[0][0]
                    sm[:] = wfs.signalCube[d_idx,:]

        if i == nCycle - 1:
            if nExtra != 0:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    if nExtra == 1:
                        intMat[:, -nExtra] = np.squeeze(0.5 * (sp - sm) / stroke)
                    else:
                        intMat[:, -nExtra:] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    intMat[:, -nMeasurements:] = np.squeeze(0.5 * (sp - sm) / stroke)


        else:
            if nMeasurements == 1:
                intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                intMat[:, i * nMeasurements:((i + 1) * nMeasurements)] = np.squeeze(0.5 * (sp - sm) / stroke)
        intMat = np.squeeze(intMat)

        if print_time:
            print(str((i + 1) * nMeasurements) + '/' + str(nModes))
            b = time.time()
            print('Time elapsed: ' + str(b - a) + ' s')

        out = CalibrationVault(intMat, invert=invert)

    return out


def interactionMatrix_withTR_withintmat(ngs, atm, tel, dm, wfs, M2C, stroke, phaseOffset=0, nMeasurements=50, noise='off', invert=True,
                      print_time=True, custom_frames=True, custom_remove_frames=[]):
    #    disabled noise functionality from WFS
    if noise == 'off':
        wfs.cam.photonNoise = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')

        # separate tel from ATM
    tel.isPaired = False
    ngs * tel
    try:
        nModes = M2C.shape[1]
    except:
        nModes = 1
    intMat = np.zeros([wfs.nSignal, nModes])
    nCycle = int(np.ceil(nModes / nMeasurements))
    nExtra = int(nModes % nMeasurements)
    if nMeasurements > nModes:
        nMeasurements = nModes

    if np.ndim(phaseOffset) == 2:
        if nMeasurements != 1:
            phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

    for i in range(nCycle):
        if nModes > 1:
            if i == nCycle - 1:
                if nExtra != 0:
                    intMatCommands = np.squeeze(M2C[:, -nExtra:])
                    try:
                        phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, intMatCommands.shape[-1]))
                    except:
                        phaseBuffer = phaseOffset
                else:
                    intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
            else:
                intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
        else:
            intMatCommands = np.squeeze(M2C)

        a = time.time()
        #        push
        dm.coefs = intMatCommands * stroke
        tel * dm * wfs
        tel.src.phase += phaseBuffer
        tel * wfs
        if (wfs.modulation == 0):
            sp = wfs.signal
        else:
            sp = wfs.signalCube[0,:]

        #custom_remove_frames = np.zeros((nMeasurements))
        # custom_remove_frames[0:2] = 9
        # custom_remove_frames[2:4] = 9
        # custom_remove_frames[4:10] = 7
        # custom_remove_frames[10:16] = 5
        # custom_remove_frames[16:21] = 3
        # custom_remove_frames[21:27] = 1
        #custom_remove_frames = np.array([9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 9, 3, 5, 0, 5, 3, 3, 5, 5, 3, 7, 5, 3, 3, 1, 0, 3, 3, 1, 0, 7, 3, 0, 1, 3, 3, 0, 0, 3, 1, 1, 1, 0, 1, 3, 0, 5, 0, 1, 1, 1, 0, 0, 3, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 3, 0, 1, 0, 0, 1, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        #custom_remove_frames = np.array([11, 11, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #custom_remove_frames = np.array([41, 41, 35, 35, 39, 31, 31, 31, 31, 27, 27, 25, 27, 23, 23, 23, 19, 19, 19, 19, 19, 19, 9, 13, 13, 9, 17, 13, 13, 7, 7, 7, 7, 1, 1, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #remFrames = np.unique(custom_remove_frames)
        remFrames = wfs.remFrames

        # if custom_frames:
        #     if i == 0:
        #         for j in range(nMeasurements):
        #             d_idx = np.argwhere(remFrames == custom_remove_frames[j])[0][0]
        #             sp[:, j] = wfs.signalCube[d_idx,:,j]

        if custom_frames:
            for j in range(nMeasurements):
                idx = (i * nMeasurements) + j
                if idx < 100:
                    #print(f"i={i}, idx={idx}")
                    d_idx = np.argwhere(remFrames == custom_remove_frames[idx])[0][0]
                    sp[:] = wfs.signalCube[d_idx,:]

        #       pull
        dm.coefs = -intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        if (wfs.modulation == 0):
            sm = wfs.signal
        else:
            sm = wfs.signalCube[0,:]

        if custom_frames:
            for j in range(nMeasurements):
                idx = (i * nMeasurements) + j
                if idx < 100:
                    #print(f"i={i}, idx={idx}")
                    d_idx = np.argwhere(remFrames == custom_remove_frames[idx])[0][0]
                    sm[:] = wfs.signalCube[d_idx,:]

        if i == nCycle - 1:
            if nExtra != 0:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    if nExtra == 1:
                        intMat[:, -nExtra] = np.squeeze(0.5 * (sp - sm) / stroke)
                    else:
                        intMat[:, -nExtra:] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    intMat[:, -nMeasurements:] = np.squeeze(0.5 * (sp - sm) / stroke)


        else:
            if nMeasurements == 1:
                intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                intMat[:, i * nMeasurements:((i + 1) * nMeasurements)] = np.squeeze(0.5 * (sp - sm) / stroke)
        intMat = np.squeeze(intMat)

        if print_time:
            print(str((i + 1) * nMeasurements) + '/' + str(nModes))
            b = time.time()
            print('Time elapsed: ' + str(b - a) + ' s')

        out = CalibrationVault(intMat, invert=invert)

    return out, intMat
def interactionMatrix_withTR_no_negative(ngs, atm, tel, dm, wfs, M2C, stroke, phaseOffset=0, nMeasurements=50, noise='off', invert=True,
                      print_time=True, custom_frames=True, custom_remove_frames=[]):
    #    disabled noise functionality from WFS
    if noise == 'off':
        wfs.cam.photonNoise = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')

        # separate tel from ATM
    tel.isPaired = False
    ngs * tel
    try:
        nModes = M2C.shape[1]
    except:
        nModes = 1
    intMat = np.zeros([wfs.nSignal, nModes])
    nCycle = int(np.ceil(nModes / nMeasurements))
    nExtra = int(nModes % nMeasurements)
    if nMeasurements > nModes:
        nMeasurements = nModes

    if np.ndim(phaseOffset) == 2:
        if nMeasurements != 1:
            phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

    for i in range(nCycle):
        if nModes > 1:
            if i == nCycle - 1:
                if nExtra != 0:
                    intMatCommands = np.squeeze(M2C[:, -nExtra:])
                    try:
                        phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, intMatCommands.shape[-1]))
                    except:
                        phaseBuffer = phaseOffset
                else:
                    intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
            else:
                intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
        else:
            intMatCommands = np.squeeze(M2C)

        a = time.time()
        #        push
        dm.coefs = intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        if (wfs.modulation == 0):
            sp = wfs.signal
        else:
            sp = wfs.signalCube[0,:]

        remFrames = wfs.remFrames


        if custom_frames:
            for j in range(nMeasurements):
                idx = (i * nMeasurements) + j
                if idx < 100:
                    d_idx = np.argwhere(remFrames == custom_remove_frames[idx])[0][0]
                    sp[:] = wfs.signalCube[d_idx,:]

        #       pull
        dm.coefs = -intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        if (wfs.modulation == 0):
            sm = wfs.signal
        else:
            sm = wfs.signalCube[0,:]

        if custom_frames:
            for j in range(nMeasurements):
                idx = (i * nMeasurements) + j
                if idx < 100:
                    d_idx = np.argwhere(remFrames == custom_remove_frames[idx])[0][0]
                    sm[:] = wfs.signalCube[d_idx,:]

        if i == nCycle - 1:
            if nExtra != 0:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(sp / stroke)
                else:
                    if nExtra == 1:
                        intMat[:, -nExtra] = np.squeeze(sp / stroke)
                    else:
                        intMat[:, -nExtra:] = np.squeeze(sp / stroke)
            else:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(sp / stroke)
                else:
                    intMat[:, -nMeasurements:] = np.squeeze(sp / stroke)


        else:
            if nMeasurements == 1:
                intMat[:, i] = np.squeeze(sp / stroke)
            else:
                intMat[:, i * nMeasurements:((i + 1) * nMeasurements)] = np.squeeze(sp / stroke)
        intMat = np.squeeze(intMat)

        if print_time:
            print(str((i + 1) * nMeasurements) + '/' + str(nModes))
            b = time.time()
            print('Time elapsed: ' + str(b - a) + ' s')

        out = CalibrationVault(intMat, invert=invert)

    return out

def interactionMatrix_weightedFrames(ngs, atm, tel, dm, wfs, M2C, stroke, phaseOffset=0, nMeasurements=50, noise='off', invert=True,
                      print_time=True, custom_frames=True, frame_weights=[]):
    #    disabled noise functionality from WFS
    if noise == 'off':
        wfs.cam.photonNoise = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')

        # separate tel from ATM
    tel.isPaired = False
    ngs * tel
    try:
        nModes = M2C.shape[1]
    except:
        nModes = 1
    intMat = np.zeros([wfs.nSignal, nModes])
    nCycle = int(np.ceil(nModes / nMeasurements))
    nExtra = int(nModes % nMeasurements)
    if nMeasurements > nModes:
        nMeasurements = nModes

    if np.ndim(phaseOffset) == 2:
        if nMeasurements != 1:
            phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

    for i in range(nCycle):
        if nModes > 1:
            if i == nCycle - 1:
                if nExtra != 0:
                    intMatCommands = np.squeeze(M2C[:, -nExtra:])
                    try:
                        phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, intMatCommands.shape[-1]))
                    except:
                        phaseBuffer = phaseOffset
                else:
                    intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
            else:
                intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
        else:
            intMatCommands = np.squeeze(M2C)

        a = time.time()
        #        push
        dm.coefs = intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        if (wfs.modulation == 0):
            sp = wfs.signal
        else:
            sp = wfs.signalCube[-1,:]

        if custom_frames:
            for j in range(nMeasurements):
                idx = (i * nMeasurements) + j
                sp[:] = wfs.signalCube[idx, :]

        #       pull
        dm.coefs = -intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel * wfs
        if (wfs.modulation == 0):
            sm = wfs.signal
        else:
            sm = wfs.signalCube[-1,:]

        if custom_frames:
            for j in range(nMeasurements):
                idx = (i * nMeasurements) + j
                sm[:] = wfs.signalCube[idx, :]


        if i == nCycle - 1:
            if nExtra != 0:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    if nExtra == 1:
                        intMat[:, -nExtra] = np.squeeze(0.5 * (sp - sm) / stroke)
                    else:
                        intMat[:, -nExtra:] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    intMat[:, -nMeasurements:] = np.squeeze(0.5 * (sp - sm) / stroke)


        else:
            if nMeasurements == 1:
                intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                intMat[:, i * nMeasurements:((i + 1) * nMeasurements)] = np.squeeze(0.5 * (sp - sm) / stroke)
        intMat = np.squeeze(intMat)

        if print_time:
            print(str((i + 1) * nMeasurements) + '/' + str(nModes))
            b = time.time()
            print('Time elapsed: ' + str(b - a) + ' s')

    out = CalibrationVault(intMat, invert=invert)

    return out

def interactionMatrixFromPhaseScreen(ngs,atm,tel,wfs,phasScreens,stroke,phaseOffset=0,nMeasurements=50,noise='off',invert=True,print_time=True):
    
    #    disabled noise functionality from WFS
    if noise =='off':  
        wfs.cam.photonNoise  = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')    
    
    tel.isPaired = False
    ngs*tel
    try: 
        nModes = phasScreens.shape[2]
    except:
        nModes = 1
    intMat = np.zeros([wfs.nSignal,nModes])
    nCycle = int(np.ceil(nModes/nMeasurements))
    nExtra = int(nModes%nMeasurements)
    if nMeasurements>nModes:
        nMeasurements = nModes
    
    if np.ndim(phaseOffset)==2:
        if nMeasurements !=1:      
            phaseBuffer = np.tile(phaseOffset[...,None],(1,1,nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

        
    for i in range(nCycle):  
        if nModes>1:
            if i==nCycle-1:
                if nExtra != 0:
                    modes_in  = np.squeeze(phasScreens[:,:,-nExtra:])                
                    try:               
                        phaseBuffer     = np.tile(phaseOffset[...,None],(1,1,modes_in.shape[-1]))
                    except:
                        phaseBuffer     = phaseOffset
                else:
                    modes_in = np.squeeze(phasScreens[:,:,i*nMeasurements:((i+1)*nMeasurements)])
    
            else:
                modes_in = np.squeeze(phasScreens[:,:,i*nMeasurements:((i+1)*nMeasurements)])
        else:
            modes_in = np.squeeze(phasScreens)

        a= time.time()
#        push
        tel.OPD = modes_in*stroke
        tel.src.phase+=phaseBuffer
        tel*wfs
        sp = wfs.signal
#       pull
        tel.OPD=-modes_in*stroke
        tel.src.phase+=phaseBuffer
        tel*wfs
        sm = wfs.signal        
        if i==nCycle-1:
            if nExtra !=0:
                if nMeasurements==1:
                    intMat[:,i] = np.squeeze(0.5*(sp-sm)/stroke)                
                else:
                    if nExtra ==1:
                        intMat[:,-nExtra] =  np.squeeze(0.5*(sp-sm)/stroke)
                    else:
                        intMat[:,-nExtra:] =  np.squeeze(0.5*(sp-sm)/stroke)
            else:
                 if nMeasurements==1:
                    intMat[:,i] = np.squeeze(0.5*(sp-sm)/stroke)      
                 else:
                    intMat[:,-nMeasurements:] =  np.squeeze(0.5*(sp-sm)/stroke)


        else:
            if nMeasurements==1:
                intMat[:,i] = np.squeeze(0.5*(sp-sm)/stroke)                
            else:
                intMat[:,i*nMeasurements:((i+1)*nMeasurements)] = np.squeeze(0.5*(sp-sm)/stroke)
        intMat = np.squeeze(intMat)
        if print_time:
            print(str((i+1)*nMeasurements)+'/'+str(nModes))
            b=time.time()
            print('Time elapsed: '+str(b-a)+' s' )
    
        out=calibrationVault(intMat,invert=invert)
  
    return out



# def interactionMatrixOnePass(ngs,atm,tel,dm,wfs,M2C,stroke,phaseOffset=0,nMeasurements=50,noise='off'):
# #    disabled noise functionality from WFS
#     if noise =='off':  
#         wfs.cam.photonNoise  = 0
#         wfs.cam.readoutNoise = 0
#     else:
#         print('Warning: Keeping the noise configuration for the WFS')    
    
#     tel-atm
#     ngs*tel
#     nModes = M2C.shape[1]
#     intMat = np.zeros([wfs.nSignal,nModes])
#     nCycle = int(np.ceil(nModes/nMeasurements))
#     nExtra = int(nModes%nMeasurements)
#     if nMeasurements>nModes:
#         nMeasurements = nModes
    
#     if np.ndim(phaseOffset)==2:
#         if nMeasurements !=1:      
#             phaseBuffer = np.tile(phaseOffset[...,None],(1,1,nMeasurements))
#         else:
#             phaseBuffer = phaseOffset
#     else:
#         phaseBuffer = phaseOffset

        
#     for i in range(nCycle):  
#         if i==nCycle-1:
#             if nExtra != 0:
#                 intMatCommands  = np.squeeze(M2C[:,-nExtra:])                
#                 try:               
#                     phaseBuffer     = np.tile(phaseOffset[...,None],(1,1,intMatCommands.shape[-1]))
#                 except:
#                     phaseBuffer     = phaseOffset
#             else:
#                 intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])

#         else:
#             intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])
            
#         a= time.time()
# #        push
#         dm.coefs = intMatCommands*stroke
#         tel*dm
#         tel.src.phase+=phaseBuffer
#         tel*wfs
#         sp = wfs.signal        
#         if i==nCycle-1:
#             if nExtra !=0:
#                 if nMeasurements==1:
#                     intMat[:,i] = np.squeeze(0.5*(sp)/stroke)                
#                 else:
#                     intMat[:,-nExtra:] =  np.squeeze(0.5*(sp)/stroke)
#             else:
#                  if nMeasurements==1:
#                     intMat[:,i] = np.squeeze(0.5*(sp)/stroke)      
#                  else:
#                     intMat[:,-nMeasurements:] =  np.squeeze(0.5*(sp)/stroke)


#         else:
#             if nMeasurements==1:
#                 intMat[:,i] = np.squeeze(0.5*(sp)/stroke)                
#             else:
#                 intMat[:,i*nMeasurements:((i+1)*nMeasurements)] = np.squeeze(0.5*(sp)/stroke)

#         print(str((i+1)*nMeasurements)+'/'+str(nModes))
#         b=time.time()
#         print('Time elapsed: '+str(b-a)+' s' )

#     out=calibrationVault(intMat)
#     return out      
        
        
