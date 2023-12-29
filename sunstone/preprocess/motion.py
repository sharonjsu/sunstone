from suite2p.registration import register
import numpy as np

def motion_correct_3d(frames):
    """
    Motion correction for a 3D imaging stack using suite2p's package.

    Parameters
    ----------
    frames: imaging stack in the shape of Z x T x Y x X 
    """
    mc = []
    for i in range(0,np.shape(frames)[0]):
        refIm = register.compute_reference(frames[i])
        normIm = register.normalize_reference_image(refIm) # Idk if this is actualy a good idea or not lol
        refMask=  register.compute_reference_masks(normIm[0])
        reg = register.register_frames(refAndMasks=refMask,frames=frames[i])
        mc.append(reg[0])
    return np.array(mc)