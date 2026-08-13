import numpy as np
from scipy.linalg import orthogonal_procrustes


def align_embeddings_procrustes(image_embeds, text_embeds):
    """
    Solves orthogonal Procrustes problem:  min_R ||X R - Y||_F
    where X = image_embeds, Y = text_embeds, and R^T R = I.

    Parameters
    ----------
    image_embeds : ndarray (n, d)  — source embeddings to rotate
    text_embeds  : ndarray (n, d)  — target embeddings

    Returns
    -------
    aligned_embeds : ndarray (n, d) — rotated image embeddings  X @ R
    R              : ndarray (d, d) — optimal orthogonal rotation matrix
    frobenius_before : float — ||X - Y||_F before alignment
    frobenius_after  : float — ||XR - Y||_F after alignment
    """
    frobenius_before = np.linalg.norm(image_embeds - text_embeds, 'fro')
    R, scale = orthogonal_procrustes(image_embeds, text_embeds)
    aligned_embeds = image_embeds @ R
    frobenius_after = np.linalg.norm(aligned_embeds - text_embeds, 'fro')

    return aligned_embeds, R, frobenius_before, frobenius_after
