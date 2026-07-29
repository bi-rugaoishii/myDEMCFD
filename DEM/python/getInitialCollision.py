import numpy as np

def collision_initial_positions(d, theta_deg, L=0.2):
    """
    2粒子斜め衝突の初期位置を計算

    Parameters
    ----------
    d : float
        粒径
    theta_deg : float
        衝突角度 (deg)
    L : float
        初期間隔

    Returns
    -------
    (x1,y1),(x2,y2)
        粒子1と粒子2の初期位置
    """

    r = d/2
    theta = np.radians(theta_deg)

    x1, y1 = 0.0, 0.0
    x2 = -L
    y2 = 2*r*np.sin(theta)

    return (x1,y1),(x2,y2)
