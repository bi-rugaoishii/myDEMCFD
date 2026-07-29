import numpy as np

def analyze_collision(x1,x2,v1_before,v2_before,v1_after,v2_after):

    dx = x1-x2
    n = dx/np.linalg.norm(dx)
    t = np.array([-n[1],n[0]])

    g_before = v2_before-v1_before
    g_after  = v2_after-v1_after

    gn_before = np.dot(g_before,n)
    gn_after  = np.dot(g_after,n)

    gt_before = np.dot(g_before,t)
    gt_after  = np.dot(g_after,t)

    e = -gn_after/gn_before

    print("restitution =",e)
    print("gt before =",gt_before)
    print("gt after  =",gt_after)
