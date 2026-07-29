import numpy as np

def walton_oblique_collision(theta_deg,d,mu,m,e):

    r=d/2
    theta=np.radians(theta_deg)

    # 初期速度
    v1=np.array([0.0,0.0])
    v2=np.array([1.0,0.0])
    dx = 0.01732
    dy = -0.01

    dist=np.sqrt(dx*dx+dy*dy)

    n=np.array([dx/dist,dy/dist])
    t=np.array([-n[1],n[0]])

    v=v2-v1

    vn=np.dot(v,n)
    vt=np.dot(v,t)

    I=(2/5)*m*r*r

    meff=m/2

    mt=1/(1/m+1/m+r*r/I+r*r/I)

    Jn=meff*(1+e)*vn

    Jt_stick=-mt*vt


    if abs(Jt_stick)<=mu*Jn:
        Jt=Jt_stick
        mode="stick"
    else:
        Jt=-mu*Jn*np.sign(vt)
        mode="slip"

    impulse=Jn*n+Jt*t

    v1_after=v1+impulse/m
    v2_after=v2-impulse/m

    w1= Jt*r/I
    w2= Jt*r/I

    return v1_after,v2_after,w1,w2,mode
