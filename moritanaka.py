import numpy as np

#rotation functions
def qij_x(theta):
    """
    rotation tensor about x-axis by some theta
    input: theta (angle in radians)
    output: qij (3x3 rotation tensor)
    """
    qij = np.array([[1,0,0],
                   [0,np.cos(theta),np.sin(theta)],
                   [0,-np.sin(theta),np.cos(theta)]])
    return qij

def qij_y(theta):
    """
    rotation tensor abo|ut y-axis by some theta
    input: theta (angle in radians)
    output: qij (3x3 rotation tensor)
    """
    qij = np.array([[np.cos(theta), 0, -np.sin(theta)],
                   [0,1,0],
                   [np.sin(theta),0,np.cos(theta)]])
    return qij

def qij_z(theta):
    """
    rotation tensor about z-axis by some theta
    input: theta (angle in radians)
    output: qij (3x3 rotation tensor)
    """
    qij = np.array([[np.cos(theta),np.sin(theta),0],
                   [-np.sin(theta),np.cos(theta),0],
                   [0,0,1]])
    return qij

def R_sigma(q):
    """
    rotation matrix for engineering notation given some rotation tensor, qij
    note: uses convention for sigma = [s11, s22, s33, s23, s13, s12]
    input: q (3x3 rotation tensor)
    output: R_s (6x6 rotation matrix)
    """
    R_s = np.array([[q[0,0]**2,q[0,1]**2,q[0,2]**2,2.*q[0,1]*q[0,2],2.*q[0,0]*q[0,2], 2.*q[0,0]*q[0,1]],
                       [q[1,0]**2,q[1,1]**2,q[1,2]**2,2.*q[1,1]*q[1,2],2.*q[1,0]*q[1,2], 2.*q[1,0]*q[1,1]],
                       [q[2,0]**2,q[2,1]**2,q[2,2]**2,2.*q[2,1]*q[2,2],2.*q[2,0]*q[2,2], 2.*q[2,0]*q[2,1]],
                       [q[1,0]*q[2,0], q[1,1]*q[2,1], q[1,2]*q[2,2],q[1,2]*q[2,1]+q[1,1]*q[2,2], q[1,2]*q[2,0]+q[1,0]*q[2,2], q[1,1]*q[2,0]+q[1,0]*q[2,1]],
                       [q[0,0]*q[2,0], q[0,1]*q[2,1], q[0,2]*q[2,2],q[0,2]*q[2,1]+q[0,1]*q[2,2], q[0,2]*q[2,0]+q[0,0]*q[2,2], q[0,1]*q[2,0]+q[0,0]*q[2,1]],
                       [q[0,0]*q[1,0], q[0,1]*q[1,1], q[0,2]*q[1,2],q[0,2]*q[1,1]+q[0,1]*q[1,2], q[0,2]*q[1,0]+q[0,0]*q[1,2], q[0,1]*q[1,0]+q[0,0]*q[1,1]]])
    return R_s

# for printing purposes, recast 4th-order tensor as 6x6
def tens2eng(tens):
    """
    returns 4th-order tensor in engineering notation
    """
    A = np.zeros((6,6))
    #mapping
    code = {0:(0,0),1:(1,1),2:(2,2),3:(1,2),4:(0,2),5:(0,1)}
    for i in range(6):
        for j in range(6):
            A[i,j] = tens[code[i][0],code[i][1],code[j][0],code[j][1]]
    return A

#Eshelby tensor calculations
def eshelby(nu_m, s):
    I1 = (2.*s/(s**2.-1.)**1.5)*(s*(s**2.-1.)**.5-np.arccosh(s))
    Q = 3./(8.*(1-nu_m))
    R = (1.-2.*nu_m)/(8.*(1.-nu_m))
    T = Q*(4.-3.*I1)/(3*(s**2.-1.))
    I3 = 4.-2.*I1
    S = np.zeros((6,6))
    S[0,0] = Q + R*I1 + 0.75*T
    S[1,1] = S[0,0]
    S[2,2] = 4./3.*Q + R*I3 + 2.*s**2.*T
    S[0,1] = Q/3. - R*I1 + 4.*T/3.
    S[1,0] = S[0,1]
    S[0,2] = -R*I1 - s**2*T
    S[1,2] = S[0,2]
    S[2,0] = -R*I3 - T
    S[2,1] = S[2,0]
    S[5,5] = Q/3 + R*I1 + T/4
    S[3,3] = 2*R - I1*R/2 - (1+s**2)*T/2
    S[4,4] = S[3,3]
    #rotate S to x-direction
    Q1 = qij_y(np.pi/2)
    R1 = R_sigma(Q1)
    S1 = np.dot(R1,np.dot(S,R1.T))
    return S1

#fiber/matrix stiffness matrices
def isotropicC(E,nu):
    C = np.zeros((6,6))
    C[0:3,0:3] = nu*np.ones((3,3))
    C = C + (1.-2.*nu)*np.eye(6)
    C[3:,3:] = C[3:,3:]/2.
    C = E/((1.+nu)*(1.-2.*nu))*C
    return C

#orthotropic stiffness matrices
def orthotropicC(E1,E2,E3,vxy,vyz,vxz,Gxy,Gyz,Gxz):
    S = np.array([[1./E1, -vxy/E1, -vxz/E1, 0, 0, 0],
              [-vxy/E1, 1./E2, -vyz/E2, 0, 0, 0],
              [-vxz/E1, -vyz/E2, 1./E3, 0, 0, 0],
              [0, 0, 0, 1./(2*Gyz), 0, 0],
              [0, 0, 0, 0, 1./(2*Gxz), 0],
              [0, 0, 0, 0, 0, 1./(2*Gxy)]])
    C = np.linalg.inv(S)
    return C

#Eshelby strain concentration
def strainConcentration(vf,S,Cf,Cm):
    A_e = np.linalg.inv(np.eye(6)+np.dot(np.dot(S1,np.linalg.inv(Cm)),Cf-Cm))
    A_mt = np.dot(A_e,np.linalg.inv(vf*A_e+(1-vf)*np.eye(6)))
    return A_mt

#orientation averaging
def ornavg(C,A4,a2):
    B1 = C[0,0] + C[2,2] - 2*C[1,2] -4*C[4,4]
    B2 = C[1,2] - C[0,1]
    B3 = C[4,4] - 0.5*(C[0,0]-C[0,1])
    B4 = C[0,1]
    B5 = 0.5*(C[0,0]-C[0,1])
    #map between tensor and engineering notation
    code = {0:(0,0),1:(1,1),2:(2,2),3:(1,2),4:(0,2),5:(0,1)}
    C_avg = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            C_avg[i,j] = B1*A4[i,j] + B2*(
            a2[code[i][0],code[i][1]]*(code[j][0]==code[j][1]) +
                a2[code[j][0],code[j][1]]*(code[i][0]==code[i][1])) + B3*(
            a2[code[i][0],code[j][0]]*(code[i][1]==code[j][1]) +
            a2[code[i][0],code[j][1]]*(code[i][1]==code[j][0]) +
            a2[code[i][1],code[j][0]]*(code[i][0]==code[j][1]) +
            a2[code[i][1],code[j][1]]*(code[i][0]==code[j][0])) + B4*(
            (code[i][0]==code[i][1])*(code[j][0]==code[j][1])) + B5*(
            (code[i][0]==code[j][0])*(code[i][1]==code[j][1]) +
            (code[i][0]==code[j][1])*(code[i][1]==code[j][0]))
    return C_avg

def linearClosure(a2):
    a4 = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    a4[i,j,k,l] = -1./24.*((i==j)*(k==l) + (i==k)*(j==l) + (i==l)*(j==k)) + 1./6.*(
                     a2[i,j]*(k==l) + a2[i,k]*(j==l) +a2[i,l]*(j==k) + a2[k,l]*(i==j) +
                     a2[j,l]*(i==k) + a2[j,k]*(i==l))
    return a4

def quadraticClosure(a2):
    a4 = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    a4[i,j,k,l] = a2[i,j]*a2[k,l]
    return a4

def hybridClosure(a2):
    f = -0.5
    for i in range(3):
        for j in range(3):
            f = f + 1.5*a2[i,j]*a2[j,i]
    a4_l = linearClosure(a2)
    a4_q = quadraticClosure(a2)
    a4 = (1-f)*a4_l + f*a4_q
    return a4

#TODO: fitted orthotropic closure
#TODO: main

#uni-directional stiffness
C_uni = Cm + vf*np.dot(Cf-Cm,A_mt)
#rotate to z
C_uni = np.dot(R1,np.dot(C_uni,R1.T))
print np.round(C1,decimals=2)
