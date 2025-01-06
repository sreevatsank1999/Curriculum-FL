import numpy as np 
from numpy import linalg as LA

def Eq_Basis(A,B):
    AB=np.arccos(A.T@B)
    A_E=np.zeros((A.shape[0],A.shape[1]))
    B_E=np.zeros((B.shape[0],B.shape[1]))
    for i in range(AB.shape[0]):
        ind = np.unravel_index(np.argmin(AB, axis=None), AB.shape)
        AB[ind[0],:]=AB[:,ind[1]]=0
        A_E[:,i]=A[:,ind[0]]
        B_E[:,i]=B[:,ind[1]]
    return  A_E,B_E

def Gau_Lin_ker(A,B,kernel=None):

    Sig=np.zeros((A.shape[1],B.shape[1]))
    Ker_G=np.zeros((A.shape[1],B.shape[1]))
    Ker_L=np.zeros((A.shape[1],B.shape[1]))

    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            Sig [i,j]= LA.norm(A[:,i]-B[:,j], 2)**2
    sigma=Sig.mean()

    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            Ker_G[i,j]=np.exp(-LA.norm(A[:,i]-B[:,j], 1)**0.9/(2*sigma))
            Ker_L[i,j]=np.dot(A[:,i],B[:,j])
   
    if kernel is "Gaussian":
        return Ker_G
    elif kernel is "Linear":
        return Ker_L
    
def angle():
    sim_mat = np.zeros([num, num])
    sim_mat_tr = np.zeros([num, num])

    for i in range(num):
        for j in range(num):
            F, G = Eq_Basis (U_per_class[i],U_per_class[j])
            F_in_G = np.clip(F.T@G, a_min = -1, a_max = +1)

            Angle = np.arccos(np.abs(F_in_G))
            sim_mat[i,j] =  (180/np.pi)*np.min(Angle) 
            sim_mat_tr[i,j] =(180/np.pi)*np.trace(Angle)
    
    return sim_mat, sim_mat_tr

def hierarchical_clustering(A, thresh=1.5, linkage='maximum'):
    '''
    Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
    rows and columns replacing the minimum elements. It is working on adjacency matrix. 
    
    :param: A (adjacency matrix), thresh (stopping threshold)
    :type: A (np.array), thresh (int)
    
    :return: clusters
    '''
    label_assg = {i: i for i in range(A.shape[0])}
    
    B = copy.deepcopy(A)
    step = 0
    while A.shape[0] > 1:
        np.fill_diagonal(A,-np.NINF)
        #print(f'step {step} \n {A}')
        step+=1
        ind=np.unravel_index(np.argmin(A, axis=None), A.shape)
        
        if A[ind[0],ind[1]]>thresh:
            print('Breaking HC')
            #print(f'A {B}')
            break
        else:
            np.fill_diagonal(A,0)
            if linkage == 'maximum':
                Z=np.maximum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'minimum':
                Z=np.minimum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'average':
                Z= (A[:,ind[0]] + A[:,ind[1]])/2
            
            A[:,ind[0]]=Z
            A[:,ind[1]]=Z
            A[ind[0],:]=Z
            A[ind[1],:]=Z
            A = np.delete(A, (ind[1]), axis=0)
            A = np.delete(A, (ind[1]), axis=1)
            
            B = copy.deepcopy(A)
            if type(label_assg[ind[0]]) == list: 
                label_assg[ind[0]].append(label_assg[ind[1]])
            else: 
                label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]

            label_assg.pop(ind[1], None)

            temp = []
            for k,v in label_assg.items():
                if k > ind[1]: 
                    kk = k-1
                    vv = v
                else: 
                    kk = k 
                    vv = v
                temp.append((kk,vv))

            label_assg = dict(temp)

    clusters = []
    for k in label_assg.keys():
        if type(label_assg[k]) == list:
            clusters.append(list(flatten(label_assg[k])))
        elif type(label_assg[k]) == int: 
            clusters.append([label_assg[k]])
            
    #print(label_assg)
            
    return clusters