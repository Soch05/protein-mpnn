import torch 


def Cbeta(X):
    
        CA = X[:,:,1,:] #[B,L,3]
        N  = X[:,:,0,:] #[B,L,3]
        C = X[:,:,2,:] #[B,L,3]
        O = X[:,:,3,:] #[B,L,3]

        b = CA -  N #[B,L,3]
        c = C -  CA #[B,L,3]
        a = torch.cross(b, c, dim = -1) # cross product in order to create third axis (90°)
        Cb = - 0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA #[B,L,3]

        Cb = Cb.unsqueeze(2)#[B,L,3] --> #[B,L,1,3]
        X = torch.concat([X, Cb], dim = 2) #[B,L,5,3]
        return X



def pairwise_distances(A, B, eps = 1e-6):
        d = A.unsqueeze(1) - B.unsqueeze(2)  # [B,1,L,3] - [B,L,1,3] = [B,L,L,3]

        mat = torch.sqrt((d**2).sum(-1) + eps )  #[B,L,L]
        return mat
        

