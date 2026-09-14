import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforsm
import numpy as np

chain_split_path = 'data/chain_set_splits.json'
chain_set_path = 'data/chain_set.jsonl.txt'

# 1. Extraction des données
dataset = []
with open(chain_set_path, 'r', encoding='utf-8') as f_jsonl:
    for line in f_jsonl:
        if line.strip():
            dataset.append(json.loads(line))

with open(chain_split_path, 'r', encoding='utf-8') as f_json:
    datasplit = json.load(f_json)

# 2. Préparation des sets pour une recherche O(1), key error si train validation ou test n'existe pas 
train_keys = set(datasplit['train'])
val_keys = set(datasplit['validation'])
test_keys  = set(datasplit['test'])

# 3. Initialisation des conteneurs
train_set = []
val_set = []
test_set = []

# Initialisation du compteur en dehors de la boucle
orphelins_count = 0 

# 4. Distribution des données
for item in dataset:
    item_id = item.get('name') 
    
    if not item_id:
        continue

    if item_id in train_keys:
        train_set.append(item)
    elif item_id in val_keys:
        val_set.append(item)
    elif item_id in test_keys:
        test_set.append(item)
    else:
        # Incrémentation du compteur cumulatif
        orphelins_count += 1
        
print(f"Orphelins trouvés : {orphelins_count}")
print(f"Distribution -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

ATOM_ORDER = ['N', 'CA', 'C', 'O']
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}
UNK_IDX = 20


class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        coords = entry['coords']

        # 1. Empiler les 4 atomes -> [L, 4, 3]
        atom_arrays = [np.asarray(coords[atom], dtype=np.float32)
                       for atom in ATOM_ORDER]
        x = np.stack(atom_arrays, axis=1)

        # 2. Mask AVANT d'ecraser : un residu est valide si ses 12 valeurs le sont
        mask = ~np.isnan(x).any(axis=(1, 2))

        # 3. Ecraser les trous
        x = np.nan_to_num(x, nan=0.0)

        # 4. Sequence -> entiers
        seq = np.array([AA_TO_IDX.get(c, UNK_IDX) for c in entry['seq']],
                       dtype=np.int64)

        

        return (torch.from_numpy(x),
                torch.from_numpy(seq),
                torch.from_numpy(mask).float())

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
    print('Dimensions check ' , X.shape)
    return X


def collate_fn(batch):
    L_max = max([element[0].shape[0] for element in batch])
    B = len(batch)

    X = torch.zeros(B, L_max, 4, 3)
    S = torch.zeros(B, L_max, dtype=torch.long)
    mask = torch.zeros(B, L_max)

    for i, (x, seq, m) in enumerate(batch):
        L = x.shape[0]
        X[i, :L] = x
        S[i, :L] = seq
        mask[i, :L] = m

    X = Cbeta(X) # add virtual Cbeta --> [B,L,5,3]

    return X, S, mask



train_ds = ProteinDataset(train_set)
val_ds = ProteinDataset(val_set)
test_ds = ProteinDataset(test_set)

train_loader = DataLoader(train_ds, batch_size= 8 , shuffle=True, collate_fn= collate_fn)

val_loader = DataLoader(val_ds, batch_size= 8 , shuffle=False, collate_fn= collate_fn)

test_loader = DataLoader(test_ds, batch_size= 8 , shuffle=False, collate_fn= collate_fn)




#TEST DIM 
batch_X, batch_S, batch_mask = next(iter(train_loader))
testXCbeta = Cbeta(batch_X)
print(testXCbeta.shape)
    
    

    
