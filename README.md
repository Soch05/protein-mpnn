ProteinMPNN from scratch

A PyTorch reimplementation of ProteinMPNN, built to learn the architecture rather than to beat it. No pretrained weights.

Status (September 2026): early. Working on the data pipeline. The model is not implemented yet.

What it does

Inverse folding: given the 3D coordinates of a protein backbone, predict a sequence that would fold into it. The model represents the protein as a graph over residues and passes messages between neighbours, instead of treating it as a sequence.

Based on Ingraham et al. (2019) and Dauparas et al. (2022).


Progress
 x Repo and environment setup
 x Parsing structures and extracting backbone coordinates
 Building the k-nearest-neighbour graph
 Node and edge features
 Encoder
 Decoder
 Training
 Sequence recovery on a test set

Notes

Using the CATH-based splits from Ingraham et al., where train and test sets share no CATH topology. A random split by chain would put homologous folds on both sides, and the score would mostly measure memorisation.

Sequence recovery is the standard metric but an imperfect one: several sequences can fold into the same backbone, so a good prediction can still count as wrong.
