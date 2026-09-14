NOTE
coef to find Cbeta based on (Yang et al. 2020), reused by Ingraham.



14/09/2026 (T4)

Cbeta: cross product + literature coefs, concat -> [B,L,5,3]
distance_matrix: unsqueeze + broadcasting -> [B,L,L]

eps under the sqrt: diagonal is exactly 0, d/dx sqrt(x) -> inf at 0.
Without it, NaN on all weights at first backward. With 1e-6 the grad
peaks at 500 - finite, and clip_grad_norm_ handles it in T20.

Cbeta lives in ProteinFeatures, not collate_fn.

Missing residues are set to (0,0,0) by nan_to_num and stay at their
position, they are not grouped at the end. So [:L] includes them and
wrecks the averages (I got 6.11 instead of 3.80). Select with the bool
mask, and for a consecutive pair require both ends valid:
m[:-1] & m[1:]. Same logic in T5: mask the 2D pairs BEFORE the topk,
otherwise the (0,0,0) residues look like perfect neighbours.

Why virtual Cbeta? The real one is in the side chain -> leaks the
residue identity into the features, and that is exactly what the model
has to predict. Gly has no Cbeta: real ones would leave a hole that
says "Gly here". Virtual comes from N/CA/C only -> orientation without
identity. Verified: 1.53 A, std 0.054.

Checks: splits 18024/608/1120 - Ca-Ca 3.807 - Ca-Cb 1.527 (std 0.054)




