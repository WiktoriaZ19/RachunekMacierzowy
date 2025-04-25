# Sprawozdanie 4 - SVD

# Wojciech Smolarczyk, Wiktoria Zalińska

```python
A = np.array([[2, 2, 1, 3], [7, 8, 2, 8], [5, 0, 8, 7], [3, 5, 6, 3], [6, 9, 8, 4]])
n, m = A.shape


```

### Krok 7

Proszę obliczyć i wypisać/narysować macierz ATA(mxm)

```python
ATA = np.dot(A.T, A)

print(ATA)

```

### Krok 8

Proszę (używając stosownej biblioteki) policzyć wartości λi i wektory własne Vi macierzy ATA.

```python
lambdas, V = np.linalg.eig(ATA)
print("\nStep 8: Eigenvalues (λ_i) of A^T A:")
print(lambdas)
print("\nEigenvectors (V_i) of A^T A (columns):")
print(V)

```

Eigenvalues (λ_i) of A^T A:
[430.73196147 59.91931837 6.26331361 20.08540655]

Eigenvectors (V_i) of A^T A (columns):
[[0.61815934  0.29989507  0.57592274  0.44300674]
 [ 0.44206838 -0.10849764  0.24128459 -0.85707967]
 [ 0.56014362 -0.65084557 -0.45039358  0.24450931]
 [ 0.32968729  0.68897842 -0.63815387 -0.09682286]]

### Krok 9

```python
print("\n9. Macierz V^T:\n", V.T)


S = np.zeros((m, m))
for i in range(m):
    S[i, i] = np.sqrt(lambdas[i]) if lambdas[i] > 1e-10 else 0.0
print("\nStep 9: Diagonal S (manual):\n", S)
```

Macierz V^T:\
[[0.61815934  0.44206838  0.56014362  0.32968729]\
 [ 0.29989507 -0.10849764 -0.65084557  0.68897842]\
 [ 0.57592274  0.24128459 -0.45039358 -0.63815387]\
 [ 0.44300674 -0.85707967  0.24450931 -0.09682286]]

Diagonal S :\
[[20.75408301  0.          0.          0.        ]\
 [ 0.          7.74075696  0.          0.        ]\
 [ 0.          0.          2.50266131  0.        ]\
 [ 0.          0.          0.          4.48167452]]

### Krok 10 i 11

```python
S_inv = np.zeros_like(S)
for i in range(m):
    if S[i, i] > 1e-10:
        S_inv[i, i] = 1.0 / S[i, i]

U = A @ V @ S_inv
print("\nStep 11: U = [U_1 U_2 ... U_m]:\n", U)
```

U = [U_1 U_2 ... U_m]:\
 [[0.17681634  0.2323913  -0.29186551 -0.19503985]\
 [ 0.5599596   0.70295717 -0.01769403 -0.90170635]\
 [ 0.47603918  0.14411507 -2.07403694  0.77947387]\
 [ 0.40545003 -0.19131735 -0.67233703 -0.3971263 ]\
 [ 0.64987066 -0.21031008 -0.21124164 -0.77803363]]

### Krok 12

```python
reconstructed_A = U @ S @ V.T
print(
    "\nReconstructed A :\n",
    reconstructed_A,
    "\n\n",
    A,
)


### Wiktoria tutaj daj swój wynik dekompozycji
```

Reconstructed A (should match original A):\
 [[0.04947759 0.81811934 0.7883919  0.60408366]\
 [0.19956419 0.21971753 0.07282224 0.49019101]\
 [0.45073204 0.10379424 0.72024798 0.78083499]\
 [0.54020983 0.2094221  0.39908469 0.02106216]\
 [0.12990275 0.15128147 0.15548295 0.33681011]]

[[0.04947759 0.81811934 0.7883919  0.60408366]\
 [0.19956419 0.21971753 0.07282224 0.49019101]\
 [0.45073204 0.10379424 0.72024798 0.78083499]\
 [0.54020983 0.2094221  0.39908469 0.02106216]\
 [0.12990275 0.15128147 0.15548295 0.33681011]]

### Krok 13

Obliczmy wykorzystując wzory:

rankA = dimR(A)

dimR(A) + dimN (A) = m

```python
rank_A = np.sum(lambdas > 1e-10)
nullity_A = m - rank_A
print("\n Dimensions:")
print(f"dim(R(A)) = rank(A) = {rank_A}")
print(matrix_rank(A))

print(f"dim(N(A)) = nullity(A) = {nullity_A}")
```

Dimensions:\
dim(R(A)) = rank(A) = 4\
4\
dim(N(A)) = nullity(A) = 0
