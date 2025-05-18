# Sprawozdanie 5 - Metoda potęgowa

# Wojciech Smolarczyk, Wiktoria Zalińska

## 1. Implementacja metody potęgowej

Celem zadania było zaimplementowanie **metody potęgowej** dla macierzy `3x3` w celu wyznaczenia największej wartości własnej oraz odpowiadającego jej wektora własnego. W algorytmie uwzględniono warunki dotyczące losowania wektora początkowego oraz kryterium błędu początkowego.


Metoda potęgowa to prosta i skuteczna iteracyjna metoda numeryczna służąca do wyznaczenia największej (co do wartości bezwzględnej) wartości własnej macierzy kwadratowej `A`, oraz odpowiadającego jej wektora własnego `z`.


### Kroki algorytmu:

1. Znalezienie wektora startowego:

- Generacja losowego wektora z z przedziału (0,1).

- Obliczenie:

    - **w=Az**,

    - $\lambda = max(|w|)$,

    - $error = ||Az - \lambda * z||$

- Jeżeli $error < 10^{-8}$, to powtarzane są powyższe kroki (aby nie startować z wektora, który spełnia warunki stopu)

2. Iteracja:

- obliczenie **w=Az**

- $\lambda = max(|w|)$

- nowe z jako $z = w / \lambda$

- $error = ||Az - \lambda * z||$

Jeżeli $error < 10^{-8}$, kończymy iterację, otrzymując największą wartość własną i odpowiadający jej wektor własny.

### Kod algorytmu

```python
def power_method(A, p=2, epsilon=1e-8, max_iter=1000):

    n = A.shape[0]

    while True:
        # Generate a random 3x1 vector
        z = np.random.uniform(0.0, 1.0, n)

        # Calculate the product of A and z
        w = np.dot(A, z)

        # Calculate error
        lambda_ = np.max(np.abs(w))
        error = np.linalg.norm(w - lambda_ * z, ord=p)

        if error >= epsilon: # z is ok -> don't start froma vector that meets the stop condition
            break


    for i in range(max_iter):
        w = np.dot(A, z)
        lambda_ = np.max(np.abs(w))
        z = w / lambda_

        error = np.linalg.norm(A @ z - lambda_ * z, ord=p)


        if error < epsilon:
            break

    return lambda_, z, i+i, error
```

### Wynik działania programu
- Matrix A:

[[5 2 8]

 [3 4 2]

 [8 6 2]]

- Największa wartość własna: 13.62590694760338

- Odpowiadający wektor własny: [1.         0.50923604 0.95092936]

- Liczba iteracji: 40

- Końcowy błąd: 6.578052208635384e-09


- Iloczyn A z:

array([13.62590694,  6.93880287, 12.95727495])

- Iloczyn lambda z:

array([13.62590695,  6.93880287, 12.95727495])



##  Rozkład SVD macierzy 3x3 z wykorzystaniem metody potęgowej

W celu wyznaczenia wektorów i wartości własnych macierzy $AA^T$, wartości i wektory własne obliczamy za pomocą metody potęgowej - aby znaleźć kolejną wartość, dokonujemy deflacji macierzy zgodnie ze wzorem:

![Wzór deflacji](defl.png)

i dla niej stosujemy kolejny raz metodę potęgową.

W wyniku otrzymujemy:

- macierz U - zawierającą wektory własne

- macierz D - zawierającą wartości osobliwe

Macierz V obliczamy jako:
$V=A^TUinv(D)$
(przy czym macierz D jest diagonalna taka że $inv(D)_{ii} = 1/D_{ii}$)

W wyniku czego uzyskujemy zrekonstruowaną macierz A:

$A_{rec} = U  D V^T$


### Kod:

```python

AAT = np.dot(A, A.T)
B = AAT.copy()

eigenvalues = []
eigenvectors = []

# Calculate eigenvalues and eigenvectors of AAT
for i in range(3):
    lambda_, eigenvector_, _, _ = power_method(B)
    eigenvalues.append(lambda_)
    eigenvectors.append(eigenvector_)

    # deflate
    B = B - lambda_ * np.outer(eigenvector_, eigenvector_) / np.dot(eigenvector_, eigenvector_)

# Normalize eigenvectors
for i in range(len(eigenvectors)):
    eigenvectors[i] = eigenvectors[i] / np.linalg.norm(eigenvectors[i])

# Matrix U — wektory własne
U = np.column_stack(eigenvectors)

# Matrix D - pierwiastki z wartości własnych
D_vals = np.sqrt(np.array(eigenvalues))
D = np.diag(D_vals)

# Inverse D 
D_inv = np.diag([1 / d if d > 1e-12 else 0 for d in D_vals])

V = A.T @ U @ D_inv

A_reconstructed = U @ D @ V.T
```

### Wyniki:
Macierz A:

[[5 2 8]

 [3 4 2]

 [8 6 2]]

Macierz U:

[[ 0.61918806  0.78051443 -0.0860428 ]

 [ 0.3685361  -0.19209505  0.90954969]

 [ 0.69338827 -0.59489218 -0.40659071]]

Macierz D:

[[13.86943642  0.          0.        ]

 [ 0.          5.61901815  0.        ]

 [ 0.          0.          1.437139  ]]

Macierz V:

[[ 0.70288759 -0.25500014 -0.66402106]

 [ 0.49553925 -0.49416185  0.7143143 ]

 [ 0.51028412  0.83113115  0.22097765]]

Odtworzona macierz A:

[[5. 2. 8.]

 [3. 4. 2.]
 
 [8. 6. 2.]]