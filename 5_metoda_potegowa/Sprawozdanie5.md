# Sprawozdanie 5 - Metoda potęgowa

# Wojciech Smolarczyk, Wiktoria Zalińska

## 1. Implementacja metody potęgowej

Celem zadania było zaimplementowanie **metody potęgowej** dla macierzy `3x3` w celu wyznaczenia największej wartości własnej oraz odpowiadającego jej wektora własnego. W algorytmie uwzględniono warunki dotyczące losowania wektora początkowego oraz kryterium błędu początkowego.

---

Metoda potęgowa to prosta i skuteczna iteracyjna metoda numeryczna służąca do wyznaczenia największej (co do wartości bezwzględnej) wartości własnej macierzy kwadratowej `A`, oraz odpowiadającego jej wektora własnego `z`.

--- 

### Kroki algorytmu:

1. Znalezienie wektora startowego:

- Generacja losowego wektora z z przedziału (0,1).

- Obliczenie:

    - **w=Az**,

    - lambda = max(|w|),

    - error = ||Az - lambda * z||

- Jeżeli error < 10^-8, to powtarzane są powyższe kroki (aby nie startować z wektora, który spełnia warunki stopu)

2. Iteracja:

- obliczenie **w=Az**

- lambda = max(|w|)

- nowe z jako z = w / lambda

- error = ||Az - lambda * z||

Jeżeli error < 10^-8, kończymy iterację, otrzymując największą wartość własną i odpowiadający jej wektor własny.

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

