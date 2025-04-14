# Sprawozdanie 3 - Normy macierzowe i współczynniki uwarunkowania

# Wojciech Smolarczyk, Wiktoria Zalińska

## Normy macierzowe

Norma macierzowa to uogólnienie pojęcia normy wektorowej na macierze. Jest to funkcja przypisująca każdej macierzy (kwadratowej lub prostokątnej) nieujemną liczbę rzeczywistą, która mierzy jej "rozmiar" lub "wzrost" w pewnym sensie. Normy macierzowe są używane w analizie numerycznej do badania stabilności algorytmów, błędów zaokrągleń i uwarunkowania problemów.

### Główne właściwości norm macierzowych:

- Nieujemność: ∥M∥≥0
- Jednorodność ∥αM∥ = |α| \* ∥M∥ dla skalara α
-

## Wskaźnik (współczynnik) uwarunkowania

Wskaźnik uwarunkowania określa, w jakim stopniu błąd reprezentacji numerycznej danych wejściowych danego problemu wpływa na błąd wyniku. Wskaźnik uwarunkowania definiuje się jako maksymalny stosunek błędu względnego rozwiązania do błędu względnego danych. Problem o niskim wskaźniku uwarunkowania nazywamy dobrze uwarunkowanym, zaś problemy o wysokim wskaźniku uwarunkowania – źle uwarunkowanymi. Zagadnienia o zbyt dużym wskaźniku uwarunkowania nie nadają się do numerycznego rozwiązywania, ponieważ już sam błąd wynikający z numerycznej reprezentacji liczb wprowadza nieproporcjonalnie duży błąd w odpowiedzi.

Wskaźnik uwarunkowania jest cechą problemu i jest niezależny od numerycznych właściwości konkretnych algorytmów. W odróżnieniu od błędu zaokrągleń wprowadzonego przez algorytm, wskaźnik uwarunkowania stanowi informację o błędzie przeniesionym z danych.

### Poszczególne normy i wskaźniki (współczynniki) uwarunkowania

- Norma macierzowa ∥A∥2

  ![alt text](image-3.png)

- Współczynnik uwarunkowania macierzy ∥A∥2

  ![alt text](image-4.png)

- Norma macierzowa ∥A∥∞ (maksymalna suma wartości bezwzględnych z wierszy)

  ![alt text](image-2.png)

- Współczynnik uwarunkowania macierzy ∥M∥∞

  ![alt text](image-5.png)

## Implementacja

- Odwracanie macierzy

```python
def macierz_odwrotna(A):

    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Macierz musi być kwadratowa")

    rozszerzona = np.hstack((A, np.identity(n)))

    for i in range(n):

        if rozszerzona[i, i] == 0:
            for j in range(i + 1, n):
                if rozszerzona[j, i] != 0:
                    rozszerzona[[i, j]] = rozszerzona[[j, i]]
                    break
            else:
                raise ValueError(
                    "Macierz jest osobliwa - nie istnieje macierz odwrotna"
                )

        dzielnik = rozszerzona[i, i]
        rozszerzona[i] = rozszerzona[i] / dzielnik

        for j in range(n):
            if j != i and rozszerzona[j, i] != 0:
                czynnik = rozszerzona[j, i]
                rozszerzona[j] = rozszerzona[j] - czynnik * rozszerzona[i]

    A_odwrotna = rozszerzona[:, n:]

    return A_odwrotna
```

- Norma macierzowa ∥A∥2

```python
    import numpy as np
    from numpy.linalg import eig, inv

    def wartosci_wlasne(M):
        return np.linalg.eig(M).eigenvalues

    def norma_macierzowaA2(M):
        return max(np.abs(wartosci_wlasne(M)))
```

- Współczynnik uwarunkowania macierzy ∥A∥2

```python
def wspl_warunkowyA2(M):
    if np.linalg.det(M) != 0:
        return norma_macierzowaA2(macierz_odwrotna(M)) * norma_macierzowaA2(M)
    else:
        return "Macierz jest osobliwa (wyznacznik = 0)"
```

- Norma macierzowa ∥A∥∞ (maksymalna suma wartości bezwzględnych z wierszy)

```python
def norma_macierzowaAinf(M):
    return max(np.sum(np.abs((M)), axis=1))
```

- Współczynnik uwarunkowania macierzy ∥M∥∞

```python
def wspl_warunkowyAinf(M):
    if np.linalg.det(M) != 0:
        return norma_macierzowaAinf(macierz_odwrotna(M)) * norma_macierzowaAinf(M)
    else:
        return "Macierz jest osobliwa (wyznacznik = 0)"
```

### Wyniki

```python
norma_macierzowaA2([[1, 2], [3, 4]])
```

np.float64(5.372281323269014)

```python
wspl_warunkowyA2(np.array([[1000, 999], [999, 998]]))
```

np.float64(3992006.000094148)

```python
norma_macierzowaAinf([[1, 2], [3, 4]])
```

norma_macierzowaAinf([[1, 2], [3, 4]])

```python
wspl_warunkowyAinf(np.array([[1000, 999], [999, 998]]))
```

np.float64(3996001.000094493)
