# Hit 6 — Longitudes de prefijo en CUDA (mediciones)

**Objetivo**  
Probar diferentes longitudes de prefijo en el hash MD5 para ver cuál es el prefijo más largo que logramos encontrar, cuánto tardó y cuál es la relación entre la longitud del prefijo y el tiempo requerido.

**Qué se hizo**  
Se usó el minero GPU y se probaron los prefijos: `"0"`, `"00"`, `"000"`, `"0000"`, `"00000"`.  
En todas las pruebas se fijó el mismo rango de búsqueda: `0 .. 200000`. Para cada prefijo se registró: el nonce encontrado (si lo hubo), el hash resultante (inicio), la cantidad de intentos que reportó el minero y el tiempo total de la ejecución.

**Resultados (ejecución reciente)**

| Prefijo  | Nonce encontrado | Intentos | Tiempo (s) | Hashes/s (aprox.) | Hash (inicio) |
|----------|------------------:|---------:|-----------:|------------------:|---------------:|
| `0`      | 355              | 40960    | 0.497      | 82,414            | 0c7656ba...    |
| `00`     | 16715            | 40960    | 0.375      |109,227            | 005508cc...    |
| `000`    | 12485            | 40960    | 0.339      |120,826            | 00072600...    |
| `0000`   | 27522            | 43264    | 0.34       |127,247            | 0000ce90...    |
| `00000`  | – (no)           | 200001   | 0.333      |600,604            | —              |
| `000000` | – (no)           | 200001   | 0.358      |559,044            | —              |
| `0000000`| – (no)           | 200001   | 0.322      |622,084            | —              |

---

## Respuestas a las preguntas del enunciado

**¿Cuál es el prefijo más largo que logró encontrar?**  
Con el rango `0..200000` el prefijo más largo que se encontró fue **`0000`** (4 ceros). El nonce que produjo ese hash en la corrida mostrada fue **27522**.

**¿Cuánto tardó?**  
La prueba con prefijo `0000` tardó **0.34 s** en la ejecución que pegaste.

**¿Cuál es la relación entre la longitud del prefijo a buscar y el tiempo requerido?**

La relación es directa: cuanto más largo es el prefijo que se busca, más tiempo tarda el programa en encontrar un resultado. Esto pasa porque cada vez que se agrega un carácter al prefijo, hay muchos menos hashes que cumplen la condición, entonces el minero tiene que probar muchos más números antes de tener éxito.

En las pruebas se ve claramente que con prefijos cortos (como “0”, “00” o “000”) el resultado aparece rápido dentro del rango elegido. En cambio, a partir de prefijos más largos, como “00000”, ya no se encontró ningún valor en el mismo rango, lo que indica que haría falta probar muchos más números o esperar mucho más tiempo.