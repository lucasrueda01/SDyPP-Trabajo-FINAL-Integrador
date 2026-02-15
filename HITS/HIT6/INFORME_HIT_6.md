# Hit 6 — Longitudes de prefijo en CUDA (mediciones)

**Objetivo**  
Probar diferentes longitudes de prefijo en el hash MD5 para ver cuál es el prefijo más largo que logramos encontrar, cuánto tardó y cuál es la relación entre la longitud del prefijo y el tiempo requerido.

**Qué se hizo**  
 Usamos el minero GPU que ya teníamos y ejecutamos pruebas con los siguientes prefijos: “0”, “00”, “000”, “0000”, “00000”, “000000”, “0000000”. Fijamos el mismo rango de búsqueda en todas las pruebas: de 0 a 200000. Para cada prefijo ejecutamos el minero y tomamos: el nonce encontrado (si lo hubo), el hash resultante, la cantidad de intentos y el tiempo de ejecución.

**Resultados**

| Prefijo   | Nonce encontrado | Intentos | Tiempo (s) | Hashes/s (aprox.) |
| --------- | ---------------: | -------: | ---------: | ----------------: |
| `0`       |             7172 |    40960 |      0.402 |          101873.8 |
| `00`      |            16715 |    40960 |      0.279 |          146592.4 |
| `000`     |             7590 |    40960 |      0.248 |          165006.5 |
| `0000`    |            27522 |    40960 |      0.235 |          174080.7 |
| `00000`   |                – |   200001 |      0.242 |          825035.2 |
| `000000`  |                – |   200001 |      0.226 |          882574.4 |
| `0000000` |                – |   200001 |      0.227 |          879299.4 |

---

## Respuestas a las preguntas del enunciado

**¿Cuál es el prefijo más largo que logró encontrar?**  
Con el rango `0..200000` el prefijo más largo que se encontró fue **`0000`**. El nonce que produjo ese hash en la corrida mostrada fue **27522**.

**¿Cuánto tardó?**  
La prueba con prefijo `0000` tardó **0.235 s** en la ejecución que pegaste.

**¿Cuál es la relación entre la longitud del prefijo a buscar y el tiempo requerido?**

La relación es directa: cuanto más largo es el prefijo que se busca, más tiempo tarda el programa en encontrar un resultado. Esto pasa porque cada vez que se agrega un carácter al prefijo, hay muchos menos hashes que cumplen la condición, entonces el minero tiene que probar muchos más números antes de tener éxito.

En las pruebas se ve claramente que con prefijos cortos (como “0”, “00” o “000”) el resultado aparece rápido dentro del rango elegido. En cambio, a partir de prefijos más largos, como “00000”, ya no se encontró ningún valor en el mismo rango, lo que indica que haría falta probar muchos más números o esperar mucho más tiempo.