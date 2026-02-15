# Hit 7 —  HASH por fuerza bruta con CUDA (con límites)

**Objetivo**  
Buscamos probar diferentes longitudes de prefijo en el hash MD5 utilizando el minero en GPU, permitiendo definir el rango de búsqueda poor parámetro, para analizar cu+al es el prefijo más largo encontrado dentro de ese rango y cuánto tardó en encontrarlo.

**Qué se hizo**  
Se utilizó el minero GPU y se probaron los siguientes prefijos: “0”, “00”, “000”, “0000”, “00000”, “000000” y “0000000”.  A diferencia del Hit 6, en este caso el rango de búsqueda no está hardcodeado en el código, sino que se pasa como parámetro al ejecutar el script. En este caso lo hicimos de 0 - 850000.

**Resultados**

| Prefijo  | Nonce encontrado | Intentos | Tiempo (s) | Hash / s   |
|----------|-----------------:|---------:|-----------:|-----------:|
| 0        | 12355            | 40992    | 0.389      | 105340.3   |
| 00       | 21469            | 41216    | 0.268      | 153866.5   |
| 000      | 12485            | 57280    | 0.241      | 237881.5   |
| 0000     | 290797           | 100864   | 0.239      | 421475.6   |
| 00000    | 677029           | 539249   | 0.236      | 2286855.1  |
| 000000   | –                | 850001   | 0.260      | 3273937.9  |
| 0000000  | –                | 850001   | 0.260      | 3272273.3  |

---

## Respuestas a las preguntas del enunciado

**¿Cuál es el prefijo más largo que logró encontrar?**  
Con el rango `0..850000` el prefijo más largo que se encontró fue **`00000`**. El nonce que produjo ese hash en la corrida mostrada fue **677029**.

**¿Cuánto tardó?**  
La prueba con prefijo `00000` tardó **0.236 s** en la ejecución que pegaste.