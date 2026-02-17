# Hit 4

El objetivo de este hit es escribir un programa que reciba un texto por parámetro, calcule su hash **MD5** utilizando la **GPU**, y muestre el resultado por consola. El enunciado aclara que se pueden usar implementaciones públicas para este fin.

MD5 es un algoritmo de hashing ampliamente conocido. Si bien hoy en día ya no se considera seguro para aplicaciones criptográficas reales, sigue siendo útil con fines educativos para entender cómo funcionan los algoritmos de hash y por qué su cálculo puede resultar costoso computacionalmente.

## Descripción de la solución

Para poder calcular el MD5, el mensaje de entrada debe cumplir un formato específico. Esto implica agregar un byte especial (`0x80`), completar con ceros hasta alcanzar un tamaño determinado y finalmente incluir la longitud original del mensaje.  
Esta preparación del mensaje se realiza en la **CPU**, ya que es una tarea simple y no requiere procesamiento intensivo.

Una vez que el mensaje está listo, se copia a la **GPU**, donde se ejecuta el cálculo del hash MD5. La GPU se utiliza porque este tipo de algoritmos involucra muchas operaciones matemáticas repetitivas, y está especialmente diseñada para ejecutar este tipo de cálculos de manera eficiente.

Cuando la GPU finaliza el cálculo, el resultado se copia nuevamente a la CPU y se imprime por consola en formato hexadecimal, que es la forma habitual de representar un hash MD5.

## Prueba realizada

Para verificar el funcionamiento del programa se utilizó el siguiente texto de entrada:

- **Entrada:** `hola mundo`

El resultado obtenido fue:

- **MD5:** `0ad066a5d29f3f2a2a1c7c17dd082a79`

Este valor coincide con el hash MD5 esperado para dicha cadena, lo que confirma que el programa funciona correctamente y que el cálculo se realizó de forma adecuada utilizando la GPU.