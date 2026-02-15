# Hit 5

El objetivo es modificar el programa anterior para que reciba dos parámetros, y encuentre un número tal que al calcular `MD5(nonce + cadena)` el resultado comience con el prefijo indicado.

Como se mencionó anteriormente, partimos del MD5 que ya habíamos implementado y lo reutilizamos dentro del minero. Lo que hicimos fue usar la GPU para probar muchísimos números al mismo tiempo. Cada número que se prueba se llama *nonce*. Para cada nonce se lo convierte a texto, se lo concatena con la cadena que recibimos por parámetro y se calcula su hash MD5. Luego se verifica si ese hash empieza con el prefijo pedido.

La GPU trabaja con muchos hilos al mismo tiempo; cuando alguno encuentra un resultado válido, el programa guarda ese número y el hash correspondiente, y termina la búsqueda. También se lleva la cuenta de cuántos **intentos** se hicieron en total.

## Resultados

- **Entrada:** prefijo `"00"`, cadena `"hola mundo"`  
- **Salida:** Hash resultante `004f794a54ef74ac507467c11af48c9d`, intentos `40960`