Informe provisional HIT 2

Se realizó un programa mínimo en CUDA que lanza un kernel en la GPU. Cada hilo calcula su identificador global y lo imprime por pantalla. Además, se implementó una versión equivalente en CPU que realiza la misma numeración, con el objetivo de poder comparar los resultados. Ambos programas fueron ejecutados y comprobados en Google Colab utilizando una GPU NVIDIA Tesla T4

El programa CUDA lanza un kernel con una configuración de 2 bloques de 4 hilos cada uno. Cada hilo calcula su identificador global utilizando la expresión global_id = blockIdx.x * blockDim.x + threadIdx.x. 
 Para el control de errores se utilizan las funciones  cudaGetLastError() y cudaDeviceSynchronize() o que permite detectar fallos tanto en el lanzamiento del kernel como durante su ejecución.

Problemas encontrados y cómo se solucionaron

El único problema detectado ocurrió al ejecutar el código CUDA en Google Colab. En ese caso, aparecía el mensaje de error: “the provided PTX was compiled with an unsupported toolchain”. 
 Este inconveniente se resolvió compilando el programa con la opción -arch=sm_75, correspondiente a la arquitectura de la GPU Tesla T4, lo que nos permite que nvcc genere código binario compatible con el driver del entorno.

Salidas obtenidas:
Hola desde el hilo global 0 (block 0, thread 0)
...
Hola desde el hilo global 7 (block 1, thread 3)

 La versión CPU produce la misma salida, lo que confirma que la ejecución del kernel en la GPU fue correcta.

 En conclusión, el entorno CUDA en Google Colab quedó validado, el kernel se ejecutó correctamente y los resultados coinciden  salida coincide con la versión en CPU. Con esto se cumple lo solicitado en este hit.