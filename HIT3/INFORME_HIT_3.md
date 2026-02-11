## Hit 3

En este hit revisamos y documentamos las librerías CCCL y Thrust, compilamos y ejecutamos el ejemplo oficial que aparece en la documentación de Thrust y analizamos las diferencias prácticas entre programar en CUDA manualmente y usar Thrust/CCCL. Todas las pruebas se hicieron en Google Colab usando nvcc y la GPU que proporciona el entorno.

### CCCL ¿Qué es? ¿Por qué importa?

CCCL (CUDA Core Compute Libraries) es el proyecto que agrupa y mantiene varias librerías clave para la programación con CUDA en C++. Entre los componentes que reúne están Thrust, CUB y libcudacxx. El objetivo de CCCL es ofrecer un conjunto coherente y actualizado de herramientas y utilidades que los desarrolladores usan con frecuencia, de modo que no haya versiones dispersas o incompatibilidades entre paquetes.

Para el desarrollador práctico, CCCL significa poder acceder a algoritmos y utilidades optimizadas (por ejemplo ordenación, reducción, scan) sin tener que reescribirlas. El proyecto está activo y recibe actualizaciones periódicas: en los últimos meses se han publicado releases y mejoras, lo que indica que NVIDIA mantiene y prueba estas bibliotecas de forma continua.

### Thrust ¿Qué es? ¿Cómo se utiliza?

Thrust es una librería de C++ diseñada para operaciones de datos paralelos con una API parecida a la biblioteca estándar de C++. Proporciona contenedores y algoritmos, como sort, reduce, transform, generate y scan, que pueden ejecutarse en GPU sin necesidad de escribir kernels manuales.

**Modo de uso práctico:**

- Definir datos en `thrust::host_vector` (CPU) o `thrust::device_vector` (GPU).
- Llamar a un algoritmo de Thrust (por ejemplo `thrust::sort(d_vec.begin(), d_vec.end())`).
- Thrust se encarga de la copia al device si hace falta, ejecuta la operación usando implementaciones optimizadas y devuelve los resultados al host si corresponde.

Thrust suele ofrecer muy buen rendimiento en operaciones masivas (millones de elementos) y reduce significativamente el trabajo de implementación y depuración para tareas comunes.  
Thrust está incluido en el CUDA Toolkit, así que para compilar y ejecutar ejemplos en C++ no es necesario instalar nada extra si ya tenemos nvcc disponible.

### Ejecución del ejemplo oficial

Compilamos y ejecutamos el ejemplo oficial de Thrust en Colab. Para pruebas rápidas usamos `N = 1 << 20` (1 millón), mientras que la versión de la documentación original propone `32 << 20` para tests de rendimiento. El código que ejecutamos genera números aleatorios en host, copia los datos a device, ordena en GPU con `thrust::sort` y copia una porción de vuelta al host para verificación.

### ¿Se necesita instalar algo adicional?

No. Para C++ puro con nvcc no hace falta instalar Thrust, ya que viene incluido con el CUDA Toolkit. Si se trabaja desde Python y se quieren usar paquetes específicos que ofrece CCCL, existen paquetes pip, pero no son necesarios para compilar C++ con Thrust.

### Diferencias prácticas: programar CUDA “a mano” vs usar Thrust / CCCL

**Productividad:**  
Thrust permite implementar operaciones paralelas estándar con pocas líneas. Es ideal para prototipado y para soluciones que se ajusten a los algoritmos que la librería ofrece.

**Simplicidad y mantenimiento:**  
Con Thrust se evita la gestión manual de memoria y sincronizaciones, lo que implica menos código y menor riesgo de errores.

**Control y optimización fina:**  
Si necesitamos exprimir al máximo el hardware en un kernel muy específico (por ejemplo con uso intensivo de memoria compartida y optimización de accesos), programar en CUDA “a mano” sigue siendo necesario. En esos casos se puede complementar Thrust usando primitivas de bajo nivel como CUB, que forma parte de CCCL.

**Ecosistema:**  
CCCL reúne lo mejor de ambos mundos: abstracciones de alto nivel (Thrust) y primitivas de alto rendimiento (CUB), lo que facilita empezar con código legible y, si hace falta, optimizar las partes críticas.

