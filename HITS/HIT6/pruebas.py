

#Este codigo no va a funcionar porque ejecuta el ejecutar minero de otra carpeta, 
#para ello habria que poner bien la ruta, pero lo dejo asi pq lo tengo ejecutado en el Google Colab

import time
import csv
from minero_gpu import ejecutar_minero

# --- CONFIG ---
cadena = "hola mundo"
prefijos = ["0","00","000","0000","00000"] #Prefijos a probar
RANGO_FROM = 0
RANGO_TO   = 200000
REPETICIONES = 1
# -------------


resultados = []

for pref in prefijos:
    tiempos = []
    resultados_rep = []
    for rep in range(REPETICIONES):
        t0 = time.time()
        res = ejecutar_minero(RANGO_FROM, RANGO_TO, pref, cadena)
        t1 = time.time()
        tiempo = t1 - t0
        filas = {
            "prefijo": pref,
            "desde": RANGO_FROM,
            "hasta": RANGO_TO,
            "numero": res.get("numero",0),
            "hash": res.get("hash_md5_result",""),
            "intentos": res.get("intentos",0),
            "tiempo_s": round(tiempo,3),
            "hashes_por_s": round(res.get("intentos",0)/tiempo,1) if tiempo>0 else None
        }
        tiempos.append(tiempo)
        resultados_rep.append(filas)
        print(f"pref={pref} rep={rep+1} -> {filas}")

    resultados.append(resultados_rep[0])

print("\n--- RESULTADOS RESUMEN ---")
print("prefijo | numero | intentos | tiempo_s | hashes/s | hash")
for r in resultados:
    print(f"{r['prefijo']:6} | {r['numero']:6} | {r['intentos']:8} | {r['tiempo_s']:8} | {r['hashes_por_s']:10} | {r['hash'][:8]}...")