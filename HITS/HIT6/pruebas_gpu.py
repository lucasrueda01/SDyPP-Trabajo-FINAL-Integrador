

#Este codigo no va a funcionar porque ejecuta el ejecutar minero de otra carpeta, 
#para ello habria que poner bien la ruta, pero lo dejo asi pq lo tengo ejecutado en el Google Colab

import time
from statistics import mean
from minero_gpu import ejecutar_minero

# --- CONFIG ---
cadena = "hola mundo"
prefijos = ["0","00","000","0000","00000","000000","0000000"]  # prefijos a probar
REPETICIONES = 1   # cuantas ejecuciones por cada prefijo
# ----------------

resumen = []
all_runs = []

for pref in prefijos:
    print(f"\n--- Prefijo: '{pref}' ---")
    tiempos = []
    hashes_por_s_list = []
    resultados_rep = []

    for rep in range(1, REPETICIONES + 1):
        print(f"Ejecutando rep {rep}/{REPETICIONES} ...", end="", flush=True)
        t0 = time.perf_counter()
        res = ejecutar_minero(pref, cadena)
        t1 = time.perf_counter()
        tiempo = t1 - t0

        numero = int(res.get("numero", 0))
        hash_md5 = res.get("hash_md5_result", "")
        intentos = int(res.get("intentos", 0))

        hashes_por_s = round(intentos/tiempo, 1) if tiempo > 0 else None

        fila = {
            "prefijo": pref,
            "repeticion": rep,
            "nonce": numero,
            "hash": hash_md5,
            "intentos": intentos,
            "tiempo_s": round(tiempo, 6),
            "hashes_por_s": hashes_por_s,
        }

        resultados_rep.append(fila)
        all_runs.append(fila)
        tiempos.append(tiempo)
        if hashes_por_s is not None:
            hashes_por_s_list.append(hashes_por_s)

        print(f" OK -> nonce={numero} intentos={intentos} tiempo={fila['tiempo_s']}s hashes/s={fila['hashes_por_s']}")

    resumen_pref = {
        "prefijo": pref,
        "repeticiones": REPETICIONES,
        "encontrado_en_alguna": any(r["nonce"] != 0 for r in resultados_rep),
        "nonce_ejemplo": next((r["nonce"] for r in resultados_rep if r["nonce"] != 0), 0),
        "hash_ejemplo": next((r["hash"] for r in resultados_rep if r["nonce"] != 0), ""),
        "intentos_promedio": int(mean(r["intentos"] for r in resultados_rep)),
        "tiempo_min_s": round(min(tiempos), 6),
        "tiempo_prom_s": round(mean(tiempos), 6),
        "tiempo_max_s": round(max(tiempos), 6),
        "hashes_s_prom": round(mean(hashes_por_s_list), 1) if hashes_por_s_list else None
    }
    resumen.append(resumen_pref)

# imprimir resumen por prefijo
print("\n\n--- RESUMEN POR PREFIJO ---")
print("pref | found | nonce | intentos_prom | tiempo_prom_s | hashes/s_prom")
for r in resumen:
    print(f"{r['prefijo']:5} | {str(r['encontrado_en_alguna']):5} | {r['nonce_ejemplo']:9} | {r['intentos_promedio']:13} | {r['tiempo_prom_s']:12} | {r['hashes_s_prom']}")

# determinar el prefijo más largo encontrado
prefijos_encontrados = [r for r in resumen if r["encontrado_en_alguna"]]
if prefijos_encontrados:
    mejor = max(prefijos_encontrados, key=lambda x: len(x["prefijo"]))
    print("\nPrefijo más largo encontrado:", repr(mejor["prefijo"]))
    print("Nonce:", mejor["nonce_ejemplo"])
    print("Hash:", mejor["hash_ejemplo"])
    print("Tiempo promedio (s):", mejor["tiempo_prom_s"])
else:
    print("\nNo se encontró ninguna coincidencia para los prefijos probados en los rangos internos del binario CUDA.")