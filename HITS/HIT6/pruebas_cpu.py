import time
from statistics import mean
from minero_cpu import ejecutar_minero_cpu

# --- CONFIG ---
cadena = "hola mundo"
# prefijos a probar
prefijos = ["0","00","000","0000","00000","000000","0000000"]
REPETICIONES = 1
# ----------------

def medir_prefijo(pref, repeticiones):
    detalles = []
    for rep in range(1, repeticiones + 1):
        t0 = time.perf_counter()
        res = ejecutar_minero_cpu(pref, cadena)
        t1 = time.perf_counter()
        tiempo = t1 - t0

        numero = int(res.get("numero", 0))
        hash_md5 = res.get("hash_md5_result", "")
        intentos = int(res.get("intentos", 0))
        hashes_por_s = round(intentos / tiempo, 1) if tiempo > 0 else None

        detalles.append({
            "prefijo": pref,
            "repeticion": rep,
            "nonce": numero,
            "hash": hash_md5,
            "intentos": intentos,
            "tiempo_s": round(tiempo, 3),
            "hashes_por_s": hashes_por_s,
        })
        
        print(f"pref='{pref}' rep={rep} -> nonce={numero} intentos={intentos} tiempo={round(tiempo,3)}s hashes/s={hashes_por_s}")
    return detalles

def main():

    resumen = []

    for pref in prefijos:
        detalles = medir_prefijo(pref, REPETICIONES)

        tiempos = [d["tiempo_s"] for d in detalles]
        intentos_prom = int(mean(d["intentos"] for d in detalles))
        hashes_prom = round(mean(d["hashes_por_s"] for d in detalles if d["hashes_por_s"] is not None), 1) if any(d["hashes_por_s"] is not None for d in detalles) else None
        encontrado = any(d["nonce"] != 0 for d in detalles)
        nonce_ej = next((d["nonce"] for d in detalles if d["nonce"] != 0), 0)
        hash_ej = next((d["hash"] for d in detalles if d["hash"] != ""), "")

        resumen.append({
            "prefijo": pref,
            "nonce": nonce_ej,
            "encontrado": encontrado,
            "intentos_prom": intentos_prom,
            "tiempo_min": round(min(tiempos), 3),
            "tiempo_prom": round(mean(tiempos), 3),
            "tiempo_max": round(max(tiempos), 3),
            "hashes_s_prom": hashes_prom,
            "hash_ej": hash_ej,
        })

    # imprimir tabla resumen similar al informe
    print("\n--- RESUMEN POR PREFIJO ---\n")
    print("| Prefijo  | Nonce encontrado | Intentos | Tiempo (s) | Hashes/s (aprox.) | Hash (inicio) |")
    print("|----------|-----------------:|---------:|-----------:|------------------:|---------------:|")
    for r in resumen:
        nonce_str = f"{r['nonce']}" if r['encontrado'] else "–"
        hash_start = (r["hash_ej"][:8] + "...") if r["hash_ej"] else "—"
        hashes_str = f"{r['hashes_s_prom']}" if r['hashes_s_prom'] is not None else "—"
        print(f"| `{r['prefijo']}`{' '*(6-len(r['prefijo']))} | {nonce_str:16} | {r['intentos_prom']:8} | {r['tiempo_prom']:9} | {hashes_str:18} | {hash_start:13} |")

    # determinar prefijo más largo encontrado
    prefijos_encontrados = [r for r in resumen if r["encontrado"]]
    if prefijos_encontrados:
        mejor = max(prefijos_encontrados, key=lambda x: len(x["prefijo"]))
        print("\nPrefijo más largo encontrado:", repr(mejor["prefijo"]))
        print("Nonce:", mejor["nonce"])
        print("Hash:", mejor["hash_ej"][:32] if mejor["hash_ej"] else "")
        print("Tiempo promedio (s):", mejor["tiempo_prom"])
    else:
        print("\nNo se encontró ninguna coincidencia para los prefijos probados (en el rango 0..200000).")

if __name__ == "__main__":
    main()