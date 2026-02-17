import sys
import hashlib

def md5_of_string(s: str) -> str:
    """Devuelve el MD5 (hex) de la cadena s."""
    # hashlib.md5 necesita bytes, por eso encode()
    m = hashlib.md5(s.encode('utf-8'))
    return m.hexdigest()

def main():
    if len(sys.argv) < 2:
        print("Uso: python md5_cpu.py \"texto a hashear\"")
        sys.exit(1)

    texto = sys.argv[1]
    resultado = md5_of_string(texto)
    print(resultado)

if __name__ == "__main__":
    main()
    
#python md5_cpu.py "hola mundo"