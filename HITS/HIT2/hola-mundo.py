#Hola mundo CPU

blocks = 2
threads_per_block = 4

print("Version CPU del hola mundo")
for block in range(blocks):
    for thread in range(threads_per_block):
        global_id = block * threads_per_block + thread
        print(f"Hola mundo desde el hilo global {global_id} (block {block}, thread {thread})")