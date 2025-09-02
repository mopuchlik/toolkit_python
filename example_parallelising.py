import threading
import time


def io_task(n):
    time.sleep(1)
    print(f"Task {n} done")


start = time.time()

threads = []
for i in range(5):
    t = threading.Thread(target=io_task, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("Threading time:", time.time() - start)

#############
from multiprocessing import Process
import time


def cpu_task(n):
    count = 0
    for i in range(10**7):
        count += i
    print(f"Task {n} done")


start = time.time()

processes = []
for i in range(5):
    p = Process(target=cpu_task, args=(i,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

print("Multiprocessing time:", time.time() - start)

#####################
import asyncio
import time


async def io_task(n):
    await asyncio.sleep(1)
    print(f"Task {n} done")


async def main():
    tasks = [io_task(i) for i in range(5)]
    await asyncio.gather(*tasks)


start = time.time()
asyncio.run(main())
print("Asyncio time:", time.time() - start)
