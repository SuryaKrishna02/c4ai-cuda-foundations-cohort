from time import sleep
from random import random
from multiprocessing import Process
from multiprocessing import Queue
from queue import Empty
 
# generate work
def producer(queue):
    print('Producer: Running', flush=True)
    # generate work
    for _ in range(10):
        # generate a value
        value = random()
        # block
        sleep(value)
        # add to the queue
        queue.put(value)
    # all done
    queue.put(None)
    print('Producer: Done', flush=True)
 
# consume work
def consumer(queue):
    print('Consumer: Running', flush=True)
    # consume work
    while True:
        # get a unit of work
        try:
            item = queue.get(block=False)
        except Empty:
            print('Consumer: got nothing, waiting a while...', flush=True)
            sleep(0.5)
            continue
        # check for stop
        if item is None:
            break
        # report
        print(f'>got {item}', flush=True)
    # all done
    print('Consumer: Done', flush=True)
 
# entry point
if __name__ == '__main__':
    # create the shared queue
    queue = Queue()
    # start the consumer process
    consumer_process = Process(target=consumer, args=(queue,))
    consumer_process.start()
    # start the producer process
    producer_process = Process(target=producer, args=(queue,))
    producer_process.start()
    # wait for all processes to finish
    producer_process.join()
    consumer_process.join()