from time import sleep
from multiprocessing import Process
 
# function to execute in a new process
def task():
    sleep(1)

# entry point
if __name__ == '__main__':
    # create the process
    process = Process()
    # report the process name
    print(process.name)
    print(process.daemon)
    print(process.is_alive())
    # report the exit status
    print(process.exitcode)
    # report the process identifier
    print(process.pid)
    # start the process
    process.start()
    # report the process identifier
    print(process.pid)
    print(process.is_alive())
    # report the exit status
    print(process.exitcode)
    # wait for the process to finish
    process.join()
    # report the exit status
    print(process.exitcode)