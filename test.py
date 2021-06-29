from datetime import datetime
import time

now = 0

def timediff():
    global now
    if not now:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print( current_time)

    else:
        later=datetime.now()
        current_time = later.strftime("%H:%M:%S")
        diff=later-now
        print(diff)

timediff()
time.sleep(100)
timediff()