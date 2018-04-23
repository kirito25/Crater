from multiprocessing.dummy import Pool as ThreadPool
import sys
import os
import subprocess
import datetime

LOGFILEDIR = datetime.datetime.now().strftime('log_%H_%M_%d_%m_%Y')

# command list is an array that is [command, logname]
def commandexec(commandlist):
    print "Command %s started" % (str(commandlist[1]))
    logname = "./%s/%s" % (LOGFILEDIR, commandlist[1]) 
    log = open(logname, 'w')
    process = subprocess.Popen(commandlist[0], stdout=log, stderr=log, shell=True)
    process.wait()
    print "Command %s finished" % (str(commandlist[1]))

def commandbuilder(hidden1, hidden2, hidden3):
    output = "python run_experiment.py -N 10 -S "
    if hidden1:
        output += "%s " % (str(hidden1))
    if hidden2:
        output += "%s " % (str(hidden2))
    if hidden3:
        output += "%s " % (str(hidden3))
    output += "-eta 2.02 -m 50"
    return output

def bruteforcemode():
    commandlist = []
#    hidden1 = 0
#    hidden2 = 0
#    hidden3 = 0
    index = 0
    for i in range(51):
        for j in range(51):
            for k in range(51):
                command = [commandbuilder(i, j, k), index]
                commandlist.append(command)
                index += 1
    pool = ThreadPool(10)
    pool.map(commandexec, commandlist)
    pool.close()
    pool.join()

def regularmode(commandfile):
    pool = ThreadPool(2)
    commandlist = []
    with open(commandfile) as commands:
        commands = commands.read().splitlines()
    for i in range(len(commands)):
        commandline = []
        commandline.append(commands[i])
        commandline.append(i)
        commandlist.append(commandline)
    pool.map(commandexec, commandlist)
    pool.close()
    pool.join()

def main():
    try:
        os.makedirs(LOGFILEDIR)
    except:
        print "directory exists"
    bruteforce = False
    try:
        commandfile = sys.argv[1]
    except:
        bruteforce = True
    #if bruteforce:
    bruteforcemode()
#    else:
#        regularmode(commandfile)


main()
