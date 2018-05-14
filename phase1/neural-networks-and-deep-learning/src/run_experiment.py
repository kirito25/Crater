import crater_loader, crater_network
import sys
import argparse

parser = argparse.ArgumentParser(description="Run Crater experiment.")
parser.add_argument("-N", metavar='N', type=int, default=1 , help="number of experiments, default is 1")
parser.add_argument("-S", metavar='sizes', nargs="+", type=int, default=[50,20],
                    help="size of hidden layer, default is [50, 20]")
parser.add_argument("-eta", metavar='eta', type=float, default=2.0 , help="the learning rate, default is 2.0")
parser.add_argument("-m", metavar='m', type=int, default=100 , help="the mini batch size, default is 100")
parser.add_argument("-epoch", metavar='epoch', type=int, default=10 , help="the number of epoch, default is 10")

args = parser.parse_args()
N = args.N
eta = args.eta
epoch = args.epoch
mini_batch_size = args.m
training_data, validation_data, test_data = crater_loader.load_data_wrapper()
sizes = [len(training_data[0][0][0]) ] + args.S + [len(training_data[0][1])]
print "Running %d experiments with layers %s, eta = %s, mini_batch_size = %s, epoch = %s" % \
                                                            (N, sizes, eta, mini_batch_size, epoch)

fp = 0.0
fn = 0.0
tp = 0.0
detection_rate = 0.0
false_rate = 0.0
quality_rate = 0.0
hard = {}

for i in range(N):
    print "Running experiment #%s" % (i + 1)
    net = crater_network.Network(sizes)
    net.SGD(training_data, epoch, mini_batch_size, eta, test_data)
    print "Running evaluate on validation_data of size %d" % (len(validation_data))
    net.evaluate(validation_data, show = True, updatelist = True)
    # when show = True it does not print a new line
    print
    fp += net.fp
    fn += net.fn
    tp += net.tp
    # adding 1 to avoid divide by 0
    detection_rate += float(net.tp) / (net.tp + net.fn)
    false_rate     += float(net.fp) / (net.tp + net.fp)
    quality_rate   += float(net.tp) / (net.tp + net.fp + net.fn)

    for (path, output) in net.failures:
        try:
            hard[path][0] += 1
        except KeyError:
            hard[path] = [1, output]

    # Load data every time at random
    training_data, validation_data, test_data = crater_loader.load_data_wrapper()


print "\nAverages of runs:"
fp = fp / N
fn = fn / N
tp = tp / N
detection_rate = detection_rate / N
false_rate = false_rate / N
quality_rate = quality_rate / N

s = "TP = %d  FP = %d  FN = %d  " % (tp, fp, fn)
s += "detection_rate = %.2f  false_rate = %.2f  quality_rate = %.2f " % (detection_rate, 
                                                                         false_rate,
                                                                         quality_rate)
print s

for path, val in hard.items():
    if val[0] > 1:
        print "%s : count = %s  output = %s" % (path, val[0], val[1].flatten().tolist())

