import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import *
from scipy import optimize

def plot(fig, axes):
    axes.set_title('KMeans runtime for K=50 clusters')
    axes.set_xlabel('Number of cell objects')
    axes.set_ylabel('Runtime in seconds')

    N = np.array([1000, 3162, 10000, 31622, 100000])
    times = np.array([0.034161, 0.117471, 0.548778, 3.374183, 15.323001])
    times_err = np.array([0.004078, 0.018561, 0.084003, 0.737345, 6.137106])

    ##fitfunc = lambda p,x: p[2] * (x**2) + p[1] * (x**1) + p[0]
    #fitfunc = lambda p,x: p[1] * (x**1.5) + p[0]
    #errfunc = lambda p,x,y: fitfunc(p, x) - y

    #p_found, cov_x, infodict, mesg, success = optimize.leastsq(errfunc, [1., 1.], args=(N, times), full_output=True)
    #print p_found

    #new_N = np.concatenate((N, [10**6, 10**7]))
    #print new_N
    #pred_times = fitfunc(p_found, new_N)
    #print pred_times
    #axes.loglog(new_N, pred_times, '-', color='green')
    #print N.shape
    #print pred_times.shape

    axes.errorbar(N, times, yerr=times_err, ecolor='red')
    axes.loglog(N, times, '-', label='machine with 8 cores', color='green')
    #axes.xlim(0.8*1000, 1.2*100000)

    N = np.array([1000, 3162, 10000, 31622, 100000])
    times = np.array([0.054639, 0.272583, 1.924919, 95.798508, 110.483040])
    times_err = np.array([0.006005, 0.044011, 0.490695, 20.565280, 17.829137])

    axes.errorbar(N, times, yerr=times_err, ecolor='red')
    axes.loglog(N, times, '-', label='machine with 1 core', color='blue')
    axes.set_xlim(0.8*1000, 1.2*100000)

    #axes.set_xlim(0.8*1000, 1.2*10**7)

    axes.legend()

fig = plt.figure()
axes = fig.add_subplot(111)
plot(fig, axes)
plt.show()

outputfile = 'runtime_plot.pdf'
pp = PdfPages(outputfile)
plt.switch_backend('cairo.pdf')
fig = plt.figure()
axes = fig.add_subplot(111)
plot(fig, axes)
pp.savefig(fig)
pp.close()
