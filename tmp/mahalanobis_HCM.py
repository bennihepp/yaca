import sys
import cPickle

import numpy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def do_stuff(fig, axes):
    f = open(sys.argv[1], 'r')
    up = cPickle.Unpickler(f)
    d = up.load()
    f.close()

    fig.subplots_adjust( bottom=0.15 )

    labels, total, total_error, nonControl, nonControl_error = d['labels'], d['total'], d['total_error'], d['nonControl'], d['nonControl_error']
    for i,label in enumerate(labels):
        labels[i] = label + ' '

    error_kw = { 'ecolor' : 'black' }

    x = 2 * numpy.arange(0, len(labels))

    axes.set_xticks(x)
    xlabels = axes.set_xticklabels(labels, rotation='90')

    axes.set_title('Median Mahalanobis distance')
    axes.bar(x - 0.45, total, width=0.8, yerr=total_error, error_kw=error_kw, align='center', color='red', label='Valid cell objects')
    axes.bar(x + 0.45, nonControl, width=0.8, yerr=nonControl_error, error_kw=error_kw, align='center', color='green', label='Non-control-like cell objects')
    
    axes.legend(loc=0)
    
    axes.set_xlim(-1.2, 2*len(labels) - 0.8)
    axes.set_ylabel( 'Mahalanobis distance', rotation='90' )
    axes.grid( True )

params = {
    'axes.labelsize': 24,
    'axes.titlesize' : 32,
    'text.fontsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 20,
    'ytick.labelsize': 16,
    'text.usetex': False,
    'figure.figsize': [12,6]}

plt.rcParams.update(params)

fig = plt.figure()
axes = fig.add_subplot(111)
do_stuff(fig, axes)
plt.show()

outputfile = '.'.join(sys.argv[1].split('.')[:-1]) + '.pdf'
pp = PdfPages(outputfile)
plt.switch_backend('cairo.pdf')
fig = plt.figure()
axes = fig.add_subplot(111)
do_stuff(fig, axes)
pp.savefig(fig)
pp.close()
