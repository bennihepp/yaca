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

    labels, total, nonControl = d['labels'], d['total'], d['nonControl']

    x = 2 * numpy.arange(0, len(labels))

    axes.set_xticks(x)
    xlabels = axes.set_xticklabels(labels, rotation='90')

    #plt.set_title( 'Population of valid cell objects' )
    axes.bar(x - 0.45, total, width=0.8, align='center', color='red', label='Valid cell objects')
    axes.bar(x + 0.45, nonControl, width=0.8, align='center', color='green', label='Non-control-like cell objects')
    
    axes.legend(loc=0)
    
    axes.set_xlim(-1.2, 2*len(labels) - 0.8)
    axes.set_ylabel( 'Number of cells', rotation='90' )
    axes.grid( True )

params = {
#    'axes.labelsize': 10,
#    'text.fontsize': 10,
    'legend.fontsize': 12,
#    'xtick.labelsize': 8,
#    'ytick.labelsize': 8,
    'text.usetex': False,
    'figure.figsize': [8,6]}

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
