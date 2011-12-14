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
    xlabels = axes.set_xticklabels(labels, rotation='0')

    axes.set_title( 'Number of cell objects' )
    axes.bar(x - 0.45, total, width=0.8, align='center', color='red', label='Valid cell objects')
    axes.bar(x + 0.45, nonControl, width=0.8, align='center', color='green', label='Non-control-like cell objects')
    
    axes.legend(loc=0)
    
    axes.set_xlim(-1.2, 2*len(labels) - 0.8)
    axes.set_ylabel( 'Number of cells objects', rotation='90' )
    axes.grid( True )

params = {
    'axes.labelsize': 24,
    'axes.titlesize' : 34,
    'text.fontsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 24,
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
