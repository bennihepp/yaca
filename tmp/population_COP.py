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

    fig.subplots_adjust( bottom=0.2 )

    labels, total, nonControl = d['labels'], d['total'], d['nonControl']
    mask = []
    newLabels = []
    for label in labels:
        if label in 'neg9,mock,SAR1B,SAR1A,S24B,S23A_b,S23A_a,S16_b,S16_a,S13,S12_a,CKAP5,CB2_b,CB2_a,CB1_c,CB1_b,CB1_a,CA_b,CA_a'.split(','):
            mask.append(True)
            newLabels.append(label)
        else:
            mask.append(False)
    labels = newLabels
    mask = numpy.array(mask, dtype=bool)
    total = total[mask]
    nonControl = nonControl[mask]
    for i,label in enumerate(labels):
        labels[i] = label + ' '

    x = 2 * numpy.arange(0, len(labels))

    axes.set_xticks(x)
    xlabels = axes.set_xticklabels(labels, rotation='90')

    axes.set_title( 'Number of cell objects' )
    axes.bar(x - 0.45, total, width=0.8, align='center', color='red', label='Valid cell objects')
    axes.bar(x + 0.45, nonControl, width=0.8, align='center', color='green', label='Non-control-like cell objects')
    
    axes.legend(loc=0)
    
    axes.set_xlim(-1.2, 2*len(labels) - 0.8)
    axes.set_ylabel( 'Number of cells objects', rotation='90' )
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
