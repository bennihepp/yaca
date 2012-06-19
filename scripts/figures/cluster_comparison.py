import sys
import cPickle

import numpy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def do_stuff(fig, axes):
    fig.subplots_adjust(bottom=0.32)

    good = numpy.array([10,6,4,9])
    medium = numpy.array([4,4,6,7])
    bad = numpy.array([6,10,10,4])
    labels = ['K-Means ', 'Single-linkage ', 'Complete-linkage ', 'Ward ']

    x = numpy.arange(len(labels))
    axes.bar(x, good, width=0.8, align='center', color='green', label='Good')
    axes.bar(x, medium, bottom=good, width=0.8, align='center', color='yellow', label='Tie')
    axes.bar(x, bad, bottom=medium+good, width=0.8, align='center', color='red', label='Bad')

    axes.set_xticks(x)
    #labels = range(profile.shape[0])
    xlabels = axes.set_xticklabels(labels, rotation='45')

    xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
    for tick in xticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    v_max = numpy.max(good+medium+bad)
    axes.set_xlim(-1, len(labels))
    axes.set_ylim(0.0, 1.1 * v_max)

    axes.set_ylabel('# of clusters', rotation='90')

    for i in xrange(len(labels)):
        g, m, b = good[i], medium[i], bad[i]
        gs = '%d' % g
        ms = '%d' % m
        bs = '%d' % b
        clr = 'black'
        fs = 16
        weight = 'bold'
        axes.text(x[i], g / 2.0, gs, horizontalalignment='center', verticalalignment='center', color=clr, weight=weight,
                fontsize=fs)
        axes.text(x[i], g + m / 2.0, ms, horizontalalignment='center', verticalalignment='center', color=clr, weight=weight,
                fontsize=fs)
        axes.text(x[i], g + m + b/ 2.0, bs, horizontalalignment='center', verticalalignment='center', color=clr, weight=weight,
                fontsize=fs)

    axes.legend(loc=1)

    axes.grid(True)

params = {
    'axes.labelsize': 24,
    'axes.titlesize' : 32,
    'text.fontsize': 16,
    'legend.fontsize': 15,
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
