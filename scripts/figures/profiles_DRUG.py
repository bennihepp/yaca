import sys
import cPickle

import numpy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def do_stuff(fig, axes, label1, profile1, label2, profile2):
    fig.subplots_adjust(bottom=0.15)

    #mask = []
    #newLabels = []
    #for label in labels:
        #if label in 'neg9,mock,SAR1B,SAR1A,S24B,S23A_b,S23A_a,S16_b,S16_a,S13,S12_a,CKAP5,CB2_b,CB2_a,CB1_c,CB1_b,CB1_a,CA_b,CA_a'.split(','):
            #mask.append(True)
            #newLabels.append(label)
        #else:
            #mask.append(False)
    #labels = newLabels
    #mask = numpy.array(mask, dtype=bool)
    #total = total[mask]
    #nonControl = nonControl[mask]
    #for i,label in enumerate(labels):
        #labels[i] = label + ' '

    x = 2*numpy.arange(profile.shape[0])
    axes.bar(x-0.45, profile1, width=0.8, align='center', color='yellow', label='replicate 1')
    axes.bar(x+0.45, profile2, width=0.8, align='center', color='cyan', label='replicate 2')
    axes.set_xticks(x)
    labels = range(profile.shape[0])
    xlabels = axes.set_xticklabels(labels, rotation='0')

    axes.legend(loc=0)

    xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
    for tick in xticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    axes.set_xlim(-1.5, 2*profile.shape[0] - .5)
    axes.set_ylim(0.0, 1.05 * numpy.max([profile1, profile2]))

    axes.set_xlabel('cluster index')
    axes.set_ylabel('cluster profile value', rotation='90')

    label = label1.split(',')[0]
    #axes.set_title(label)

    axes.grid(True)

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


f = open(sys.argv[1], 'r')
up = cPickle.Unpickler(f)
d = up.load()
f.close()

labels, profiles = d['labels'], d['profiles']

for i, (label, profile) in enumerate(zip(labels, profiles)):

    if i % 2 != 0:
        continue

    label1 = label
    profile1 = profile
    label2 = labels[i+1]
    profile2 = profiles[i+1]

    if i == 0:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        do_stuff(fig, axes, label1, profile1, label2, profile2)
        plt.show()
    
    outputfile = '.'.join(sys.argv[1].split('.')[:-1]) + ('_%s.pdf' % label)
    pp = PdfPages(outputfile)
    plt.switch_backend('cairo.pdf')
    fig = plt.figure()
    axes = fig.add_subplot(111)
    do_stuff(fig, axes, label1, profile1, label2, profile2)
    pp.savefig(fig)
    pp.close()
