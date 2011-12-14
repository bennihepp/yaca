import sys
import cPickle

import numpy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def do_stuff(fig, axes, label1, profile1, label2, profile2, label3, profile3):
    fig.subplots_adjust( bottom=0.15 )

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

    profile1 = profile1[::-1]
    profile2 = profile2[::-1]

    repl1, repl2, repl3 = 'replicate 1', 'replicate 2', 'replicate 3'
    if label1.startswith('121'):
        repl1 = 'treatment 121 = CLPB'
        repl2 = 'treatment 130 = GPT'

    if profile3 == None:
        x = 2*numpy.arange(profile.shape[0])
        axes.bar(x-0.45, profile1, width=0.8, align='center', color='yellow', label=repl1)
        axes.bar(x+0.45, profile2, width=0.8, align='center', color='cyan', label=repl2)

    else:
        x = 3*numpy.arange(profile.shape[0])
        axes.bar(x-0.9, profile1, width=0.8, align='center', color='yellow', label=repl1)
        axes.bar(x, profile2, width=0.8, align='center', color='cyan', label=repl2)
        axes.bar(x+0.9, profile3, width=0.8, align='center', color='magenta', label=repl3)
        axes.set_xticks(x)

    axes.set_xticks(x)
    labels = range(profile.shape[0])
    xlabels = axes.set_xticklabels(labels, rotation='0')

    leg = axes.legend(loc=3, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
    for tick in xticks:
        tick.tick1On = False
        tick.tick2On = False
        tick.label1On = True
        tick.label2On = False

    if profile3 == None:
        v_max = numpy.max([profile1, profile2])
    else:
        v_max = numpy.max([profile1, profile2, profile3])
    axes.set_xlim(-1.5, 2*profile.shape[0] - .5)
    axes.set_ylim(0.0, 1.05 * v_max)

    axes.set_xlabel('cluster index')
    axes.set_ylabel('cluster profile value', rotation='90')

    label = label1.split(',')[0]
    #axes.set_title(label)

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


f = open(sys.argv[1], 'r')
up = cPickle.Unpickler(f)
d = up.load()
f.close()

labels, profiles = d['labels'], d['profiles']

groups = [
    ['mock,[0]', 'mock,[1]'],
    ['neg9,[0]', 'neg9,[1]'],
    ['SAR1A,[0]', 'SAR1A,[1]'],
    ['SAR1B,[0]', 'SAR1B,[1]'],
    ['CA_b,[0]', 'CA_b,[1]'],
    ['CB1_b,[0]', 'CB1_b,[1]'],
    ['CB1_a,[0]', 'CB1_a,[1]'],
    ['CB1_c,[0]', 'CB1_c,[1]']
]

import itertools
for i, (label, profile) in enumerate(zip(labels, profiles)):

    label1 = label
    profile1 = profile
    label2 = label3 = None
    profile2 = profile3 = None

    for group in groups:
        if label == group[0]:
            g = list(itertools.ifilter(lambda x: x != label, group))
            #g = list(iterator)
            if len(g) > 0:
                label2 = g[0]
            if len(g) > 1:
                label3 = g[1]
            break
    if label2 != None:
        for l, p in zip(labels, profiles):
            if l == label2:
                profile2 = p
                break
    if label3 != None:
        for l, p in zip(labels, profiles):
            if l == label3:
                profile3 = p
                break
    if profile2 == None:
        continue
    if profile3 == None:
        label3 = None

    if label1 in groups[0]:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        do_stuff(fig, axes, label1, profile1, label2, profile2, label3, profile3)
        plt.show()
    
    outputfile = '.'.join(sys.argv[1].split('.')[:-1]) + ('_%s.pdf' % label.split(',')[0])
    pp = PdfPages(outputfile)
    plt.switch_backend('cairo.pdf')
    fig = plt.figure()
    axes = fig.add_subplot(111)
    do_stuff(fig, axes, label1, profile1, label2, profile2, label3, profile3)
    pp.savefig(fig)
    pp.close()
