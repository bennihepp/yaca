import sys, os
import cPickle

import numpy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DRUG_stats = {'total_cells':20831, 'total_images':594, 'valid_cells':11493, 'valid_images':551,
              'noncontrol_cells':3323, 'cutoff_threshold':4.336133, 'median_mahal':3.565376, 'mad_mahal':0.770757}
DRUG_xticks = None
HCM_stats = {'total_cells':118633, 'total_images':1021, 'valid_cells':60738, 'valid_images':947,
              'noncontrol_cells':16024, 'cutoff_threshold':4.386456, 'median_mahal':3.616204, 'mad_mahal':0.770252}
HCM_xticks = [[0, 40000, 80000, 120000], None]
COP_stats = {'total_cells':199997, 'total_images':3828, 'valid_cells':114481, 'valid_images':3749,
              'noncontrol_cells':27005, 'cutoff_threshold':4.399585, 'median_mahal':3.576702, 'mad_mahal':0.822883}
COP_xticks = [None, [0, 1000, 2000, 3000, 4000]]

def do_stuff(fig, name, axes1, axes2, stats, yticks_cells, yticks_images, flag):

    fig.subplots_adjust(left=0.15)

    v = numpy.array([stats['total_cells'], stats['valid_cells'], stats['total_cells'] - stats['valid_cells']])

    x = numpy.arange(0, v.shape[0])
    axes1.set_xticks(x)
    if yticks_cells != None:
        axes1.set_yticks(yticks_cells)
    if flag:
        yticks = axes1.yaxis.get_major_ticks() + axes1.yaxis.get_minor_ticks()
        for tick in yticks:
            #tick.tick1On = False
            #tick.tick2On = False
            tick.label1On = False
            tick.label2On = False
    labels = ['Total ', 'Valid ', 'Invalid ']
    xlabels = axes1.set_xticklabels(labels, rotation='0')
    rects = axes1.bar(x, v, align='center', color=['yellow', 'green', 'red', 'yellow', 'green', 'red'])

    axes1.set_title(name)

    #axes.legend(loc=0)

    axes1.set_ylim(0, 205000)
    axes1.set_xlim(-0.8, len(labels) - 0.2)
    #axes.set_ylabel('Number of cells objects', rotation='90')
    axes1.grid(True)

    yloc = None
    for i,rect in enumerate(rects):
        height = int(rect.get_height())
        if height < 50000:
            yloc = height + 20000
        else:
            yloc = height / 2.0
        if yloc == None:
            yloc = rect.get_y() + 1000
        clr = 'black'
        p = v[i]
        xloc = rect.get_x() + (rect.get_width() / 2.0)
        axes1.text(xloc, yloc, p, rotation='90', horizontalalignment='center', verticalalignment='center', color=clr, weight='normal',
                   fontsize=16)

    return axes1.get_xticks()

    #v = numpy.array([stats['total_images'], stats['valid_images'], stats['total_images'] - stats['valid_images']])

    #y = numpy.arange(0, v.shape[0])
    #axes2.set_yticks(y)
    #if xticks_images:
        #axes2.set_xticks(xticks_images)
    #labels = ['Total', 'Valid', 'Invalid']
    #ylabels = axes2.set_yticklabels(labels[::-1], rotation='0')
    #rects = axes2.barh(y[::-1], v, align='center', color=['yellow', 'green', 'red', 'yellow', 'green', 'red'])

    #axes2.set_title('Number of images')

    ##axes2.legend(loc=0)

    #axes2.set_xlim(0, 1.05*numpy.max(v))
    #axes2.set_ylim(-0.8, len(labels) - 0.2)
    ##axes2.set_ylabel('Number of cells objects', rotation='90')
    #axes2.grid(True)

    #xloc = None
    #for i,rect in enumerate(rects):
        #width = int(rect.get_width())
        #if width < 100:
            #xloc = width + 0.01 * numpy.max(v)
        #else:
            #xloc = width / 2.0
        #if xloc == None:
            #xloc = rect.get_x() + 0.2 * width
        #clr = 'black'
        #p = v[i]
        #yloc = rect.get_y() + (rect.get_height() / 2.0)
        #axes2.text(xloc, yloc, p, horizontalalignment='left', verticalalignment='center', color=clr, weight='normal',
                   #fontsize=16)



params = {
    'axes.titlesize' : 22,
    'axes.labelsize': 18,
    'text.fontsize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 18,
    'text.usetex': False,
    'figure.figsize': [12,6]}

plt.rcParams.update(params)

fig = plt.figure()
ax = []
ax.append(fig.add_subplot(1,3,1))
ax.append(fig.add_subplot(1,3,2, sharex=ax[0]))
ax.append(fig.add_subplot(1,3,3, sharex=ax[0]))

#for i in xrange(3):
    #ax.append(fig.add_subplot('31%d' % (3-i)))
    #print ax[-1]

yticks = numpy.array([0, 50000, 100000, 150000, 200000])
for index, (name, stats, xticks) in enumerate([('DRUG dataset', DRUG_stats, DRUG_xticks), ('HCM dataset', HCM_stats, HCM_xticks), ('COP dataset', COP_stats, COP_xticks)]):
    #if xticks != None:
    #    xticks_cells, xticks_images = xticks
    #else:
    xticks_cells, xticks_images = None, None
    if index > 0:
        flag = True
    else:
        flag = False

    #fig = plt.figure()
    #axes1 = fig.add_subplot(211)
    #axes2 = fig.add_subplot(212)
    axes1 = ax[index]
    axes2 = None
    do_stuff(fig, name, axes1, axes2, stats, yticks, xticks_images, flag)

plt.show()

outputfile = sys.argv[1]
pp = PdfPages(outputfile)
plt.switch_backend('cairo.pdf')
fig = plt.figure()
ax = []
ax.append(fig.add_subplot(1,3,1))
ax.append(fig.add_subplot(1,3,2, sharex=ax[0]))
ax.append(fig.add_subplot(1,3,3, sharex=ax[0]))

for index, (name, stats, xticks) in enumerate([('DRUG dataset', DRUG_stats, DRUG_xticks), ('HCM dataset', HCM_stats, HCM_xticks), ('COP dataset', COP_stats, COP_xticks)]):
    #if xticks != None:
    #    xticks_cells, xticks_images = xticks
    #else:
    xticks_cells, xticks_images = None, None
    if index > 0:
        flag = True
    else:
        flag = False

    #fig = plt.figure()
    #axes1 = fig.add_subplot(211)
    #axes2 = fig.add_subplot(212)
    axes1 = ax[index]
    axes2 = None
    do_stuff(fig, name, axes1, axes2, stats, xticks_cells, xticks_images, flag)

pp.savefig(fig)
pp.close()
