import scipy.stats
import matplotlib

mahalFeatureId = pdc.objFeatureIds['Mahalanobis Distance']
mahalFeatures = pdc.objFeatures[:, mahalFeatureId]
N = len(pdc.treatments)
pageList = zip(N*[3], map(lambda tr: tr.name, pdc.treatments))
plotWindow = multiplot(pageList, show_toolbar=True)
mahalDict = {}
#bins = np.linspace(0, 50, 20)
#bins = 30
#hist_range = [0, 100]
bins = np.linspace(0, 50, 50)
ddbins = 50
normed = True
for i,tr in enumerate(pdc.treatments):
    tr_mask = pl.mask_and(pl.get_treatment_cell_mask(tr.index),
                          pl.get_valid_cell_mask())
    tr_ctrl_mask = pl.mask_and(pl.get_treatment_cell_mask(tr.index),
                               pl.get_control_cell_mask())
    tr_non_ctrl_mask = pl.mask_and(pl.get_treatment_cell_mask(tr.index),
                                   pl.get_non_control_cell_mask())
    mahal = mahalFeatures[tr_mask]**2
    ctrl_mahal = mahalFeatures[tr_ctrl_mask]**2
    non_ctrl_mahal = mahalFeatures[tr_non_ctrl_mask]**2

    plotWindow.get_child_window(i).set_caption(0, 'All cell objects')
    fig, axes = plotWindow.get_child_window(i).get_figure_and_axes(0)
    axes.set_xlim(np.min(bins), np.max(bins))
    axes.set_ylim(0, 0.2)
    n, new_bins, patches = axes.hist(mahal, bins, normed=normed)
    print pl.mahal_cutoff_window
    rect = matplotlib.patches.Rectangle(
        (pl.mahal_cutoff_window[0]**2, 0), pl.mahal_cutoff_window[1]**2-pl.mahal_cutoff_window[0]**2, axes.get_ylim()[1],
        facecolor='yellow', alpha=0.3, fill=True)
    axes.add_patch(rect)
    #if type(bins) is int:
    #    bins = new_bins

    plotWindow.get_child_window(i).set_caption(1, 'Control-like cell objects')
    fig, axes = plotWindow.get_child_window(i).get_figure_and_axes(1)
    axes.set_xlim(np.min(bins), np.max(bins))
    axes.set_ylim(0, 0.2)
    n, new_bins, patches = axes.hist(ctrl_mahal, bins, normed=normed)
    rect = matplotlib.patches.Rectangle(
        (pl.mahal_cutoff_window[0]**2, 0), pl.mahal_cutoff_window[1]**2-pl.mahal_cutoff_window[0]**2, axes.get_ylim()[1],
        facecolor='yellow', alpha=0.3, fill=True)
    axes.add_patch(rect)

    plotWindow.get_child_window(i).set_caption(2, 'Non-control-like cell objects')
    fig, axes = plotWindow.get_child_window(i).get_figure_and_axes(2)
    axes.set_xlim(np.min(bins), np.max(bins))
    axes.set_ylim(0, 0.2)
    n, new_bins, patches = axes.hist(non_ctrl_mahal, bins, normed=normed)
    rect = matplotlib.patches.Rectangle(
        (pl.mahal_cutoff_window[0]**2, 0), pl.mahal_cutoff_window[1]**2-pl.mahal_cutoff_window[0]**2, axes.get_ylim()[1],
        facecolor='yellow', alpha=0.3, fill=True)
    axes.add_patch(rect)

    #plotWindow.set_child_window_title(i, 'Treatment %s' % tr.name)
    #plotWindow.get_child_window(i).draw_histogram(
    #    0, 'Valid cell objects', mahal, bins
    #)
    #plotWindow.get_child_window(i).draw_histogram(
    #    1, 'Control-like cell objects', ctrl_mahal, bins
    #)
    #plotWindow.get_child_window(i).draw_histogram(
    #    2, 'Non-control-like cell objects', non_ctrl_mahal, bins
    #)
    print '%s mean: %f' % (tr.name, np.mean(mahal))

    data = pdc.objFeatures[tr_mask][:, pl.featureIds]
    data = pl.pca.transform(data)
    ddhist, ddedges = np.histogramdd(data, ddbins, normed=True)
    if type(ddbins) == int:
        ddbins = ddedges
    mahalDict[tr.name] = ddhist

print 'featureIds:', pl.featureIds

ddarr = np.empty((len(pdc.treatments), len(pdc.treatments)))
for i, tr1 in enumerate(pdc.treatments):
    for j, tr2 in enumerate(pdc.treatments):
        ddhist1 = mahalDict[tr1.name]
        ddhist2 = mahalDict[tr2.name]
        #dd = np.sum(ddhist1 * ddhist2)
        dd = np.sum((ddhist1 - ddhist2)**2)
        #if tr1.name == 'mock' and tr2.name == 'mock':
        #    ddnorm = dd
        ddarr[tr1.index, tr2.index] = dd
        #print 'tr1=%s, tr2=%s, match=%s' % (tr1.name, tr2.name, dd)

ddarr = ddarr / np.min(ddarr[ddarr > 0])
for i, tr1 in enumerate(pdc.treatments):
    for j, tr2 in enumerate(pdc.treatments):
        dd = ddarr[tr1.index, tr2.index]
        print 'tr1=%s, tr2=%s, match=%s' % (tr1.name, tr2.name, dd)

    #D, p = scipy.stats.ks_2samp(mahalDict[tr.name], mahalDict['mock'])
    #print '%s: D=%f, p=%f' % (tr.name, D, p)

plotWindow.show()
