import scipy.stats
import matplotlib

mahalFeatureId = pdc.objFeatureIds['Mahalanobis Distance']
mahalFeatures = pdc.objFeatures[:, mahalFeatureId]
N = len(pdc.treatments)
pageList = zip(N*[3], N*['Page'])
plotWindow = multiplot(pageList, show_toolbar=True)
mahalDict = {}
#bins = np.linspace(0, 50, 20)
#bins = 30
#hist_range = [0, 100]
bins = np.linspace(0, 50, 50)
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

    plotWindow.get_child_window(0).set_caption(i, tr.name)
    fig, axes = plotWindow.get_child_window(0).get_figure_and_axes(i)
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

    plotWindow.get_child_window(1).set_caption(i, tr.name)
    fig, axes = plotWindow.get_child_window(1).get_figure_and_axes(i)
    axes.set_xlim(np.min(bins), np.max(bins))
    axes.set_ylim(0, 0.2)
    n, new_bins, patches = axes.hist(ctrl_mahal, bins, normed=normed)
    rect = matplotlib.patches.Rectangle(
        (pl.mahal_cutoff_window[0]**2, 0), pl.mahal_cutoff_window[1]**2-pl.mahal_cutoff_window[0]**2, axes.get_ylim()[1],
        facecolor='yellow', alpha=0.3, fill=True)
    axes.add_patch(rect)

    plotWindow.get_child_window(2).set_caption(i, tr.name)
    fig, axes = plotWindow.get_child_window(2).get_figure_and_axes(i)
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
    mahalDict[tr.name] = mahal

for i,tr in enumerate(pdc.treatments):
    D, p = scipy.stats.ks_2samp(mahalDict[tr.name], mahalDict['mock'])
    print '%s: D=%f, p=%f' % (tr.name, D, p)

plotWindow.show()
