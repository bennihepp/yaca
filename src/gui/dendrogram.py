# -*- coding: utf-8 -*-

"""
dendrogram.py -- Drawing of dendrograms.
"""

# This code is taken from hcluster 0.2.0 (http://pypi.python.org/pypi/hcluster/0.2.0).
# It was slightly modified to allow drawing dendrograms to non-interactive backends in matplotlib.

# hcluster is distributed under the New BSD License:
#
# Author: Damian Eads
# Date:   September 22, 2007
#
# Copyright (c) 2007, 2008, Damian Eads
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   - Redistributions of source code must retain the above
#     copyright notice, this list of conditions and the
#     following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer
#     in the documentation and/or other materials provided with the
#     distribution.
#   - Neither the name of the author nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from hcluster import *
import numpy as np
import matplotlib.pyplot

def _append_singleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func, i, labels):
    # If the leaf id structure is not None and is a list then the caller
    # to dendrogram has indicated that cluster id's corresponding to the
    # leaf nodes should be recorded.

    if lvs is not None:
        lvs.append(int(i))

    # If leaf node labels are to be displayed...
    if ivl is not None:
        # If a leaf_label_func has been provided, the label comes from the
        # string returned from the leaf_label_func, which is a function
        # passed to dendrogram.
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i)))
        else:
            # Otherwise, if the dendrogram caller has passed a labels list
            # for the leaf nodes, use it.
            if labels is not None:
                ivl.append(labels[int(i-n)])
            else:
                # Otherwise, use the id as the label for the leaf.x
                ivl.append(str(int(i)))

def _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func, i, labels, show_leaf_counts):
    # If the leaf id structure is not None and is a list then the caller
    # to dendrogram has indicated that cluster id's corresponding to the
    # leaf nodes should be recorded.

    if lvs is not None:
        lvs.append(int(i))
    if ivl is not None:
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i)))
        else:
            if show_leaf_counts:
                ivl.append("(" + str(int(Z[i-n, 3])) + ")")
            else:
                ivl.append("")

def _append_contraction_marks(Z, iv, i, n, contraction_marks):
    _append_contraction_marks_sub(Z, iv, Z[i-n, 0], n, contraction_marks)
    _append_contraction_marks_sub(Z, iv, Z[i-n, 1], n, contraction_marks)

def _append_contraction_marks_sub(Z, iv, i, n, contraction_marks):
    if (i >= n):
        contraction_marks.append((iv, Z[i-n, 2]))
        _append_contraction_marks_sub(Z, iv, Z[i-n, 0], n, contraction_marks)
        _append_contraction_marks_sub(Z, iv, Z[i-n, 1], n, contraction_marks)



_link_line_colors=['g', 'r', 'c', 'm', 'y', 'k']


def set_link_color_palette(palette):
    """
    Changes the list of matplotlib color codes to use when coloring
    links with the dendrogram color_threshold feature.

    :Arguments:
        - palette : A list of matplotlib color codes. The order of
        the color codes is the order in which the colors are cycled
        through when color thresholding in the dendrogram.

    """

    if type(palette) not in (types.ListType, types.TupleType):
        raise TypeError("palette must be a list or tuple")
    _ptypes = [type(p) == types.StringType for p in palette]

    if False in _ptypes:
        raise TypeError("all palette list elements must be color strings")

    for i in list(_link_line_colors):
        _link_line_colors.remove(i)
    _link_line_colors.extend(list(palette))


def _dendrogram_calculate_info(Z, p, truncate_mode, \
                               color_threshold=np.inf, get_leaves=True, \
                               orientation='top', labels=None, \
                               count_sort=False, distance_sort=False, \
                               show_leaf_counts=False, i=-1, iv=0.0, \
                               ivl=[], n=0, icoord_list=[], dcoord_list=[], \
                               lvs=None, mhr=False, \
                               current_color=[], color_list=[], \
                               currently_below_threshold=[], \
                               leaf_label_func=None, level=0,
                               contraction_marks=None,
                               link_color_func=None):
    """
    Calculates the endpoints of the links as well as the labels for the
    the dendrogram rooted at the node with index i. iv is the independent
    variable value to plot the left-most leaf node below the root node i
    (if orientation='top', this would be the left-most x value where the
    plotting of this root node i and its descendents should begin).

    ivl is a list to store the labels of the leaf nodes. The leaf_label_func
    is called whenever ivl != None, labels == None, and
    leaf_label_func != None. When ivl != None and labels != None, the
    labels list is used only for labeling the the leaf nodes. When
    ivl == None, no labels are generated for leaf nodes.

    When get_leaves==True, a list of leaves is built as they are visited
    in the dendrogram.

    Returns a tuple with l being the independent variable coordinate that
    corresponds to the midpoint of cluster to the left of cluster i if
    i is non-singleton, otherwise the independent coordinate of the leaf
    node if i is a leaf node.

    Returns a tuple (left, w, h, md)

      * left is the independent variable coordinate of the center of the
        the U of the subtree

      * w is the amount of space used for the subtree (in independent
        variable units)

      * h is the height of the subtree in dependent variable units

      * md is the max(Z[*,2]) for all nodes * below and including
        the target node.

    """
    if n == 0:
        raise ValueError("Invalid singleton cluster count n.")

    if i == -1:
        raise ValueError("Invalid root cluster index i.")

    if truncate_mode == 'lastp':
        # If the node is a leaf node but corresponds to a non-single cluster,
        # it's label is either the empty string or the number of original
        # observations belonging to cluster i.
        if i < 2*n-p and i >= n:
            d = Z[i-n, 2]
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func,
                                           i, labels, show_leaf_counts)
            if contraction_marks is not None:
                _append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func, i, labels)
            return (iv + 5.0, 10.0, 0.0, 0.0)
    elif truncate_mode in ('mtica', 'level'):
        if i > n and level > p:
            d = Z[i-n, 2]
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func,
                                           i, labels, show_leaf_counts)
            if contraction_marks is not None:
                _append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func, i, labels)
            return (iv + 5.0, 10.0, 0.0, 0.0)
    elif truncate_mode in ('mlab',):
        pass


    # Otherwise, only truncate if we have a leaf node.
    #
    # If the truncate_mode is mlab, the linkage has been modified
    # with the truncated tree.
    #
    # Only place leaves if they correspond to original observations.
    if i < n:
        _append_singleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func, i, labels)
        return (iv + 5.0, 10.0, 0.0, 0.0)

    # !!! Otherwise, we don't have a leaf node, so work on plotting a
    # non-leaf node.
    # Actual indices of a and b
    aa = Z[i-n, 0]
    ab = Z[i-n, 1]
    if aa > n:
        # The number of singletons below cluster a
        na = Z[aa-n, 3]
        # The distance between a's two direct children.
        da = Z[aa-n, 2]
    else:
        na = 1
        da = 0.0
    if ab > n:
        nb = Z[ab-n, 3]
        db = Z[ab-n, 2]
    else:
        nb = 1
        db = 0.0

    if count_sort == 'ascending' or count_sort == True:
        # If a has a count greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if na > nb:
            # The cluster index to draw to the left (ua) will be ab
            # and the one to draw to the right (ub) will be aa
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif count_sort == 'descending':
        # If a has a count less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if na > nb:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    elif distance_sort == 'ascending' or distance_sort == True:
        # If a has a distance greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if da > db:
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif distance_sort == 'descending':
        # If a has a distance less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if da > db:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    else:
        ua = aa
        ub = ab

    # The distance of the cluster to draw to the left (ua) is uad
    # and its count is uan. Likewise, the cluster to draw to the
    # right has distance ubd and count ubn.
    if ua < n:
        uad = 0.0
        uan = 1
    else:
        uad = Z[ua-n, 2]
        uan = Z[ua-n, 3]
    if ub < n:
        ubd = 0.0
        ubn = 1
    else:
        ubd = Z[ub-n, 2]
        ubn = Z[ub-n, 3]

    # Updated iv variable and the amount of space used.
    (uiva, uwa, uah, uamd) = \
          _dendrogram_calculate_info(Z=Z, p=p, \
                                     truncate_mode=truncate_mode, \
                                     color_threshold=color_threshold, \
                                     get_leaves=get_leaves, \
                                     orientation=orientation, \
                                     labels=labels, \
                                     count_sort=count_sort, \
                                     distance_sort=distance_sort, \
                                     show_leaf_counts=show_leaf_counts, \
                                     i=ua, iv=iv, ivl=ivl, n=n, \
                                     icoord_list=icoord_list, \
                                     dcoord_list=dcoord_list, lvs=lvs, \
                                     current_color=current_color, \
                                     color_list=color_list, \
                                     currently_below_threshold=currently_below_threshold, \
                                     leaf_label_func=leaf_label_func, \
                                     level=level+1, contraction_marks=contraction_marks, \
                                     link_color_func=link_color_func)

    h = Z[i-n, 2]
    if h >= color_threshold or color_threshold <= 0:
        c = 'b'

        if currently_below_threshold[0]:
            current_color[0] = (current_color[0] + 1) % len(_link_line_colors)
        currently_below_threshold[0] = False
    else:
        currently_below_threshold[0] = True
        c = _link_line_colors[current_color[0]]

    (uivb, uwb, ubh, ubmd) = \
          _dendrogram_calculate_info(Z=Z, p=p, \
                                     truncate_mode=truncate_mode, \
                                     color_threshold=color_threshold, \
                                     get_leaves=get_leaves, \
                                     orientation=orientation, \
                                     labels=labels, \
                                     count_sort=count_sort, \
                                     distance_sort=distance_sort, \
                                     show_leaf_counts=show_leaf_counts, \
                                     i=ub, iv=iv+uwa, ivl=ivl, n=n, \
                                     icoord_list=icoord_list, \
                                     dcoord_list=dcoord_list, lvs=lvs, \
                                     current_color=current_color, \
                                     color_list=color_list, \
                                     currently_below_threshold=currently_below_threshold,
                                     leaf_label_func=leaf_label_func, \
                                     level=level+1, contraction_marks=contraction_marks, \
                                     link_color_func=link_color_func)

    # The height of clusters a and b
    ah = uad
    bh = ubd

    max_dist = max(uamd, ubmd, h)

    icoord_list.append([uiva, uiva, uivb, uivb])
    dcoord_list.append([uah, h, h, ubh])
    if link_color_func is not None:
        v = link_color_func(int(i))
        if type(v) != types.StringType:
            raise TypeError("link_color_func must return a matplotlib color string!")
        color_list.append(v)
    else:
        color_list.append(c)
    return ( ((uiva + uivb) / 2), uwa+uwb, h, max_dist)



_dtextsizes = {20: 12, 30: 10, 50: 8, 85: 6, np.inf: 5}
_drotation =  {20: 0,          40: 45,       np.inf: 90}
_dtextsortedkeys = list(_dtextsizes.keys())
_dtextsortedkeys.sort()
_drotationsortedkeys = list(_drotation.keys())
_drotationsortedkeys.sort()

def _remove_dups(L):
    """
    Removes duplicates AND preserves the original order of the elements. The
    set class is not guaranteed to do this.
    """
    seen_before = set([])
    L2 = []
    for i in L:
        if i not in seen_before:
            seen_before.add(i)
            L2.append(i)
    return L2

def _get_tick_text_size(p):
    for k in _dtextsortedkeys:
        if p <= k:
            return _dtextsizes[k]

def _get_tick_rotation(p):
    for k in _drotationsortedkeys:
        if p <= k:
            return _drotation[k]


def _plot_dendrogram(axes, icoords, dcoords, ivl, p, n, mh, orientation, no_labels, color_list, leaf_font_size=None, leaf_rotation=None, contraction_marks=None):
    # Independent variable plot width
    ivw = len(ivl) * 10
    # Depenendent variable plot height
    dvw = mh + mh * 0.05
    ivticks = np.arange(5, len(ivl)*10+5, 10)
    if orientation == 'top':
        axes.set_ylim([0, dvw])
        axes.set_xlim([0, ivw])
        xlines = icoords
        ylines = dcoords
        if no_labels:
            axes.set_xticks([])
            axes.set_xticklabels([])
        else:
            axes.set_xticks(ivticks)
            axes.set_xticklabels(ivl)
        axes.xaxis.set_ticks_position('bottom')
        lbls=axes.get_xticklabels()
        if leaf_rotation:
            matplotlib.pyplot.setp(lbls, 'rotation', leaf_rotation)
        else:
            matplotlib.pyplot.setp(lbls, 'rotation', float(_get_tick_rotation(len(ivl))))
        if leaf_font_size:
            matplotlib.pyplot.setp(lbls, 'size', leaf_font_size)
        else:
            matplotlib.pyplot.setp(lbls, 'size', float(_get_tick_text_size(len(ivl))))
#            txt.set_fontsize()
#            txt.set_rotation(45)
        # Make the tick marks invisible because they cover up the links
        for line in axes.get_xticklines():
            line.set_visible(False)
    elif orientation == 'bottom':
        axes.set_ylim([dvw, 0])
        axes.set_xlim([0, ivw])
        xlines = icoords
        ylines = dcoords
        if no_labels:
            axes.set_xticks([])
            axes.set_xticklabels([])
        else:
            axes.set_xticks(ivticks)
            axes.set_xticklabels(ivl)
        lbls=axes.get_xticklabels()
        if leaf_rotation:
            matplotlib.pyplot.setp(lbls, 'rotation', leaf_rotation)
        else:
            matplotlib.pyplot.setp(lbls, 'rotation', float(_get_tick_rotation(p)))
        if leaf_font_size:
            matplotlib.pyplot.setp(lbls, 'size', leaf_font_size)
        else:
            matplotlib.pyplot.setp(lbls, 'size', float(_get_tick_text_size(p)))
        axes.xaxis.set_ticks_position('top')
        # Make the tick marks invisible because they cover up the links
        for line in axes.get_xticklines():
            line.set_visible(False)
    elif orientation == 'left':
        axes.set_xlim([0, dvw])
        axes.set_ylim([0, ivw])
        xlines = dcoords
        ylines = icoords
        if no_labels:
            axes.set_yticks([])
            axes.set_yticklabels([])
        else:
            axes.set_yticks(ivticks)
            axes.set_yticklabels(ivl)

        lbls=axes.get_yticklabels()
        if leaf_rotation:
            matplotlib.pyplot.setp(lbls, 'rotation', leaf_rotation)
        if leaf_font_size:
            matplotlib.pyplot.setp(lbls, 'size', leaf_font_size)
        axes.yaxis.set_ticks_position('left')
        # Make the tick marks invisible because they cover up the
        # links
        for line in axes.get_yticklines():
            line.set_visible(False)
    elif orientation == 'right':
        axes.set_xlim([dvw, 0])
        axes.set_ylim([0, ivw])
        xlines = dcoords
        ylines = icoords
        if no_labels:
            axes.set_yticks([])
            axes.set_yticklabels([])
        else:
            axes.set_yticks(ivticks)
            axes.set_yticklabels(ivl)
        lbls=axes.get_yticklabels()
        if leaf_rotation:
            matplotlib.pyplot.setp(lbls, 'rotation', leaf_rotation)
        if leaf_font_size:
            matplotlib.pyplot.setp(lbls, 'size', leaf_font_size)
        axes.yaxis.set_ticks_position('right')
        # Make the tick marks invisible because they cover up the links
        for line in axes.get_yticklines():
            line.set_visible(False)

    # Let's use collections instead. This way there is a separate legend item for each
    # tree grouping, rather than stupidly one for each line segment.
    colors_used = _remove_dups(color_list)
    color_to_lines = {}
    for color in colors_used:
        color_to_lines[color] = []
    for (xline,yline,color) in zip(xlines, ylines, color_list):
        color_to_lines[color].append(zip(xline, yline))

    colors_to_collections = {}
    # Construct the collections.
    for color in colors_used:
        coll = matplotlib.collections.LineCollection(color_to_lines[color], colors=(color,))
        colors_to_collections[color] = coll

    # Add all the non-blue link groupings, i.e. those groupings below the color threshold.

    for color in colors_used:
        if color != 'b':
            axes.add_collection(colors_to_collections[color])
    # If there is a blue grouping (i.e., links above the color threshold),
    # it should go last.
    if 'b' in colors_to_collections:
        axes.add_collection(colors_to_collections['b'])

    if contraction_marks is not None:
        #xs=[x for (x, y) in contraction_marks]
        #ys=[y for (x, y) in contraction_marks]
        if orientation in ('left', 'right'):
            for (x,y) in contraction_marks:
                e=matplotlib.patches.Ellipse((y, x), width=dvw/100, height=1.0)
                axes.add_artist(e)
                e.set_clip_box(axes.bbox)
                e.set_alpha(0.5)
                e.set_facecolor('k')
        if orientation in ('top', 'bottom'):
            for (x,y) in contraction_marks:
                e=matplotlib.patches.Ellipse((x, y), width=1.0, height=dvw/100)
                axes.add_artist(e)
                e.set_clip_box(axes.bbox)
                e.set_alpha(0.5)
                e.set_facecolor('k')

            #matplotlib.pylab.plot(xs, ys, 'go', markeredgecolor='k', markersize=3)

            #matplotlib.pylab.plot(ys, xs, 'go', markeredgecolor='k', markersize=3)
#    matplotlib.pylab.draw_if_interactive()



def dendrogram(axis, Z, p=30, truncate_mode=None, color_threshold=None,
           get_leaves=True, orientation='top', labels=None,
           count_sort=False, distance_sort=False, show_leaf_counts=True,
           no_plot=False, no_labels=False, color_list=None,
           leaf_font_size=None, leaf_rotation=None, leaf_label_func=None,
           no_leaves=False, show_contracted=False,
           link_color_func=None):
    r"""
    Plots the hiearchical clustering defined by the linkage Z as a
    dendrogram. The dendrogram illustrates how each cluster is
    composed by drawing a U-shaped link between a non-singleton
    cluster and its children. The height of the top of the U-link is
    the distance between its children clusters. It is also the
    cophenetic distance between original observations in the two
    children clusters. It is expected that the distances in Z[:,2] be
    monotonic, otherwise crossings appear in the dendrogram.
    
    :Arguments:
    
      - Z : ndarray
        The linkage matrix encoding the hierarchical clustering to
        render as a dendrogram. See the ``linkage`` function for more
        information on the format of ``Z``.
    
      - truncate_mode : string
        The dendrogram can be hard to read when the original
        observation matrix from which the linkage is derived is
        large. Truncation is used to condense the dendrogram. There
        are several modes:
    
           * None/'none': no truncation is performed (Default)
    
           * 'lastp': the last ``p`` non-singleton formed in the linkage
           are the only non-leaf nodes in the linkage; they correspond
           to to rows ``Z[n-p-2:end]`` in ``Z``. All other
           non-singleton clusters are contracted into leaf nodes.
    
           * 'mlab': This corresponds to MATLAB(TM) behavior. (not
           implemented yet)
    
           * 'level'/'mtica': no more than ``p`` levels of the
           dendrogram tree are displayed. This corresponds to
           Mathematica(TM) behavior.
    
       - p : int
         The ``p`` parameter for ``truncate_mode``.
    `
       - color_threshold : double
         For brevity, let :math:`t` be the ``color_threshold``.
         Colors all the descendent links below a cluster node
         :math:`k` the same color if :math:`k` is the first node below
         the cut threshold :math:`t`. All links connecting nodes with
         distances greater than or equal to the threshold are colored
         blue. If :math:`t` is less than or equal to zero, all nodes
         are colored blue. If ``color_threshold`` is ``None`` or
         'default', corresponding with MATLAB(TM) behavior, the
         threshold is set to ``0.7*max(Z[:,2])``.
    
       - get_leaves : bool
         Includes a list ``R['leaves']=H`` in the result
         dictionary. For each :math:`i`, ``H[i] == j``, cluster node
         :math:`j` appears in the :math:`i` th position in the
         left-to-right traversal of the leaves, where :math:`j < 2n-1`
         and :math:`i < n`.
    
       - orientation : string
         The direction to plot the dendrogram, which can be any
         of the following strings
    
           * 'top': plots the root at the top, and plot descendent
           links going downwards. (default).
    
           * 'bottom': plots the root at the bottom, and plot descendent
           links going upwards.
    
           * 'left': plots the root at the left, and plot descendent
           links going right.
    
           * 'right': plots the root at the right, and plot descendent
           links going left.
    
       - labels : ndarray
         By default ``labels`` is ``None`` so the index of the
         original observation is used to label the leaf nodes.
    
         Otherwise, this is an :math:`n` -sized list (or tuple). The
         ``labels[i]`` value is the text to put under the :math:`i` th
         leaf node only if it corresponds to an original observation
         and not a non-singleton cluster.
    
       - count_sort : string/bool
         For each node n, the order (visually, from left-to-right) n's
         two descendent links are plotted is determined by this
         parameter, which can be any of the following values:
    
            * False: nothing is done.
    
            * 'ascending'/True: the child with the minimum number of
            original objects in its cluster is plotted first.
    
            * 'descendent': the child with the maximum number of
            original objects in its cluster is plotted first.
    
         Note ``distance_sort`` and ``count_sort`` cannot both be
         ``True``.
    
       - distance_sort : string/bool
         For each node n, the order (visually, from left-to-right) n's
         two descendent links are plotted is determined by this
         parameter, which can be any of the following values:
    
            * False: nothing is done.
    
            * 'ascending'/True: the child with the minimum distance
            between its direct descendents is plotted first.
    
            * 'descending': the child with the maximum distance
            between its direct descendents is plotted first.
    
         Note ``distance_sort`` and ``count_sort`` cannot both be
         ``True``.
    
       - show_leaf_counts : bool
    
         When ``True``, leaf nodes representing :math:`k>1` original
         observation are labeled with the number of observations they
         contain in parentheses.
    
       - no_plot : bool
         When ``True``, the final rendering is not performed. This is
         useful if only the data structures computed for the rendering
         are needed or if matplotlib is not available.
    
       - no_labels : bool
         When ``True``, no labels appear next to the leaf nodes in the
         rendering of the dendrogram.
    
       - leaf_label_rotation : double
    
         Specifies the angle (in degrees) to rotate the leaf
         labels. When unspecified, the rotation based on the number of
         nodes in the dendrogram. (Default=0)
    
       - leaf_font_size : int
         Specifies the font size (in points) of the leaf labels. When
         unspecified, the size based on the number of nodes in the
         dendrogram.
    
       - leaf_label_func : lambda or function
    
         When leaf_label_func is a callable function, for each
         leaf with cluster index :math:`k < 2n-1`. The function
         is expected to return a string with the label for the
         leaf.
    
         Indices :math:`k < n` correspond to original observations
         while indices :math:`k \geq n` correspond to non-singleton
         clusters.
    
         For example, to label singletons with their node id and
         non-singletons with their id, count, and inconsistency
         coefficient, simply do::
    
           # First define the leaf label function.
           def llf(id):
               if id < n:
                   return str(id)
               else:
                   return '[%d %d %1.2f]' % (id, count, R[n-id,3])
    
           # The text for the leaf nodes is going to be big so force
           # a rotation of 90 degrees.
           dendrogram(Z, leaf_label_func=llf, leaf_rotation=90)
    
       - show_contracted : bool
         When ``True`` the heights of non-singleton nodes contracted
         into a leaf node are plotted as crosses along the link
         connecting that leaf node.  This really is only useful when
         truncation is used (see ``truncate_mode`` parameter).
    
       - link_color_func : lambda/function When a callable function,
         link_color_function is called with each non-singleton id
         corresponding to each U-shaped link it will paint. The
         function is expected to return the color to paint the link,
         encoded as a matplotlib color string code.
    
         For example::
    
           dendrogram(Z, link_color_func=lambda k: colors[k])
    
         colors the direct links below each untruncated non-singleton node
         ``k`` using ``colors[k]``.
    
    :Returns:
    
       - R : dict
         A dictionary of data structures computed to render the
         dendrogram. Its has the following keys:
    
           - 'icoords': a list of lists ``[I1, I2, ..., Ip]`` where
           ``Ik`` is a list of 4 independent variable coordinates
           corresponding to the line that represents the k'th link
           painted.
    
           - 'dcoords': a list of lists ``[I2, I2, ..., Ip]`` where
           ``Ik`` is a list of 4 independent variable coordinates
           corresponding to the line that represents the k'th link
           painted.
    
           - 'ivl': a list of labels corresponding to the leaf nodes.
    
           - 'leaves': for each i, ``H[i] == j``, cluster node
           :math:`j` appears in the :math:`i` th position in the
           left-to-right traversal of the leaves, where :math:`j < 2n-1`
           and :math:`i < n`. If :math:`j` is less than :math:`n`, the
           :math:`i` th leaf node corresponds to an original
           observation.  Otherwise, it corresponds to a non-singleton
           cluster.
    """

    # Features under consideration.
    #
    #         ... = dendrogram(..., leaves_order=None)
    #
    #         Plots the leaves in the order specified by a vector of
    #         original observation indices. If the vector contains duplicates
    #         or results in a crossing, an exception will be thrown. Passing
    #         None orders leaf nodes based on the order they appear in the
    #         pre-order traversal.
    Z = np.asarray(Z, order='c')
    
    is_valid_linkage(Z, throw=True, name='Z')
    Zs = Z.shape
    n = Zs[0] + 1
    if type(p) in (types.IntType, types.FloatType):
        p = int(p)
    else:
        raise TypeError('The second argument must be a number')
    
    if truncate_mode not in ('lastp', 'mlab', 'mtica', 'level', 'none', None):
        raise ValueError('Invalid truncation mode.')
    
    if truncate_mode == 'lastp' or truncate_mode == 'mlab':
        if p > n or p == 0:
            p = n
    
    if truncate_mode == 'mtica' or truncate_mode == 'level':
        if p <= 0:
            p = np.inf
    if get_leaves:
        lvs = []
    else:
        lvs = None
    icoord_list=[]
    dcoord_list=[]
    color_list=[]
    current_color=[0]
    currently_below_threshold=[False]
    if no_leaves:
        ivl=None
    else:
        ivl=[]
    if color_threshold is None or \
       (type(color_threshold) == types.StringType and color_threshold=='default'):
        color_threshold = max(Z[:,2])*0.7
    R={'icoord':icoord_list, 'dcoord':dcoord_list, 'ivl':ivl, 'leaves':lvs,
       'color_list':color_list}
    props = {'cbt': False, 'cc':0}
    if show_contracted:
        contraction_marks = []
    else:
        contraction_marks = None
    _dendrogram_calculate_info(Z=Z, p=p,
                               truncate_mode=truncate_mode, \
                               color_threshold=color_threshold, \
                               get_leaves=get_leaves, \
                               orientation=orientation, \
                               labels=labels, \
                               count_sort=count_sort, \
                               distance_sort=distance_sort, \
                               show_leaf_counts=show_leaf_counts, \
                               i=2*n-2, iv=0.0, ivl=ivl, n=n, \
                               icoord_list=icoord_list, \
                               dcoord_list=dcoord_list, lvs=lvs, \
                               current_color=current_color, \
                               color_list=color_list, \
                               currently_below_threshold=currently_below_threshold, \
                               leaf_label_func=leaf_label_func, \
                               contraction_marks=contraction_marks, \
                               link_color_func=link_color_func)
    if not no_plot:
        mh = max(Z[:,2])
        _plot_dendrogram(axis, icoord_list, dcoord_list, ivl, p, n, mh, orientation, no_labels, color_list, leaf_font_size=leaf_font_size, leaf_rotation=leaf_rotation, contraction_marks=contraction_marks)
    
    return R
