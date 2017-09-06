# -*- coding: utf-8 -*-
"""
Created by e-bug on 24/03/17.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib2tikz import save as tikz_save
import numpy as np


# colorblind palette
colorblind_palette_dict = {'black': (0.0,0.0,0.0), 
                           'orange': (0.9,0.6,0.0), 
                           'sky_blue': (0.35,0.7,0.9), 
                           'bluish_green': (0.0,0.6,0.5), 
                           'yellow': (0.95,0.9,0.25), 
                           'blue': (0.0,0.45,0.7), 
                           'vermillion': (0.8,0.4,0.0), 
                           'reddish_purple': (0.8,0.6,0.7)}
palette_order = {0: 'vermillion', 1: 'bluish_green', 2: 'sky_blue', 3: 'orange', 
                 4: 'black', 5: 'yellow', 6: 'blue', 7: 'reddish_purple'}
palette_order2 = {0: 'blue', 1: 'orange', 2: 'bluish_green', 3: 'yellow', 
                 4: 'reddish_purple', 5: 'sky_blue', 6: 'vermillion', 7: 'black'}
n_colors = len(colorblind_palette_dict)

# ideal line
ideal_color = (0.5,0.5,0.5)

# markers
markers = ['o', '^', 's', 'D', '*', 'h', '.', '+']
n_markers = len(markers)

# linestyles
linestyles = [':', '-.', '--', '-']
n_linestyles = len(linestyles)


def plot_tts(n_nodes, lines, labels=None, legend_title='Problem size', xlabel='Number of nodes', xscale='log2',
             ylabel='Time to solution [s]', yscale='log', cmap_name=None, filename=None, saveas='tikz',
             figureheight = '\\figureheight', figurewidth = '\\figurewidth'):
    """
    Plots the time to solution as a function of the number of nodes.
    :param n_nodes: list of values in x-axis (i.e. number of nodes)
    :param lines: list of lists, each with y values for each x value
    :param labels: labels of the lines
    :param legend_title: title of the legend
    :param xlabel: label of x-axis
    :param xscale: scale of x-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param ylabel: label of y-axis
    :param yscale: scale of y-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param cmap_name: name of colormap to be used (see: http://matplotlib.org/examples/color/colormaps_reference.html).
                      If None, colorblind palette is used
    :param saveas:
    """

    plt.figure(figsize=(12,8))
    plt.grid()

    # colormap
    n_lines = len(lines)
    line_colors = []
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        line_colors = cmap(np.linspace(0.25, 0.9, n_lines))
    else:
        line_colors = [colorblind_palette_dict[palette_order[i%n_colors]] for i in range(n_lines)]

    # plot lines
    for i,tts in enumerate(lines):
        plt.plot(n_nodes, tts, 
                 color=line_colors[i], linestyle=linestyles[i%n_linestyles], 
                 marker=markers[i%n_markers], markerfacecolor=line_colors[i], markersize=7)
    # x-axis
    if xscale == 'log2':
        plt.xscale('log', basex=2)
    elif xscale == 'log':
        plt.xscale('log')
    plt.xticks(n_nodes, fontsize='large')
    plt.xlabel(xlabel, fontsize='x-large')

    # y-axis
    if yscale == 'log2':
        plt.yscale('log', basex=2)
    elif yscale == 'log':
        plt.yscale('log')
    plt.yticks(fontsize='large')
    plt.ylabel(ylabel, fontsize='x-large')

    # legend
    if labels is not None:
        if len(labels) == n_lines:
            legend = plt.legend(labels, loc='upper right', bbox_to_anchor=[1, 1], 
                                ncol=min(n_lines,4), shadow=False, fancybox=True,
                                title=legend_title, fontsize='large')
            plt.setp(legend.get_title(),fontsize='x-large')
        else:
            raise ValueError('Number of labels does not match number of lines')

    # ticks formatting
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # ax.yaxis.set_major_formatter(ScalarFormatter())

    # save figure
    if (saveas is None) or (filename is None):
        plt.show()
    elif saveas == 'tikz':
        tikz_save(filename + '.' + saveas, figureheight = figureheight, figurewidth = figurewidth)
    else:
        plt.savefig(filename + '.' + saveas)
        


def plot_speedup(n_nodes, lines, ax=None, labels=None, title=None, 
                 legend_title='Problem size', legend_loc='upper left',
                 xlabel='Number of nodes', xscale='log2', ylabel='Speedup', yscale='log2', 
                 plot_ideal=True, cmap_name=None, cmap_min_grad=0.25, cmap_max_grad=0.9,
                 filename=None, saveas=None, figureheight = '\\figureheight', figurewidth = '\\figurewidth'):
    """
    Plots the speedup as a function of the number of nodes.
    :param n_nodes: list of values in x-axis (i.e. number of nodes)
    :param lines: list of lists, each with y values for each x value
    :param ax: a single matplotlib.axes.Axes object
    :param labels: labels of the lines
    :param legend_title: title of the legend
    :param legend_loc: location of the legend (e.g., "upper right")
    :param xlabel: label of x-axis
    :param xscale: scale of x-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param ylabel: label of y-axis
    :param yscale: scale of y-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param plot_ideal: if True, plots ideal speedup line
    :param cmap_name: name of colormap to be used (see: http://matplotlib.org/examples/color/colormaps_reference.html).
                      If None, colorblind palette is used
                      If a list of (r,g,b) tuples is passed, each tuple is applied to a different line
    :param cmap_min_grad: minimum value for cmap_name ([0.0, 1.0])
    :param cmap_max_grad: maximum value for cmap_name ([0.0, 1.0])
    :param saveas: either "show", "tikz", or any other extension (such as "png" and "svg").
                   Use None (default) if you want to use the plot in a subplot
    """

    if ax is None:
        f, ax = plt.subplots(figsize=(12,8))
    ax.grid()

    # colormap
    n_lines = len(lines)
    line_colors = []
    if type(cmap_name) is list:
        line_colors = cmap_name
    elif cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        line_colors = cmap(np.linspace(cmap_min_grad, cmap_max_grad, n_lines))
    else:
        line_colors = [colorblind_palette_dict[palette_order2[i%n_colors]] for i in range(n_lines)]

    # plot lines
    for i,tts in enumerate(lines):
        ax.plot(n_nodes, tts, 
                color=line_colors[i], linestyle=linestyles[i%n_linestyles], linewidth=3,
                marker=markers[i%n_markers], markersize=7)

    if plot_ideal:
        ax.plot(n_nodes, n_nodes, linewidth=3, color=ideal_color)
        ax.text(n_nodes[-2]+0.02*n_nodes[-2], n_nodes[-2]+0.2*n_nodes[-2], 
                'ideal', fontsize='x-large', color=ideal_color)

    # x-axis
    if xscale == 'log2':
        ax.set_xscale('log', basex=2)
    elif xscale == 'log':
        ax.set_xscale('log')
    ax.set_xticklabels(n_nodes, minor=True, fontsize='large')
    ax.set_xlabel(xlabel, fontsize='x-large')

    # y-axis
    if yscale == 'log2':
        ax.set_yscale('log', basex=2)
    elif yscale == 'log':
        ax.set_yscale('log')
    ax.set_yticklabels(n_nodes, minor=True, fontsize='large')
    #ax.minorticks_off()
    ax.set_ylabel(ylabel, fontsize='x-large')

    # title
    if title is not None:
        ax.set_title(title, fontsize='xx-large')
    
    # legend
    if labels is not None:
        if len(labels) == n_lines:
            legend = ax.legend(labels, loc=legend_loc,
                                ncol=min(n_lines,4), shadow=False, fancybox=True,
                                title=legend_title, fontsize='large')
            plt.setp(legend.get_title(),fontsize='x-large')
        else:
            raise ValueError('Number of labels does not match number of lines')

    # ticks formatting
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.0f'))
    
    ax.grid(which='minor', linestyle='-', linewidth='0.5')
    
    # save figure
    if saveas is None:
        pass
    elif saveas == 'show':
        plt.show()
    elif saveas == 'tikz':
        assert filename is not None, "No filename specified"
        tikz_save(filename + '.' + saveas, figureheight = figureheight, figurewidth = figurewidth)
    else:
        assert filename is not None, "No filename specified"
        plt.savefig(filename + '.' + saveas)


def plot_efficiency(n_nodes, lines, labels=None, legend_title='Problem size', xlabel='Number of nodes', xscale='log2',
                    ylabel='Efficiency', yscale=None, plot_ideal=True, cmap_name=None, filename=None, saveas='tikz',
                    figureheight = '\\figureheight', figurewidth = '\\figurewidth'):
    """
    Plots the efficiency as a function of the number of nodes.
    :param n_nodes: list of values in x-axis (i.e. number of nodes)
    :param lines: list of lists, each with y values for each x value
    :param labels: labels of the lines
    :param legend_title: title of the legend
    :param xlabel: label of x-axis
    :param xscale: scale of x-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param ylabel: label of y-axis
    :param yscale: scale of y-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param plot_ideal: if True, plots ideal speedup line
    :param cmap_name: name of colormap to be used (see: http://matplotlib.org/examples/color/colormaps_reference.html).
                      If None, colorblind palette is used
    :param saveas:
    """

    plt.figure(figsize=(12,8))
    plt.grid()

    # colormap
    n_lines = len(lines)
    line_colors = []
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        line_colors = cmap(np.linspace(0.25, 0.9, n_lines))
    else:
        line_colors = [colorblind_palette_dict[palette_order[i%n_colors]] for i in range(n_lines)]

    # plot lines
    for i,tts in enumerate(lines):
        plt.plot(n_nodes, tts, 
                 color=line_colors[i], linestyle=linestyles[i%n_linestyles], 
                 marker=markers[i%n_markers], markerfacecolor=line_colors[i], markersize=7)

    if plot_ideal:
        plt.plot(n_nodes, np.ones(len(n_nodes)), color=ideal_color)
        plt.text(n_nodes[-1]-10, 0.96, 'ideal', fontsize='x-large')

    # x-axis
    if xscale == 'log2':
        plt.xscale('log', basex=2)
    elif xscale == 'log':
        plt.xscale('log')
    plt.xticks(n_nodes, fontsize='large')
    plt.xlabel(xlabel, fontsize='x-large')

    # y-axis
    if yscale == 'log2':
        plt.yscale('log', basex=2)
    elif yscale == 'log':
        plt.yscale('log')
    plt.yticks(fontsize='large')
    plt.ylabel(ylabel, fontsize='x-large')

    # legend
    if labels is not None:
        if len(labels) == n_lines:
            legend = plt.legend(labels, loc='lower left',
                                ncol=min(n_lines,4), shadow=False, fancybox=True,
                                title=legend_title, fontsize='large')
            plt.setp(legend.get_title(),fontsize='x-large')
        else:
            raise ValueError('Number of labels does not match number of lines')

    # ticks formatting
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax.yaxis.set_major_formatter(ScalarFormatter())

    # save figure
    if (saveas is None) or (filename is None):
        plt.show()
    elif saveas == 'tikz':
        tikz_save(filename + '.' + saveas, figureheight = figureheight, figurewidth = figurewidth)
    else:
        plt.savefig(filename + '.' + saveas)


def plot_tts_bar(machines, values_lists, ax=None, width=0.35, labels=None, title=None, 
                 legend_title='Program', legend_loc='upper right', xlabel='Machine', 
                 ylabel='Time to solution [s]', yscale=None,  ymin=None, ymax=None, 
                 cmap_name=None, cmap_min_grad=0.25, cmap_max_grad=0.9,
                 filename=None, saveas=None, figureheight='\\figureheight', figurewidth='\\figurewidth'):
    """
    Plots the time to solution as a function of the number of nodes.
    :param machines: list of strings for each machine
    :param values_lists: list of lists of values corresponding to the passed machines and programs
    :param ax: a single matplotlib.axes.Axes object
    :param width: the width of the bars
    :param labels: labels of the programs
    :param title: title of the plot
    :param legend_title: title of the legend
    :param legend_loc: location of the legend (e.g., "upper right")
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param yscale: scale of y-axis: None: normal, log: base-10 logarithm, log2: base-2 logarithm
    :param ymin: minimum y value
    :param ymax: maximum y value
    :param cmap_name: name of colormap to be used (see: http://matplotlib.org/examples/color/colormaps_reference.html).
                      If None, colorblind palette is used
                      If a list of colormaps is passed, each colormap is applied to a different machine
    :param cmap_min_grad: minimum value for cmap_name ([0.0, 1.0])
    :param cmap_max_grad: maximum value for cmap_name ([0.0, 1.0])
    :param saveas: either "show", "tikz", or any other extension (such as "png" and "svg").
                   Use None (default) if you want to use the plot in a subplot
    """

    if ax is None:
        f, ax = plt.subplots(figsize=(12,8))
    ax.grid()

    # colormap
    n_machines = len(machines)
    n_labels = len(values_lists[0])
    bar_colors = []
    machine_colors = []
    if type(cmap_name) is list:
        if len(cmap_name) != n_machines:
            raise ValueError('Length of cmap does not match number of machines')
        for cmap_ in cmap_name[::-1]:
            cmap = plt.get_cmap(cmap_)
            bar_colors.insert(0, cmap(np.linspace(cmap_min_grad, cmap_max_grad, n_labels)))
    elif cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        bar_colors = cmap(np.linspace(cmap_min_grad, cmap_max_grad, n_labels))
    else:
        bar_colors = [colorblind_palette_dict[palette_order2[i%n_colors]] for i in range(n_labels)]

    # plot bars -- label by label
    x_values = np.arange(1, n_machines+1)
    max_value = max(max(values_lists))
    for i in range(n_labels):
        values = [val_list[i] for val_list in values_lists]
        if type(cmap_name) is list:
            for m_idx,value in enumerate(values):
                ax.bar(x_values[m_idx]+i*width, value, 0.95*width, align='center', color=bar_colors[m_idx][i])
        else:
            ax.bar(x_values+i*width, values, 0.95*width, align='center', color=bar_colors[i])
        
        for idx, v in enumerate(values):
            ax.text(idx+1+i*width, v+max_value/100, str("%.2f"%v), fontsize='large', horizontalalignment='center')
    
    # x-axis
    ax.set_xticks(x_values+(n_labels-1)*width/2)
    ax.set_xticklabels(machines)
    ax.set_xlabel(xlabel, fontsize='x-large')

    # y-axis
    if ymin is not None:
        ax.set_ylim(ymin=ymin)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)
    if yscale == 'log2':
        ax.set_yscale('log', basex=2)
    elif yscale == 'log':
        ax.set_yscale('log')
    ax.set_ylabel(ylabel, fontsize='x-large')
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize('large')

    # title
    if title is not None:
        ax.set_title(title, fontsize='xx-large')
    
    # legend
    if labels is not None:
        if n_labels == len(labels):
            legend = ax.legend(labels, loc=legend_loc, ncol=min(n_labels,4), 
                               shadow=False, fancybox=True,
                               title=legend_title, fontsize='large')
            plt.setp(legend.get_title(),fontsize='x-large')
        else:
            raise ValueError('Number of labels does not match number of lines')

    # save figure
    if saveas is None:
        pass
    elif saveas == 'show':
        plt.show()
    elif saveas == 'tikz':
        assert filename is not None, "No filename specified"
        tikz_save(filename + '.' + saveas, figureheight = figureheight, figurewidth = figurewidth)
    else:
        assert filename is not None, "No filename specified"
        plt.savefig(filename + '.' + saveas)
