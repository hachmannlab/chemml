import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import re
import seaborn as sns

def plot_lime(rel_df, max_features):
    '''
    Parameters
    ----------
    rel_df: pandas dataframe
        global or local relevance scores; use un-modified relevance dataframes returned from LRP, DeepSHAP, or LIME methods
    
    file_path: str
        path to save the figure
    
    max_features: int
        no. of most impactful features to show in the plot
    '''
    f, ax = plt.subplots(figsize=(15,10))
    vals = list(rel_df['local_relevance'][:max_features].values)
    labels = list(rel_df.labels[:max_features])
    colors = ['green' if i>0 else 'red' for i in vals]
            
    pos = np.arange(len(vals)) + .5
    ax.barh(pos, vals, align='center', color=colors)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels,size=20)

    xlab = [np.round(i,2) for i in np.arange(rel_df['local_relevance'].min(), rel_df['local_relevance'].max()+0.1,0.1)]

    ax.set_xticks(xlab)
    ax.set_xticklabels([str(i) for i in xlab],size=20)
    ax.set_xlabel('Relevance Score',size=30)
    ax.set_ylabel('Feature and its Range', size=30)
    plt.tight_layout()
    return f
    f.savefig(file_path,bbox='tight',dpi=330)

def plot_lrp(rel_df, max_features):

    rel_df = rel_df[:max_features]
    val = max(abs(rel_df[rel_df.columns[0]].min()),rel_df[rel_df.columns[0]].max())

    f = plt.figure(figsize=(25,6))
    ax = sns.heatmap(rel_df.transpose(),annot=False, cmap='coolwarm',linecolor='k',linewidths=0.02,vmin=-val,vmax=val)
    ax.set_xticklabels(list(rel_df.index),fontsize=20)
    plt.tight_layout()

    return f
    plt.savefig(file_path,bbox_inches='tight',dpi=330)

def _format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s

def plot_shap_local(expected_value, shap_values=None, features=None, feature_names=None, max_display=10):

    # support passing an explanation object
    upper_bounds = None
    lower_bounds = None
    if str(type(expected_value)).endswith("Explanation'>"):
        shap_exp = expected_value
        expected_value = shap_exp.expected_value
        shap_values = shap_exp.values
        features = shap_exp.data
        feature_names = shap_exp.feature_names
        lower_bounds = getattr(shap_exp, "lower_bounds", None)
        upper_bounds = getattr(shap_exp, "upper_bounds", None)

    # make sure we only have a single output to explain
    if (type(expected_value) == np.ndarray and len(expected_value) > 0) or type(expected_value) == list:
        raise Exception("waterfall_plot requires a scalar expected_value of the model output as the first "
                        "parameter, but you have passed an array as the first parameter! "
                        "Try shap.waterfall_plot(explainer.expected_value[0], shap_values[0], X[0]) or "
                        "for multi-output models try "
                        "shap.waterfall_plot(explainer.expected_value[0], shap_values[0][0], X[0]).")

    # make sure we only have a single explanation to plot
    if len(shap_values.shape) == 2:
        raise Exception(
            "The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!")

    # unwrap pandas series
    if not isinstance(features, np.ndarray):
        features = features.values
    
    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(shap_values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(shap_values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = expected_value + shap_values.sum()
    yticklabels = ["" for i in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    f, ax = plt.subplots(figsize=(14,10))
    # plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(shap_values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = shap_values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            ax.plot([loc, loc], [rng[i] - 1 - 0.4, rng[i] + 0.4],
                    color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = _format_value(features[order[i]], "%0.03f") + " = " + feature_names[order[i]]

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(shap_values):
        yticklabels[0] = "%d other features" % (len(shap_values) - num_features + 1)
        remaining_impact = expected_value - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = '#CD5C5C'
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = '#5c5ccd'

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + \
        list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    ax.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw,
            left=np.array(pos_lefts) - 0.01*dataw, color='#CD5C5C', alpha=0)
    label_padding = np.array([-0.1*dataw if -w < 1 else 0 for w in neg_widths])
    ax.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw,
            left=np.array(neg_lefts) + 0.01*dataw, color='#5c5ccd', alpha=0)

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    # fig = ax.gcf()
    # ax = plt.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = f.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color='#CD5C5C', width=bar_width,
            head_width=bar_width
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor='#FA8072'
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], _format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=20
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], _format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color='#CD5C5C',
                fontsize=20
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color='#5c5ccd', width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor='#5ca9cd'
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], _format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=20
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], _format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color='#5c5ccd',
                fontsize=20
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    ax.set_yticks(list(range(num_features)) + list(np.arange(num_features)+1e-8))
    ax.set_yticklabels(yticklabels[:-1] +
            [l.split('=')[-1] for l in yticklabels[:-1]],fontsize=20)
    # ax.set_yticklabels(yticklabels[:-1],)

    # put horizontal lines for each feature row
    for i in range(num_features):
        ax.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    ax.axvline(expected_value, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = expected_value + shap_values.sum()
    ax.axvline(fx, 0, 1, color="#cccccc", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    # ax.set_tick_params(direction='bottom')
    # ax.set_tick_params(direction='none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=13)
    #plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([expected_value-0.01, expected_value+0.01])
    ax2.set_xticklabels(["\n$E[f(X)$]", "\n = "+_format_value(expected_value, "%0.03f")], fontsize=20, ha="left")
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    ax2.spines['left'].set_visible(True)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks([expected_value + shap_values.sum() - 0.01, expected_value + shap_values.sum() + 0.01])
    ax3.set_xticklabels([r"$f(x)$", "$ = "+_format_value(fx, "%0.03f")+"$"], fontsize=20, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-10/72., 0, f.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(12/72., 0, f.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-20/72., 0, f.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(22/72., -1/72., f.dpi_scale_trans))
    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    plt.tight_layout()
    return f
    f.savefig(path_to_file,dpi=330)

def plot_shap_global(data, shap_values,max_display,feature_names):
    '''
    data = actual feature values
    shap_values = relevance scores (shapley values)
    max_display = no. of features u want to see
    
    '''
    feature_order = np.argsort(np.mean(np.abs(shap_values), axis=0))
    # print(feature_order)
    feature_order = feature_order[-min(max_display,len(list(feature_order))):]
    row_height = 0.4
    # plt.gcf().set_size_inches(8,len(feature_order)* row_height+1.5)
    f, ax = plt.subplots(figsize=(20,15))
    # plt.figure(figsize=(15,10))
    ax.axvline(x=0, color="#999999", zorder=-1)
    features = data
    for pos, i in enumerate(feature_order):
        ax.axhline(y=pos,color='#cccccc',lw=0.5,dashes=(1,5),zorder=-1)
        shaps = shap_values[:,i]
        values = features[:,i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        values = np.array(values, dtype=np.float64)
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer=0
            ys[ind] = np.ceil(layer/2) * ((layer % 2) * 2 -1)
            layer += 1
            last_bin = quant[ind]

        ys *= 0.9 * (row_height / np.max(ys + 1))
        vmin = np.nanpercentile(values, 5)
        vmax = np.nanpercentile(values, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(values, 1)
            vmax = np.nanpercentile(values, 99)
            if vmin == vmax:
                vmin = np.min(values)
                vmax = np.max(values)
        if vmin > vmax:
            vmin = vmax
        
        nan_mask = np.isnan(values)
        ax.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#5c5ccd", vmin=vmin, vmax=vmax, s=16, alpha=1, linewidth=0, zorder=3) #, rasterized=len(shaps)>500) 
        cvals = values[np.invert(nan_mask)].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (vmin+vmax)/2
        cvals[cvals_imp>vmax] = vmax
        cvals[cvals_imp<vmin] = vmin
        ax.scatter(shaps[np.invert(nan_mask)], pos+ys[np.invert(nan_mask)], vmin=vmin, vmax=vmax, s=16, c=cvals, alpha=1, linewidth=0, zorder=3) #, rasterized=len(shaps)>500)
    # print(cvals)

    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap = plt.get_cmap())
    m.set_array([0, 1])
    cb = f.colorbar(m, ticks=[0, 1])
    cb.set_ticklabels(['Low', 'High'],size=18)
    cb.set_label('Feature Value', size=20, labelpad=0)
    cb.ax.tick_params(labelsize=22, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 8) * 20)

    # ax.set_xticks_position('bottom')
    # ax.set_yticks_position('none')
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(color='k')
    ax.set_yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=25)
    ax.tick_params('y', length=20, width=0.5, which='major')
    ax.tick_params('x', labelsize=18)
    ax.set_ylim(-1, len(feature_order))
    ax.set_xlabel('SHAP Values', fontsize=20)
    ax.set_ylabel('Feature Names',fontsize=20)
    plt.tight_layout()
    return f
    f.savefig(path_to_file,dpi=330)
    