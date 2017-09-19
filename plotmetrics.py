# coding=utf-8
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from StringIO import StringIO
import itertools
from sklearn.metrics import brier_score_loss

def plot_auc(gold_labels, pred_scores, kind='ROC', classes_to_evaluate=[1], return_as='TF_image_summary'):
    fig = plt.figure(figsize=(10.5, 7))
    ax = plt.subplot(111)
    lw = 2
    colors = color_pallete_iter()
    greys = itertools.cycle(matplotlib.cm.gray(np.linspace(0, 6, 12)))
    for clazz in classes_to_evaluate:
        if kind == 'ROC':
            x, y, thresholds = metrics.roc_curve(gold_labels, pred_scores, pos_label=clazz)
            x_label = 'False Positive Rate / threshold (t)'
            y_label = 'True Positive Rate'
            auc = metrics.auc(x, y)
        elif kind == 'PR':
            y, x, thresholds = metrics.precision_recall_curve(gold_labels, pred_scores, pos_label=clazz)
            # threshold array is one shorter (append to match recall)
            thresholds = np.append(thresholds, [thresholds[-1] + 1e-6])
            x_label = 'recall / threshold (t)'
            y_label = 'precision'
            auc = metrics.average_precision_score(gold_labels, pred_scores)
            max_f1, max_f1_threshold, recall_at_max_f1, f1s = max_F1(precisions=y, recalls=x, thresholds=thresholds)
            # max F1
            best_f1_color = colors.next()
            plt.plot([recall_at_max_f1, recall_at_max_f1], [0, 1], color=best_f1_color, linestyle=':',  lw=lw)
            plt.plot(recall_at_max_f1, max_f1, color=best_f1_color, marker='s', lw=lw,
                     label='F1_max=%0.3f for c=%i at t=%0.3f' % (max_f1, clazz, max_f1_threshold))
            # F1 curve
            plt.plot(x, f1s, color=colors.next(), linestyle='-.', lw=lw, label='F1 over T')
        else:
            raise Exception("Curve plot type:", kind, 'not implemented')
        plt.plot(x, y, color=colors.next(), lw=lw, label='%s curve c=%i (AUC = %0.3f)' % (kind, clazz, auc))
        plt.plot(x, thresholds, color=greys.next(), lw=lw, linestyle=':', label='T for c=%i' % clazz)
    if kind == 'ROC':
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='random guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(kind + ' curve over classes (c)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.66, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12)
    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    fig.canvas.draw()
    if return_as == 'fig':
        return fig
    elif return_as == 'TF_image_summary':
        height, width, img_as_string = get_img_as_np_array_and_as_string(fig)
        plt.close(fig)  # not needed anymore
        return height, width, img_as_string

def brier_score_classifier_probability_miscalibration(y_true, y_prob, classes_to_evaluate=[1]):
    """ BS = 0 means probabilities match labels, BS=1 means probabilites are the oposite of the labels.
    :param y_true: gold labels
    :param y_prob: predicted probabilities
    :return: brier score. Also interpretable as "miscalibration of classifier" when used with other measures. Basically
    the difference between a discrete objective (classification) and a mean squared error etc. (regression)
    """
    bs = []
    for clazz in classes_to_evaluate:
        bs.append(brier_score_loss(y_true, y_prob, pos_label=clazz))
    return bs if len(bs) > 1 else bs[0]


def max_F1(precisions, recalls, thresholds):
    ''' Given data from metrics.precision_recall_curve(gold_labels, pred_scores,..) compute F1 curve values '''
    f1s = []
    max_f1 = 0
    max_f1_threshold = 0
    recall_at_max_f1 = 0
    for r, p, t in zip(recalls, precisions, thresholds):
        f1 = np.nan_to_num([2 * (p * r) / (p + r)])[0]  # NaN guard. NaN -> 0
        f1s.append(f1)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_threshold = t
            recall_at_max_f1 = r
    return max_f1, max_f1_threshold, recall_at_max_f1, f1s

def plot_matthews_corrcoef(gold_labels, pred_scores, kind='ROC', classes_to_evaluate=[1], return_as='TF_image_summary'):
    """
    Plot the matthews correlation scalar (binary and multi-class(1 hot encoded with 1 = argmax(sample probabilites)))
    :param gold_labels:
    :param pred_scores:
    :param kind:
    :param classes_to_evaluate:
    :param return_as:
    :return:
    """
    # TODO: 1. sklearn.metrics.matthews_corrcoef(y_true, y_pred, sample_weight=None)
    # tf.summary.scalar via NonTensorTensorboardLogger.log_scalar
    raise Exception("Implement Matthews correlation curve")


def find_max_F1(self, y_true_bin, y_score, pos_label):
    """
    Since class imbalance can move the class separation probability threshold away form 0.5 this finds a better
    threshold
    :param y_true_bin:
    :param y_score:
    :return:
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y_true=y_true_bin, probas_pred=y_score)
    aucpr  = metrics.auc(recall, precision)
    print "AUC PR:", aucpr
    max_f1 = 0
    max_f1_threshold = 0
    for r, p, t in zip(recall, precision, thresholds):
        if p + r == 0: continue
        y_pred = (y_score > t).astype(int)
        f1 = metrics.f1_score(y_true=y_true_bin, y_pred=y_pred, average='binary', pos_label=pos_label)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_threshold = t
    print "Max F1 C" + str(pos_label) + ":", max_f1, "prob threshold", max_f1_threshold
    return max_f1, max_f1_threshold, (y_score > max_f1_threshold).astype(int)

def get_img_as_np_array_and_as_string(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Write the image to a string
    img_as_string = StringIO()
    plt.imsave(img_as_string, img, format='png')
    height = img.shape[0]
    width = img.shape[1]
    return height, width, img_as_string

def color_pallete_iter_MPL_gray(r1=0, r2=6, r3=12):
    """ Mostly for documentation of how to use Matplotlib color palletes"""
    itertools.cycle(matplotlib.cm.gray(np.linspace(r1, r2, r3)))

def color_pallete_iter():
    # 100 destinct colors for plotting
    crayons = {'Almond': '#EFDECD',
       'Antique Brass': '#CD9575',
       'Apricot': '#FDD9B5',
       'Aquamarine': '#78DBE2',
       'Asparagus': '#87A96B',
       'Atomic Tangerine': '#FFA474',
       'Banana Mania': '#FAE7B5',
       'Beaver': '#9F8170',
       'Bittersweet': '#FD7C6E',
       'Black': '#000000',
       'Blue': '#1F75FE',
       'Blue Bell': '#A2A2D0',
       'Blue Green': '#0D98BA',
       'Blue Violet': '#7366BD',
       'Blush': '#DE5D83',
       'Brick Red': '#CB4154',
       'Brown': '#B4674D',
       'Burnt Orange': '#FF7F49',
       'Burnt Sienna': '#EA7E5D',
       'Cadet Blue': '#B0B7C6',
       'Canary': '#FFFF99',
       'Caribbean Green': '#00CC99',
       'Carnation Pink': '#FFAACC',
       'Cerise': '#DD4492',
       'Cerulean': '#1DACD6',
       'Chestnut': '#BC5D58',
       'Copper': '#DD9475',
       'Cornflower': '#9ACEEB',
       'Cotton Candy': '#FFBCD9',
       'Dandelion': '#FDDB6D',
       'Denim': '#2B6CC4',
       'Desert Sand': '#EFCDB8',
       'Eggplant': '#6E5160',
       'Electric Lime': '#CEFF1D',
       'Fern': '#71BC78',
       'Forest Green': '#6DAE81',
       'Fuchsia': '#C364C5',
       'Fuzzy Wuzzy': '#CC6666',
       'Gold': '#E7C697',
       'Goldenrod': '#FCD975',
       'Granny Smith Apple': '#A8E4A0',
       'Gray': '#95918C',
       'Green': '#1CAC78',
       'Green Yellow': '#F0E891',
       'Hot Magenta': '#FF1DCE',
       'Inchworm': '#B2EC5D',
       'Indigo': '#5D76CB',
       'Jazzberry Jam': '#CA3767',
       'Jungle Green': '#3BB08F',
       'Laser Lemon': '#FEFE22',
       'Lavender': '#FCB4D5',
       'Macaroni and Cheese': '#FFBD88',
       'Magenta': '#F664AF',
       'Mahogany': '#CD4A4C',
       'Manatee': '#979AAA',
       'Mango Tango': '#FF8243',
       'Maroon': '#C8385A',
       'Mauvelous': '#EF98AA',
       'Melon': '#FDBCB4',
       'Midnight Blue': '#1A4876',
       'Mountain Meadow': '#30BA8F',
       'Navy Blue': '#1974D2',
       'Neon Carrot': '#FFA343',
       'Olive Green': '#BAB86C',
       'Orange': '#FF7538',
       'Orchid': '#E6A8D7',
       'Outer Space': '#414A4C',
       'Outrageous Orange': '#FF6E4A',
       'Pacific Blue': '#1CA9C9',
       'Peach': '#FFCFAB',
       'Periwinkle': '#C5D0E6',
       'Piggy Pink': '#FDDDE6',
       'Pine Green': '#158078',
       'Pink Flamingo': '#FC74FD',
       'Pink Sherbert': '#F78FA7',
       'Plum': '#8E4585',
       'Purple Heart': '#7442C8',
       "Purple Mountains' Majesty": '#9D81BA',
       'Purple Pizzazz': '#FE4EDA',
       'Radical Red': '#FF496C',
       'Raw Sienna': '#D68A59',
       'Razzle Dazzle Rose': '#FF48D0',
       'Razzmatazz': '#E3256B',
       'Red': '#EE204D',
       'Red Orange': '#FF5349',
       'Red Violet': '#C0448F',
       "Robin's Egg Blue": '#1FCECB',
       'Royal Purple': '#7851A9',
       'Salmon': '#FF9BAA',
       'Scarlet': '#FC2847',
       "Screamin' Green": '#76FF7A',
       'Sea Green': '#93DFB8',
       'Sepia': '#A5694F',
       'Shadow': '#8A795D',
       'Shamrock': '#45CEA2',
       'Shocking Pink': '#FB7EFD',
       'Silver': '#CDC5C2',
       'Sky Blue': '#80DAEB',
       'Spring Green': '#ECEABE',
       'Sunglow': '#FFCF48'}
    abc_colors = [crayons[color] for color in sorted(crayons.keys())]
    colors = itertools.cycle(abc_colors)
    return colors