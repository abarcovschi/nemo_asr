import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator


def plot_conformertransducer():
    fig, ax = plt.subplots()
    plt.plot(ct_model_sizes, ct_wer_myst, marker='o', color='b', alpha=0.3, label="MyST_test")
    plt.plot(ct_model_sizes, ct_wer_cmu, marker='o', color='b', alpha=0.3, label="CMU_test")
    plt.plot(ct_model_sizes, ct_wer_pfs, marker='o', color='b', alpha=0.3, label="PFS_test")
    plt.plot(ct_model_sizes, ct_wer_avg_child, marker='o', color='b', label="avg child")

    plt.plot(ct_model_sizes, ct_wer_test_clean, color='r', alpha=0.3, label="test-clean")
    plt.plot(ct_model_sizes, ct_wer_test_other, color='orange', alpha=0.3, label="test-other")

    plt.annotate("MyST_test", (ct_model_sizes[-1], ct_wer_myst[-1]+0.3))
    plt.annotate("CMU_test", (ct_model_sizes[-1], ct_wer_cmu[-1]+0.3))
    plt.annotate("PFS_test", (ct_model_sizes[-1], ct_wer_pfs[-1]+0.3))
    plt.annotate("avg child", (ct_model_sizes[-1], ct_wer_avg_child[-1]+0.3))
    plt.annotate("test-clean", (ct_model_sizes[-1], ct_wer_test_clean[-1]+0.3))
    plt.annotate("test-other", (ct_model_sizes[-1], ct_wer_test_other[-1]+0.3))

    plt.xticks(ticks=ct_model_sizes, labels=ct_x_labels, rotation=45)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4)

    plt.title("Conformer-Transducer (non-finetuned) \nWER vs Number of parameters")
    plt.xlabel("Model parameters")
    plt.ylabel("WER (%)")

    plt.show(block=False)


def plot_conformertransducer_whisper_w2v2():
    fig, ax = plt.subplots()

    # conformer-transducer
    plt.plot(ct_model_sizes, ct_wer_myst, marker='o', color='b', alpha=0.3, label="MyST_test")
    plt.plot(ct_model_sizes, ct_wer_cmu, marker='o', color='b', alpha=0.3, label="CMU_test")
    plt.plot(ct_model_sizes, ct_wer_pfs, marker='o', color='b', alpha=0.3, label="PFS_test")
    plt.plot(ct_model_sizes, ct_wer_avg_child, marker='o', color='b', label="avg child")

    plt.annotate("MyST_test", (ct_model_sizes[-1]+7, ct_wer_myst[-1]-0.3), color='b')
    plt.annotate("CMU_test", (ct_model_sizes[-1]+7, ct_wer_cmu[-1]-0.3), color='b')
    plt.annotate("PFS_test", (ct_model_sizes[-1]+7, ct_wer_pfs[-1]-0.3), color='b')
    plt.annotate("avg child", (ct_model_sizes[-1]+7, ct_wer_avg_child[-1]-0.1), color='b')
    # whisper
    plt.plot(w_model_sizes, w_wer_myst, marker='o', color='r', alpha=0.3, label="MyST_test")
    plt.plot(w_model_sizes, w_wer_cmu, marker='o', color='r', alpha=0.3, label="CMU_test")
    plt.plot(w_model_sizes, w_wer_pfs, marker='o', color='r', alpha=0.3, label="PFS_test")
    plt.plot(w_model_sizes, w_wer_avg_child, marker='o', color='r', label="avg child")

    plt.annotate("MyST_test", (w_model_sizes[-1]+7, w_wer_myst[-1]-0.3), color='r')
    plt.annotate("CMU_test", (w_model_sizes[-1]+7, w_wer_cmu[-1]-0.3), color='r')
    plt.annotate("PFS_test", (w_model_sizes[-1]+7, w_wer_pfs[-1]-0.3), color='r')
    plt.annotate("avg child", (w_model_sizes[-1]+7, w_wer_avg_child[-1]-0.3), color='r')
    # w2v2
    plt.plot(w2v2_model_sizes, w2v2_wer_myst, marker='o', color='g', alpha=0.3, label="MyST_test")
    plt.plot(w2v2_model_sizes, w2v2_wer_cmu, marker='o', color='g', alpha=0.3, label="CMU_test")
    plt.plot(w2v2_model_sizes, w2v2_wer_pfs, marker='o', color='g', alpha=0.3, label="PFS_test")
    plt.plot(w2v2_model_sizes, w2v2_wer_avg_child, marker='o', color='g', label="avg child")

    plt.annotate("MyST_test", (w2v2_model_sizes[-1]+7, w2v2_wer_myst[-1]+0.3), color='g')
    plt.annotate("CMU_test", (w2v2_model_sizes[-1]+7, w2v2_wer_cmu[-1]-0.3), color='g')
    plt.annotate("PFS_test", (w2v2_model_sizes[-1]+7, w2v2_wer_pfs[-1]+0.7), color='g')
    plt.annotate("avg child", (w2v2_model_sizes[-1]+9, w2v2_wer_avg_child[-1]-0.3), color='g')

    labels_dict = dict()
    for label, value in zip(ct_x_labels, ct_model_sizes):
        labels_dict[label] = value
    for label, value in zip(w_x_labels, w_model_sizes):
        labels_dict[label] = value
    for label, value in zip(w2v2_x_labels, w2v2_model_sizes):
        labels_dict[label] = value
    sorted_tuples = [(k, v) for (k, v) in sorted(labels_dict.items(), key=lambda x: x[1])]
    x_labels, model_sizes = zip(*sorted_tuples)

    plt.xticks(ticks=model_sizes, labels=x_labels, rotation=45)

    xticks_colours = ['b', 'b', 'r', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'b', 'r']
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), xticks_colours):
        ticklabel.set_color(tickcolor)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4)

    plt.title("Non-finetuned models \nWER vs Number of parameters")
    plt.xlabel("Model parameters")
    plt.ylabel("WER (%)")

    red_patch = mpatches.Patch(color='red', label='Whisper En')
    blue_patch = mpatches.Patch(color='blue', label='Conformer-Transducer')
    green_patch = mpatches.Patch(color='green', label='wav2vec2')

    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.show(block=False)


if __name__ == "__main__":
    # GLOBALS
    # conformer-transducer
    ct_model_sizes = [14, 32, 120, 600, 1000]
    ct_x_labels = ["14M (S)", "32M (M)", "120M (L)", "600M (XL)", "1B (XXL)"]
    ct_wer_myst = np.array([21.34, 24.99, 25.91, 24.42, 18.55])
    ct_wer_cmu = np.array([12.68, 11.58, 8.94, 8.22, 8.38])
    ct_wer_pfs = np.array([16.05, 17.51, 15.06, 14.83, 12.06])
    ct_wer_avg_child = np.average(np.array([ct_wer_myst, ct_wer_cmu, ct_wer_pfs]), axis=0)
    ct_wer_test_clean = np.array([3.8, 3.54, 2.71, 2.56, 2.64]) # librispeech
    ct_wer_test_other = np.array([7.6, 6.8, 4.78, 4, 4.22]) # librispeech

    # whisper English-only trained models + Large-V2
    w_model_sizes = [39, 72, 244, 769, 1550]
    w_x_labels = ["39M (Tiny.en)", "72M (Base.en)", "244M (Small.en)", "769M (Medium.en)", "1.55B (Large-V2)"]
    w_wer_myst = np.array([33.02, 29.15, 26.72, 28.06, 25])
    w_wer_cmu = np.array([27.32, 20.75, 16.82, 14.00, 12.69])
    w_wer_pfs = np.array([47.11, 45.7, 39, 35.25, 73.68])
    w_wer_avg_child = np.average(np.array([w_wer_myst, w_wer_cmu, w_wer_pfs]), axis=0)

    # w2v2
    w2v2_model_sizes = [95, 317]
    w2v2_x_labels = ["95M (Base LS_960)", "317M (Large LL_60k)"]
    w2v2_wer_myst = np.array([15.41, 12.5])
    w2v2_wer_cmu = np.array([16.33, 14.85])
    w2v2_wer_pfs = np.array([11.2, 8.56])
    w2v2_wer_avg_child = np.average(np.array([w2v2_wer_myst, w2v2_wer_cmu, w2v2_wer_pfs]), axis=0)

    # ---------------------------------------- function calls ---------------------------------------------
    plot_conformertransducer()
    plot_conformertransducer_whisper_w2v2()
    
    plt.show() # needed as last line of script for non-blocking plotting of multiple plots