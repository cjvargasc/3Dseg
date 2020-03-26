import numpy as np
import matplotlib.pyplot as plt


class Utils:

    @staticmethod
    def print_boundaries_stats(values):
        """ Prints the distribution of predicted boundary probability values """
        print("** Point cloud stats **")
        b_mask = values[:, 6] == 1
        nb_mask = values[:, 6] == 0

        b_av = np.average(values[b_mask, 7])
        b_max = values[b_mask, 7].max()
        b_min = values[b_mask, 7].min()

        nb_av = np.average(values[nb_mask, 7])
        nb_max = values[nb_mask, 7].max()
        nb_min = values[nb_mask, 7].min()

        print("prediction distribution for boundary pts:")
        print("average: ", b_av)
        print("max: ", b_max)
        print("min: ", b_min)
        print("")
        print("prediction distribution for non-boundary pts:")
        print("average: ", nb_av)
        print("max: ", nb_max)
        print("min: ", nb_min)
        print("")

    @staticmethod
    def calc_f1score(values, thresh):
        """ Calculates the f1-score for boundary probabilities given a threshold """

        pred_mask = values[:, 7] >= thresh
        gt_mask = values[:, 6] == 1

        # count TPs
        TPs = np.sum(values[pred_mask, 6] == 1)
        # count FPs
        FPs = np.sum(values[pred_mask, 6] == 0)
        # count FNs
        FNs = np.sum(values[gt_mask, 7] < thresh)

        if TPs + FPs == 0:
            precision = 0
        else:
            precision = TPs / float(TPs + FPs)

        recall = TPs / float(TPs + FNs)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * ((precision*recall)/float(precision+recall))

        return f1


    @staticmethod
    def f1score_thresh_plot(values):
        """ Plots the f1-score vs a range of predefined thresholds """
        thresholds = np.linspace(0, 1.0, num=20)
        f1scores = [Utils.calc_f1score(values, t) for t in thresholds]

        ax = plt.axes()
        ax.plot(thresholds, f1scores)
        plt.show()
