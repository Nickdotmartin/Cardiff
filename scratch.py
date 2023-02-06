import random
from operator import itemgetter
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def get_n_rows_n_cols(n_plots):
    """
    Function to get the optimal configuration of subplot (upto n=25).
    Ideally plots will be in a square arrangement, or in a rectangle.
    Start by adding columns to each row upto 4 columns, then add rows.

    :param n_plots: number of plots
    :return: n_rows, n_cols
    """

    if n_plots > 25:
        raise ValueError(f"\t\tToo many plots for this function: {n_plots}\n\n")

    # ideally have no more than 4 rows, unless more than 16 plots
    if n_plots <= 16:
        row_whole_divide = n_plots // 4  # how many times this number of plots goes into 4.
        row_remainder = n_plots % 4  # remainder after the above calculation.
        if row_remainder == 0:
            n_rows = row_whole_divide
        else:
            n_rows = row_whole_divide + 1
    else:
        n_rows = 5

    # ideally have no more than 4 cols, unless more than 20 plots
    col_whole_divide = n_plots // n_rows  # how many times this number of plots goes into n_rows.
    col_remainder = n_plots % n_rows  # remainder after the above calculation.
    if col_remainder == 0:
        n_cols = col_whole_divide
    else:
        n_cols = col_whole_divide + 1

    return n_rows, n_cols



n_sep_vals_list = list(range(26))

for n_sep_vals in n_sep_vals_list:

    n_plots = n_sep_vals + 1
    print(f"\nn_plots: {n_plots}")

    n_rows, n_cols = get_n_rows_n_cols(n_plots)
    print(f"n_plots: {n_plots}, n_rows: {n_rows}, n_cols: {n_cols}")

    # if n_plots > 25:
    #     raise ValueError(f"\t\tToo many plots for this function: {n_plots}\n\n")
    #
    #
    # # ideally have no more than 4 rows, unless more than 16 plots
    # if n_plots <= 16:
    #     row_whole_divide = n_plots // 4
    #     row_remainder = n_plots % 4
    #     # print(f"row_whole_divide: {row_whole_divide}, row_remainder: {row_remainder}")
    #     if row_remainder == 0:
    #         n_rows = row_whole_divide
    #     else:
    #         n_rows = row_whole_divide + 1
    # else:
    #     n_rows = 5
    #
    # # ideally have no more than 4 cols, unless more than 20 plots
    # col_whole_divide = n_plots//n_rows
    # col_remainder = n_plots % n_rows
    # # print(f"col_whole_divide: {col_whole_divide}, col_remainder: {col_remainder}")
    # if col_remainder == 0:
    #     n_cols = col_whole_divide
    # else:
    #     n_cols = col_whole_divide + 1
    # print(f"n_plots: {n_plots}, n_rows: {n_rows}, n_cols: {n_cols}, empty: {(n_rows*n_cols)-n_plots}")





