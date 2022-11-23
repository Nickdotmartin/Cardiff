import pandas as pd

test_df = pd.read_csv(r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\jitter_rgb\Nick\MASTER_ave_TM2_thresh.csv")
print(test_df)

def conc_to_first_isi_col(df):
    """
    Function to sort dataframe where concurrent is given as 'ISI_-1'.
    This can appear as the last ISI column instead of first.

    This simple function won't work if ISI columns aren't at the end,
    and will require a bit more sophistication
    (e.g., extract ISI columns first, then check if ISI_-1 is last)

    :param df: dataframe to be tested and sorted if necessary.
    :return: dataframe - which has been sorted if needed"""


    if df.columns.tolist()[-1] == 'ISI_-1':
        col_list = df.columns.tolist()
        other_cols = [i for i in col_list if 'ISI' not in i]
        isi_cols = [i for i in col_list if 'ISI' in i]
        new_cols_list = other_cols + isi_cols[-1:] + isi_cols[:-1]
        out_df = df.reindex(columns=new_cols_list)
        print(f"Concurrent column moved to start\n{out_df}")
    else:
        out_df = df

    return out_df


