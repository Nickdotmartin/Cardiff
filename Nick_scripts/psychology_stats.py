import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.stats as stats
import statsmodels.api as sm


'''
This page contains code for commonly usd psychology stats such as t-tests, ANOVA and correlations, with results outputted in APA style.
'''

def p_val_sig_stars(p_val):
    """
    This function takes a p-value and returns a string with the number of stars to add to the APA style report.
    :param p_val: p-value
    :return: string of stars
    """

    # add stars to p value
    if p_val < .001:
        report_p = 'p < .001***'
    elif p_val < .01:
        report_p = 'p < .01**'
    elif p_val < .05:
        report_p = 'p < .05*'
    else:
        report_p = f"p = n/s"

    return report_p


def t_test(array1, array2, equal_var=True):
    """
    Performs a Student's t-test on two NumPy arrays if equal variance, or Welsh's test if not equal variance.
    :param array1: 1d numpy array
    :param array2: 1d numpy array
    :return: results dictionary
    """

    # Run the t-test
    t, p = stats.ttest_ind(array1, array2, equal_var=equal_var)

    # get degrees of freedom
    df = len(array1) + len(array2) - 2

    # add stars to p value
    report_p = p_val_sig_stars(p)

    if equal_var:  # student t-test
        # get mean for each array
        mean1 = np.mean(array1)
        mean2 = np.mean(array2)

        # get the standard deviation of each array
        std1 = np.std(array1)
        std2 = np.std(array2)

        # Print the results in APA style
        text_report = (f"Student's t-test: group1 M={mean1:.2f} (SD={std1:.2f}), group2 M={mean2:.2f} (SD={std2:.2f}); "
                       f"t(df={df:.2f}) = {t:.2f}, {report_p}")


        return {'mean1': mean1, 'std1': std1, 'mean2': mean2, 'std2': std2,
                'test': t, 'p': p, 'text_report': text_report}

    else:  # Welsh's t-test
        # get median for each array
        median1 = np.median(array1)
        median2 = np.median(array2)

        text_report = (f"Welsh's t-test: group1 median={median1:.2f}, group2 median={median2:.2f}; "
                       f"t(df={df:.2f}) = {t:.2f}, {report_p}")

        return {'test': t, 'p': p, 'median1': median1, 'median2': median2, 'text_report': text_report}


def mann_whitney_u(array1, array2):
    """
    Non-parametric test for comparing two groups (AKA: Wilcoxon rank-sum test).
    :param array1: 1d numpy array
    :param array2: 1d numpy array
    :return: results dictionary
    """

    # run the test
    u, p = stats.mannwhitneyu(array1, array2)

    # get median for each array
    median1 = np.median(array1)
    median2 = np.median(array2)

    # add stars to p value
    report_p = p_val_sig_stars(p)

    text_report = f"median1 = {median1:.2f}, median2 = {median2:.2f}; U = {u:.2f}, {report_p}"

    return {'test': u, 'p': p, 'median1': median1, 'median2': median2, 'text_report': text_report}


def compare_groups(array1, array2):
    """
    This script will run the correct test (Student, Welsh or Mann_whitney U) on groups
    depending on size, normality and equality of variance.
    It returns a dictionary with all the info I need to report it.

    :param array1: 1d numpy array
    :param array2: 1d numpy array
    :return: dict of stats
    """

    # create empty dictionary to store results
    output_dict = {}

    # get size of arrays
    n_1 = len(array1)
    n_2 = len(array2)

    # check that the groups are roughly equal in size (e.g., the ratio is less than 2:1)
    if n_1 / n_2 > 2 or n_2 / n_1 > 2:
        # run Mann-Whitney U test
        test_results = mann_whitney_u(array1, array2)

        # update dictionary and return results
        output_dict['equal_size'] = False
        output_dict['use_test'] = "Mann_Whitney_U"
        output_dict['test_reason'] = 'unequal sizes'
        output_dict.update(test_results)
        return output_dict

    # Datasets are roughly equal in size
    output_dict['equal_size'] = True

    # Check for normality of data for both samples.
    # If number of samples is < 50. use a Shapiro-Wilk test, else use stats.normaltest (D’Agostino and Pearson’s test).
    array_1_normal = True
    array_2_normal = True
    if np.mean([n_1, n_2]) < 50:
        output_dict['normal_test'] = "shapiro (< 50 samples)"
        norm_1_stat, norm_1_p = stats.shapiro(array1)
        norm_2_stat, norm_2_p = stats.shapiro(array2)
    else:
        output_dict['normal_test'] = "D’Agostino&Pearson (> 50 samples)"
        norm_1_stat, norm_1_p = stats.normaltest(array1)
        norm_2_stat, norm_2_p = stats.normaltest(array2)

    if norm_1_p < 0.05:
        array_1_normal = False
    if norm_2_p < 0.05:
        array_2_normal = False
    output_dict['array_1_normal'] = array_1_normal
    output_dict['array_2_normal'] = array_2_normal

    # Check if both samples are normally distributed.
    if array_1_normal and array_2_normal:

        # Both samples are normally distributed.
        # Check for equal variance using the Levene test.
        levene_statistic, lavene_p = stats.levene(array1, array2)

        if lavene_p > 0.05:
            # variances are equal, run Student's t-test.
            test_results = t_test(array1, array2, equal_var=True)
            output_dict.update(test_results)

            # update dictionary and return results
            output_dict['equal_variance'] = True
            output_dict['use_test'] = "Student_t"
            output_dict['test_reason'] = 'equal sizes, both normal, equal variance'
            return output_dict

        else:
            # variances are not equal, run Welch t-test.
            test_results = t_test(array1, array2, equal_var=False)

            # update dictionary and return results
            output_dict['equal_variance'] = False
            output_dict['use_test'] = "Welch_t"
            output_dict['test_reason'] = 'equal sizes, both normal, but unequal variance'
            output_dict.update(test_results)
            return output_dict

    elif array_1_normal or array_2_normal:
        # Only one sample is not normally distributed, run Welch t-test can be used.
        test_results = t_test(array1, array2, equal_var=False)

        # update dictionary and return results
        output_dict['use_test'] = "Welch_t"
        output_dict['test_reason'] = 'equal sizes, only one dataset was normal'
        output_dict.update(test_results)
        return output_dict

    else:
        # Neither sample is normally distributed, run Mann-Whitney U test.
        test_results = mann_whitney_u(array1, array2)

        # update dictionary and return results
        output_dict['use_test'] = "Mann_Whitney_U"
        output_dict['test_reason'] = 'equal sizes, not normal'
        output_dict.update(test_results)
        return output_dict


def linear_reg_OLS(data, DV, predictors):
    """
    This function will run a multiple linear regression using statsmodels OLS.

    :param data: dataframe containing the variables, can include extra columns
    :param DV: column_name being predicted
    :param predictors: Column(s) thought to influence DV, can be a list of column_names or a single column_name
    :return: results_dict: nested dictionary of results
    """

    IV = predictors

    # create a new dataframe with only the variables I need, as a copy not a slice
    reg_df = data[IV + [DV]].copy()
    print(f"reg_df: {reg_df.shape}\ncolumns: {list(reg_df.columns)}\n{reg_df.head()}")

    '''
    check for any columns containing strings.
    First try to convert them using pd.to_numeric, if this fails, then they are probably strings.
    In this case, make a dictionary of the unique values in the column, and then replace the strings with numbers.
    '''
    string_cols_dict = {}
    converted_cols = False
    for this_col in reg_df.columns:
        try:
            reg_df[this_col] = pd.to_numeric(reg_df[this_col])
        except ValueError:
            # print(f"this_col: {this_col}")
            unique_vals = sorted(reg_df[this_col].unique())
            # print(f"unique_vals: {unique_vals}")
            unique_vals_dict = {}
            for i, this_val in enumerate(unique_vals):
                unique_vals_dict[this_val] = i
            # print(f"unique_vals_dict: {unique_vals_dict}")
            reg_df[this_col] = reg_df[this_col].replace(unique_vals_dict)
            string_cols_dict[this_col] = unique_vals_dict
            converted_cols = True

    if converted_cols:
        print(f"string_cols_dict: {string_cols_dict}")


    # next I will visualise the data
    for this_IV in IV:
        # sns.scatterplot(x=this_IV, y=DV, data=reg_df)
        sns.regplot(x=this_IV, y=DV, data=reg_df)
        if converted_cols:
            if this_IV in string_cols_dict.keys():
                plt.xticks(list(string_cols_dict[this_IV].values()), list(string_cols_dict[this_IV].keys()))
        # get regression r and p
        r, p = stats.pearsonr(reg_df[this_IV], reg_df[DV])
        plt.suptitle(f"{this_IV} vs {DV}")
        plt.title(f"r = {r:.2f}, p = {p:.2f}")
        plt.show()



    formula_text = f"{DV} ~ "
    for this_IV in IV:
        if this_IV == IV[-1]:
            formula_text += f"{this_IV}"
        else:
            formula_text += f"{this_IV} + "
    print(f"\nformula_text: {formula_text}")

    # fit the model
    model = sm.formula.ols(formula=formula_text, data=reg_df)
    results = model.fit()
    print(f"results.summary():\n{results.summary()}")

    '''
    Create a nested dictionary of the results, with model level results at the top level, and then the results for each
    IV (and intercept) at the second level.
    All values should be shown to 3 decimal places (not scientific notation).
    '''
    results_dict = {}
    results_dict['model'] = {}
    results_dict['model']['Model'] = 'OLS'
    results_dict['model']['Method'] = 'Least Squares'
    results_dict['model']['Dep_variable'] = DV
    results_dict['model']['predictors'] = IV
    results_dict['model']['No. Obs'] = results.nobs
    results_dict['model']['Df_Residuals'] = results.df_resid
    results_dict['model']['Df_Model'] = results.df_model
    results_dict['model']['Covariance_Type'] = results.cov_type
    results_dict['model']['rsquared'] = results.rsquared
    results_dict['model']['rsquared_adj'] = results.rsquared_adj
    results_dict['model']['F-statistic'] = results.fvalue
    results_dict['model']['F-statistic_pvalue'] = results.f_pvalue
    results_dict['model']['Log-Likelihood'] = results.llf
    results_dict['model']['AIC'] = results.aic
    results_dict['model']['BIC'] = results.bic

    '''null hypothesis is that there is no relationship between IV and DV.
    If p > 0.05, then fail to reject the null hypothesis, and conclude that there is no relationship between IV and DV.
    If p < 0.05, then reject the null hypothesis, and conclude that there is a relationship between IV and DV.
    If coef > 0, then as IV increases, DV increases.
    if abs(coef) < 1, then the relationship is weak.
    
    '''


    # add IVs and Intercept to results_dict
    for this_IV in IV + ['Intercept']:
        results_dict[this_IV] = {}
        results_dict[this_IV]['pvalues'] = results.pvalues[this_IV]
        results_dict[this_IV]['p_sig_stars'] = p_val_sig_stars(results.pvalues[this_IV])
        results_dict[this_IV]['coef'] = results.params[this_IV]
        results_dict[this_IV]['std_err'] = results.bse[this_IV]
        results_dict[this_IV]['t'] = results.tvalues[this_IV]

    # print_nested_round_floats(results_dict, dict_title='results_dict')

    return results_dict



# '''
# This following section is for me to try multi-level modelling.
# '''
# # todo: get rid of shapiro and levene functions, just have lines of code in the MLM function
# # todo: add column_names for the IV levels as variables to the function,
# # todo: add DV column column name as a variable.
# # todo: write a function to make a master list to feed into the MLM.  that is, if I pass the
# #  paths to the different participants, loop through and add any missing columns to their individual CSVs
# #  then append this to the master MLM list.
#
# def MLM(data):
#     """
#     Performs a multilevel modeling (MLM) on the given data.
#
#     Args:
#     data: The dataset to be analyzed.
#
#     Returns:
#     The results of the MLM.
#     """
#
#     # Check the normality assumption.
#     normality_test = shapiro_wilk(data['threshold'])
#
#     # Check the equal variances assumption.
#     equal_variances_test = levene(data['threshold'])
#
#     # If the normality assumption is not met, use a robust estimator.
#     if normality_test.pvalue < 0.05:
#         estimator = "robust"
#     else:
#         estimator = "normal"
#
#     # If the equal variances assumption is not met, use the Huber-White estimator.
#     if equal_variances_test.pvalue < 0.05:
#         var_type = "robust"
#     else:
#         var_type = "unequal"
#
#     # Fit the MLM model.
#     model = sm.mixed_model.MixedLM(data["threshold"], data["prelim"], data["congruent"], data["separation"], data["ISI"], data["monitor"], estimator=estimator, var_type=var_type)
#     results = model.fit()
#
#     # Return the results of the MLM.
#     return results
#
#
# def shapiro_wilk(data):
#       """
#       Performs a Shapiro-Wilk test on the given data.
#
#       Args:
#         data: The data to be tested.
#
#       Returns:
#         The results of the Shapiro-Wilk test.
#       """
#
#       test = stats.shapiro(data)
#
#       return test
#
# def levene(data):
#       """
#       Performs a Levene test on the given data.
#
#       Args:
#         data: The data to be tested.
#
#       Returns:
#         The results of the Levene test.
#       """
#
#       test = stats.levene(data)
#
#       return test
#
#
# def mediation_analysis(df, IV, DV, mediator_variables_list):
#     """
#     This function performs a mediation analysis to identify the factors that mediate the impact of IV on DV.
#
#     Args:
#     df: The Pandas dataframe containing the data.
#     IV: The independent variable.
#     DV: The dependent variable.
#     mediator_variables_list: A list of the mediator variables.
#
#     Returns:
#     A dictionary containing the results of the mediation analysis.
#     """
#
#     # Import the necessary modules.
#     import statsmodels.api as sm
#     from statsmodels.stats.mediation import Mediation
#
#     # Create the regression models.
#     direct_model = sm.OLS(DV, IV).fit()
#     mediator_models = {}
#     for mediator in mediator_variables_list:
#         mediator_models[mediator] = sm.OLS(DV, IV + mediator).fit()
#
#     # Run the mediation analysis.
#     mediation = Mediation(
#       direct_model, mediator_models, dependent_var=DV, verbose=False
#     )
#     results = mediation.fit()
#
#     # Return the results of the mediation analysis.
#     return results

# master_csv_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\MLM_master.csv"
# master_csv = pd.read_csv(master_csv_path)
# print(f"master_csv: {master_csv.shape}\n{master_csv.head()}")
#
# mediation_results = mediation_analysis(master_csv['prelim'], master_csv['threshold'], master_csv['bg_type'], master_csv['flow_speed_prop'], master_csv['mask_type'], master_csv['monitor'])
# print(f"mediation_results: {mediation_results.summary()}")


# '''
# This bit will loop through various exp folders, and participant names, and look for a MASTER_all_Prelim.csv file.
# It will open the file, add any relevant columns and append it to a master list.
# '''
# MLM_master_list = []
#
# # set paths
# root_path = r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23"
#
# p_master_path_list = [
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_6_rings\Nick_act_new_dots_thrd_spd_17082023\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_6_rings\Nick_actual_new_dots_17082023\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_6_rings\Nick_half_ring_spd_16082023\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_6_rings\Nick_orig_dots_17082023\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_6_rings\Nick_third_ring_spd_16082023\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_6_rings\Nick_quarter_ring_spd_16082023\MASTER_all_Prelim.csv",
#
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\OLED_circles_rings_quartSpd\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\Nick_OLED_dots_normSpd_22082023\MASTER_all_Prelim.csv",
#
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\240_dots_spokes_23082023\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\240_new_dots_spokes_31082023\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\240_rings_halfSpd_spokes\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\240_rings_halfSpd_spokes\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\asus_cal_circles_rings_HalfSpd\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\asus_cal_circles_rings_quartSpd\MASTER_all_Prelim.csv",
#     r"C:\Users\sapnm4\OneDrive - Cardiff University\PycharmProjects\Cardiff\rad_flow_Sept23\rad_flow_7_spokes\OLED_circles_rings_quartSpd_v2\MASTER_all_Prelim.csv",
# ]
#
#
# # loop though each participant
# for idx, p_master_path in enumerate(p_master_path_list):
#
#     path, csv_name = os.path.split(p_master_path)
#     path, p_name = os.path.split(path)
#     path, script_name = os.path.split(path)
#     print(f"\n\n{idx}. p_name: {p_name}, script_name: {script_name}")
#
#     # find substring 'dots' or 'rings' in p_name
#     if 'orig_dot' in p_name:
#         bg_type = 'orig_dots'
#     elif 'dot' in p_name:
#         bg_type = 'dots'
#     elif 'ring' in p_name:
#         bg_type = 'rings'
#     else:
#         raise ValueError(f"p_name: {p_name} does not contain 'dots' or 'rings'")
#
#     # find substring 'OLED' in p_name
#     if 'OLED' in p_name:
#         mon_name = 'OLED'
#     elif 'asus_cal' in p_name:
#         mon_name = 'asus_cal'
#     else:
#         mon_name = 'asus_uncal'
#
#     # find substring 'quart' or 'half' in script_name
#     if 'quart' in p_name:
#         flow_speed_prop = .25
#     elif 'Quart' in p_name:
#         flow_speed_prop = .25
#     elif 'third' in p_name:
#         flow_speed_prop = .33
#     elif 'half' in p_name:
#         flow_speed_prop = .5
#     else:
#         flow_speed_prop = 1
#
#     # find substring 'spokes' in p_name
#     if 'spokes' in p_name:
#         mask_type = 'spokes'
#     else:
#         mask_type = 'circles'
#
#     print(f"bg_type: {bg_type}, mon_name: {mon_name}, flow_speed_prop: {flow_speed_prop}, mask_type: {mask_type}")
#
#
#     # check for master file
#     if not os.path.exists(p_master_path):
#         print(f"p_master_path: {p_master_path} does not exist")
#         continue
#
#     p_master_file = pd.read_csv(p_master_path)
#     print(f"p_master_file: {p_master_file.shape}\n{p_master_file.head()}")
#
#     # add any missing columns
#     if 'Monitor' not in p_master_file.columns:
#         p_master_file['monitor'] = mon_name
#     if 'p_name' not in p_master_file.columns:
#         p_master_file['p_name'] = p_name
#     if 'script_name' not in p_master_file.columns:
#         p_master_file['script_name'] = script_name
#     if 'bg_type' not in p_master_file.columns:
#         p_master_file['bg_type'] = bg_type
#     if 'flow_speed_prop' not in p_master_file.columns:
#         p_master_file['flow_speed_prop'] = flow_speed_prop
#     if 'mask_type' not in p_master_file.columns:
#         p_master_file['mask_type'] = mask_type
#
#
#
#     # search for column containing 'ISI_'
#     isi_col = [col for col in p_master_file.columns if 'ISI_' in col][0]
#     ISI_val = isi_col.split('_')[1]
#     p_master_file['ISI'] = ISI_val
#
#     # rename isi_col to 'threshold'
#     p_master_file.rename(columns={isi_col: 'threshold'}, inplace=True)
#
#     # append to master list
#     MLM_master_list.append(p_master_file)
#
#
#
# # concatenate all the dataframes in the master list
# MLM_master_df = pd.concat(MLM_master_list)
# MLM_master_df.drop(columns=['stair_names', 'neg_sep'], inplace=True)
# print(f"MLM_master_df: {MLM_master_df.shape}\n{MLM_master_df.head()}")
# MLM_master_df.to_csv(os.path.join(root_path, 'MLM_master.csv'), index=False)

