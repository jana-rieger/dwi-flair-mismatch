"""
This calculates the statistical significance of the differences between modal performances using McNemar's test.
"""

import os.path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

pd.set_option('display.max_columns', None)

# SET
# whether to compare test performance within folds or merged test performance from all folds
MERGED_FOLDS = True


def main(merged_folds):
    # LOAD MODEL PREDICTIONS
    # path to results
    root_results_dir = r'C:\Users\jrieger\GITrepos\mmop\results\tti\1000plus\tti_1cnn\dwi\dicho_tti'

    # baseline model
    base_result_dir = os.path.join(root_results_dir, 'groupnet', 'results')
    base_path_fold0 = os.path.join(base_result_dir,
                                   'fold0_groupnet_bs8_dr0.3_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    base_path_fold1 = os.path.join(base_result_dir,
                                   'fold1_groupnet_bs8_dr0.2_ep200_flt8_k_reg1e-06_lr0.0001_dwi_mmnt0.9_SGD_test_pred.csv')
    base_path_fold2 = os.path.join(base_result_dir,
                                   'fold2_groupnet_bs8_dr0.2_ep200_flt8_k_reg1e-06_lr0.0001_dwi_mmnt0.9_SGD_test_pred.csv')
    base_path_fold3 = os.path.join(base_result_dir,
                                   'fold3_groupnet_bs8_dr0.4_ep200_flt8_k_reg1e-06_lr0.001_dwi_mmnt0.9_SGD_test_pred.csv')

    base_df_f0 = pd.read_csv(base_path_fold0, index_col=0)
    base_df_f1 = pd.read_csv(base_path_fold1, index_col=0)
    base_df_f2 = pd.read_csv(base_path_fold2, index_col=0)
    base_df_f3 = pd.read_csv(base_path_fold3, index_col=0)

    # compared model
    # compared_result_dir = os.path.join(root_results_dir, 'groupnet_pretrained', 'results')
    # compared_path_fold0 = os.path.join(compared_result_dir,
    # 							   'fold0_groupnet_pretrained_bs8_dr0.3_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    # compared_path_fold1 = os.path.join(compared_result_dir,
    # 							   'fold1_groupnet_pretrained_bs8_dr0.2_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    # compared_path_fold2 = os.path.join(compared_result_dir,
    # 							   'fold2_groupnet_pretrained_bs8_dr0.2_ep200_flt8_k_reg1e-06_lr0.001_dwi_mmnt0.9_SGD_test_pred.csv')
    # compared_path_fold3 = os.path.join(compared_result_dir,
    # 							   'fold3_groupnet_pretrained_bs8_dr0.4_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')

    # compared_result_dir = os.path.join(root_results_dir, 'groupnet_AE', 'results')
    # compared_path_fold0 = os.path.join(compared_result_dir,
    # 							   'fold0_a0.5_groupnet_AE_bs8_d_dr0.2_d_reg0_dr0.2_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    # compared_path_fold1 = os.path.join(compared_result_dir,
    # 							   'fold1_a0.5_groupnet_AE_bs8_d_dr0.2_d_reg0_dr0.2_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    # compared_path_fold2 = os.path.join(compared_result_dir,
    # 							   'fold2_a0.5_groupnet_AE_bs8_d_dr0.0_d_reg0_dr0.2_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    # compared_path_fold3 = os.path.join(compared_result_dir,
    # 							   'fold3_a0.5_groupnet_AE_bs8_d_dr0.2_d_reg0_dr0.2_ep200_flt8_k_reg1e-06_lr0.001_dwi_mmnt0.9_SGD_test_pred.csv')

    compared_result_dir = os.path.join(root_results_dir, 'groupnet_AE_pretrained', 'results')
    compared_path_fold0 = os.path.join(compared_result_dir,
                                       'fold0_a0.75_groupnet_AE_pretrained_bs8_d_dr0.0_d_reg0_dr0.3_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    compared_path_fold1 = os.path.join(compared_result_dir,
                                       'fold1_a0.75_groupnet_AE_pretrained_bs8_d_dr0.0_d_reg0_dr0.2_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    compared_path_fold2 = os.path.join(compared_result_dir,
                                       'fold2_a0.75_groupnet_AE_pretrained_bs8_d_dr0.0_d_reg0_dr0.3_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')
    compared_path_fold3 = os.path.join(compared_result_dir,
                                       'fold3_a0.75_groupnet_AE_pretrained_bs8_d_dr0.2_d_reg0_dr0.2_ep200_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_test_pred.csv')

    compared_df_f0 = pd.read_csv(compared_path_fold0, index_col=0)
    compared_df_f1 = pd.read_csv(compared_path_fold1, index_col=0)
    compared_df_f2 = pd.read_csv(compared_path_fold2, index_col=0)
    compared_df_f3 = pd.read_csv(compared_path_fold3, index_col=0)

    # TESTING
    if merged_folds:
        merged_base_df = pd.concat([base_df_f0, base_df_f1, base_df_f2, base_df_f3], axis=0)
        merged_compared_df = pd.concat([compared_df_f0, compared_df_f1, compared_df_f2, compared_df_f3], axis=0)

        merged_base_df = merged_base_df.add_prefix('base_')
        merged_compared_df = merged_compared_df.add_prefix('compared_')

        print('base df', merged_base_df.head())
        print('compared df', merged_compared_df.head())
        print('base df shape', merged_base_df.shape)
        print('compared df shape', merged_compared_df.shape)

        # MERGE DFs
        merged_df = pd.concat([merged_base_df, merged_compared_df], axis=1)

        crosstab, contingency_table = get_contingency_table(merged_df)

        # CHOOSE THE RIGHT VERSION OF THE MCNEMAR'S TEST
        count_less_than_25 = False
        if crosstab.loc[True, False] < 25 or crosstab.loc[False, True] < 25:
            count_less_than_25 = True

        # CALCULATE MCNEMAR'S TEST
        mcnemars_test(contingency_table, count_less_than_25)
    else:
        i = 0
        for base_df, compared_df in zip([base_df_f0, base_df_f1, base_df_f2, base_df_f3],
                                        [compared_df_f0, compared_df_f1, compared_df_f2, compared_df_f3]):
            print('FOLD', i)

            base_df = base_df.add_prefix('base_')
            compared_df = compared_df.add_prefix('compared_')

            print('base df', base_df.head())
            print('compared df', compared_df.head())
            print('base df shape', base_df.shape)
            print('compared df shape', compared_df.shape)

            # MERGE DFs
            merged_df = pd.concat([base_df, compared_df], axis=1)

            # GET CONTINGENCY TABLE
            crosstab, contingency_table = get_contingency_table(merged_df)

            # CHOOSE THE RIGHT VERSION OF THE MCNEMAR'S TEST
            count_less_than_25 = False
            if crosstab.loc[True, False] < 25 or crosstab.loc[False, True] < 25:
                count_less_than_25 = True

            # CALCULATE MCNEMAR'S TEST
            mcnemars_test(contingency_table, count_less_than_25)

            i += 1
            print('-' * 20)
            print()


def get_contingency_table(merged_df):
    merged_df['base_correct'] = merged_df['base_pred_class'] == merged_df['base_label']
    merged_df['compared_correct'] = merged_df['compared_pred_class'] == merged_df['compared_label']

    print('merged df', merged_df.head(10))
    print('merged df shape', merged_df.shape)

    crosstab = pd.crosstab(index=merged_df['base_correct'], columns=merged_df['compared_correct'])

    contingency_table = [[crosstab.loc[True, True], crosstab.loc[True, False]],
                         [crosstab.loc[False, True], crosstab.loc[False, False]]]

    print('contingency table', contingency_table)

    return crosstab, contingency_table


def mcnemars_test(table, count_less_than_25):
    if count_less_than_25:
        print('Calculating McNemar\'s test, version: count less than 25.')
        result = mcnemar(table, exact=True)
    else:
        print('Calculating McNemar\'s test, version: count greater than or equal to 25.')
        result = mcnemar(table, exact=False, correction=True)

    # SUMMARIZE THE FINDINGS
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

    # INTERPRET THE P-VALUE
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')

        return result


if __name__ == '__main__':
    main(MERGED_FOLDS)
