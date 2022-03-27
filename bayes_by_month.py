import pandas as pd
import numpy as np

month_list = ['2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06']
month_list += ['2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04',
               '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12']

# f = open('S_test_0411_deleted_rm_x_aligned.fasta').read()
# lst = f.split('>')
# while '' in lst:
#     lst.remove('')
# for i in lst:
#     label.append(i.splitlines()[0].split('|')[-1])

# df = pd.read_csv('df_amino_test_total.csv', index_col=0, engine='python')
# print(np.percentile(df['Prob_PR'], 80))
# df = df.fillna('-')
# amino_total = set()
# # print(df.columns[2:-6].tolist())
# # for i in df.columns[2:-6]:
# #     amino_total.update(set(df[i]))
# print(df.columns[1:-5].tolist())
# for i in df.columns[1:-5]:
#     amino_total.update(set(df[i]))
# # amino_total.remove('X')
# print(amino_total)
# amino_total = sorted(amino_total)
# amount_total = len(df)
# amount_adapt = len(df[df['Prob_PR'] >= 0.5])
# amount_inadapt = len(df[df['Prob_PR'] < 0.5])
# print(amount_adapt)
# print(amount_inadapt)

# time = []
# df = pd.read_csv('df_amino_trying_new_revised_new_add_duplicated.csv', index_col=0, engine='python')
# df.columns = ['country', 'date']+[str(i) for i in range(1, 1277)]+['Prob_PR', 'Prob_CH', 'Prob_CA', 'Prob_AR', 'Prob_SU']
# df.to_csv('df_amino_trying_new_revised_new_add_duplicated.csv')
#
# for i in range(len(df)):
#     time.append('-'.join(str(df.iloc[i][1]).split('-')[:2]))
# # f = open('month_and_country_0411.txt').read().splitlines()
# # for i in f:
# #     time.append(i.split('\t')[0])
# print(time)
# for i in month_list:
#     ind = []
#     for j in range(len(time)):
#         if time[j] == i:
#             ind.append(j)
#     df_new = df.iloc[ind]
#     df_new.to_csv('bayes_by_month_final_duplicated/df_amino_%s.csv' % i)

for m in month_list:
    df = pd.read_csv('bayes_by_month_final_duplicated/df_amino_%s.csv' % m, index_col=0, engine='python')
    # df = df.fillna('-')
    amino_total = set()
    print(df.columns[2:-5].tolist())
    for i in df.columns[2:-5]:
        amino_total.update(set(df[i]))
    # amino_total.remove('X')
    print(amino_total)
    amino_total = sorted(amino_total)
    amount_total = len(df)
    amount_adapt = len(df[df['Prob_PR'] >= 0.5])
    amount_inadapt = len(df[df['Prob_PR'] < 0.5])
    print(amount_adapt)
    print(amount_inadapt)

    dic = {}
    dic_adapt = {}
    dic_inadapt = {}
    for i in df.columns[2:-5]:
        amino_set = set(df[i])
        for j in amino_total:
            if j in amino_set:
                amount1 = len(df[df[i] == j][df['Prob_PR'] >= 0.5])
                amount0 = len(df[df[i] == j][df['Prob_PR'] < 0.5])
                if amount1 > 20:
                    prob_adapt = amount1 / amount_adapt
                else:
                    prob_adapt = -1
                if amount0 > 20:
                    prob_inadapt = amount0 / amount_inadapt
                else:
                    prob_inadapt = -1
                if j not in dic_adapt.keys():
                    dic_adapt[j] = [prob_adapt]
                else:
                    dic_adapt[j].append(prob_adapt)
                if j not in dic_inadapt.keys():
                    dic_inadapt[j] = [prob_inadapt]
                else:
                    dic_inadapt[j].append(prob_inadapt)
            else:
                if j not in dic_adapt.keys():
                    dic_adapt[j] = [0]
                else:
                    dic_adapt[j].append(0)
                if j not in dic_inadapt.keys():
                    dic_inadapt[j] = [0]
                else:
                    dic_inadapt[j].append(0)
    df_adapt = pd.DataFrame(dic_adapt, index=df.columns[2:-5].tolist())
    df_inadapt = pd.DataFrame(dic_inadapt, index=df.columns[2:-5].tolist())
    df_adapt.to_csv('bayes_by_month_final_duplicated/bayes_adapt_%s.csv' % m)
    df_inadapt.to_csv('bayes_by_month_final_duplicated/bayes_nonadapt_%s.csv' % m)
