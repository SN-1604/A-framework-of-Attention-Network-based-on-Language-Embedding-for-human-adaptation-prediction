import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models import word2vec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random

# alpha_acc = pd.read_csv('variants/alpha_accesion.csv', header=None)
# print(alpha_acc)
alpha_acc = list(pd.read_csv('variants/alpha_accesion_new.csv', header=None)[0])
delta_acc = set(list(pd.read_csv('variants/delta_accession_number_new.csv', header=None)[0])+\
            list(pd.read_csv('variants/delta_accession_number_new_add.csv', header=None)[0]))
beta_acc = list(pd.read_csv('variants/beta_accesion_new.csv', header=None)[0])
gamma_acc = list(pd.read_csv('variants/gamma_accesion_new.csv', header=None)[0])
lambda_acc = list(pd.read_csv('variants/lambda_accesion_new.csv', header=None)[0])
mu_acc = list(pd.read_csv('variants/mu_accesion_new.csv', header=None)[0])

# attn_train = np.load('attn_out_train_SARS_MERS_human_revised_rm_except_plus.npy')
# attn_test = np.load('attn_out_test_SARS_MERS_human_revised_rm_except_plus.npy')
# attn_omicron = np.load('attn_out_omicron_SARS_MERS_human_rm_except_plus.npy')
# y_train = np.load('y_train_S_SARS_MERS_human_revised_rm_except_plus.npy')
# y_val = np.load('y_val_S_SARS_MERS_human_revised_rm_except_plus.npy')
# pred_train_val = np.load('prediction_train_SARS_MERS_human_revised_rm_except_plus.npy')

attn_train = np.load('attn_out_train_trying_new_revised.npy')
attn_test = np.load('attn_out_test_trying_new_revised_new.npy')
attn_omicron = np.load('attn_out_omicron_trying_new_revised_new.npy')
# attn_test_after = np.load('attn_out_test_after_0801_trying_new_revised.npy')
y_train = np.load('y_train_S_trying_new_revised.npy')
y_val = np.load('y_val_S_trying_new_revised.npy')
pred_train_val = np.load('prediction_train_trying_new_revised.npy')
print(attn_omicron.shape)

attn_total = np.concatenate([attn_train, attn_test, attn_omicron], axis=0)
pca = PCA(n_components=3)
pcavec = pca.fit_transform(attn_total)
dic_host = {'human': 0, 'bat': 1, 'carnivora': 2, 'artiodactyla': 3, 'swine': 4, 'high-patho': 5}
label_one_train = [dic_host[i.split('_')[0]] for i in y_train]
label_one_val = [dic_host[i.split('_')[0]] for i in y_val]
label_pca = label_one_train+label_one_val+[6]*(attn_test.shape[0]+attn_omicron.shape[0])
label_pca = np.asarray(label_pca)
print(pcavec.shape)
print(label_pca.shape)
my_members = label_pca == 1
mean_ch = np.mean(pcavec[my_members], axis=0)
pcavec -= mean_ch
my_members = label_pca == 0
mean_pc = np.mean(pcavec[my_members], axis=0)

# v_new = mean_pc/np.linalg.norm(mean_pc)
# v_base = np.array([1, 0])
# product = np.dot(v_new, v_base)
# angel = np.arccos(product)
# cross = np.cross(v_new, v_base)
# if cross.item() > 0:
#     angel = 2*math.pi-angel
# matrix = np.array([[np.cos(angel), -np.sin(angel)], [np.sin(angel), np.cos(angel)]])
# pcavec = np.dot(pcavec, matrix)

my_members = label_pca == 1
mean_ch = np.mean(pcavec[my_members], axis=0)
my_members = label_pca == 0
mean_pr = np.mean(pcavec[my_members], axis=0)
# my_members = label_pca == 5
# mean_pc = np.mean(pcavec[my_members], axis=0)
my_members = label_pca == 6
mean_sars_cov_2 = np.mean(pcavec[my_members][pcavec[my_members][:, 0] > 0], axis=0)
# print(mean_sars_cov_2)

# f = word2vec.LineSentence('S_filtered_SARS_MERS_adjusted_rm_x_revised.fasta')
# t = 1
# sequence = []
# label_train = []
# for i in f:
#     if t % 2 == 0:
#         sequence.append(i[0])
#     else:
#         label_train.append(i[0].lstrip('>'))
#     t += 1

label_test = []
f = word2vec.LineSentence('S_test_0411_deleted_rm_x.fasta')
t = 1
sequence = []
for i in f:
    if t % 2 == 0:
        sequence.append(i[0])
    else:
        label_test.append(i[0].lstrip('>'))
    t += 1

for t in range(8):
    f_test = open('./test_after_0411/S_test_after_0411_sampled_%d_rm_x.fasta' % (t+1)).read().splitlines()
    for i in range(0, len(f_test), 2):
        label_test.append(f_test[i].lstrip('>'))
    for i in range(1, len(f_test), 2):
        sequence.append(f_test[i])
f = word2vec.LineSentence('S_test_after_0801_sampled.fasta')
t = 1
# sequence = []
for i in f:
    if t % 2 == 0:
        sequence.append(i[0])
    else:
        label_test.append(i[0].lstrip('>'))
    t += 1
print(len(label_test))


# alpha = []
# beta = []
# gamma = []
# delta = []
# lambda_number = []
# mu = []
# for i in range(len(label_test)):
#     if label_test[i] in alpha_acc:
#         alpha.append(i)
#     if label_test[i] in beta_acc:
#         beta.append(i)
#     if label_test[i] in gamma_acc:
#         gamma.append(i)
#     if label_test[i] in delta_acc:
#         delta.append(i)
#     if label_test[i] in lambda_acc:
#         lambda_number.append(i)
#     if label_test[i] in mu_acc:
#         mu.append(i)
#     print(i)
# print(alpha)
# np.save('ind_alpha_new_add.npy', alpha)
# np.save('ind_beta_new_add.npy', beta)
# np.save('ind_gamma_new_add.npy', gamma)
# np.save('ind_delta_new_add.npy', delta)
# np.save('ind_lambda_new_add.npy', lambda_number)
# np.save('ind_mu_new_add.npy', mu)

alpha = np.load('ind_alpha_CASI_revised.npy')
beta = np.load('ind_beta_CASI_revised.npy')
gamma = np.load('ind_gamma_CASI_revised.npy')
delta = np.load('ind_delta_CASI_revised.npy')
lambda_number = np.load('ind_lambda_CASI_revised.npy')
mu = np.load('ind_mu_CASI_revised.npy')
np.set_printoptions(threshold=np.inf)
# pred_test = np.load('prediction_test_SARS_MERS_human_revised_rm_except_plus.npy')
pred_test = np.load('prediction_test_trying_new_revised_new.npy')
print(pred_test.shape)

f = open('omicron_0104_rm_x.fasta').read().splitlines()
# f = open('G:/Py Files/coronavirus/explore/variants/omicron_sampled_rm_x.fasta').read().splitlines()
sequence_omicron = []
label_omicron = []
for i in f:
    if '>' in i:
        label_omicron.append(i)
    else:
        sequence_omicron.append(i)
label_omicron = np.asarray(label_omicron)
# print(pred_test)
# pred_omicron = np.load('prediction_omicron.npy')
# print(pred_omicron)
others = np.setdiff1d(np.arange(0, attn_total.shape[0]), delta)
others = np.setdiff1d(others, alpha)
others = np.setdiff1d(others, beta)
others = np.setdiff1d(others, gamma)
others = np.setdiff1d(others, lambda_number)
others = np.setdiff1d(others, mu)
others = np.setdiff1d(others, np.arange(attn_train.shape[0]+attn_test.shape[0], attn_total.shape[0]))
others = np.setdiff1d(others, np.arange(0, attn_train.shape[0]))


def random_hex(length):
    result = hex(random.randint(0,16**length)).replace('0x','').upper()
    if(len(result)<length):
        result = '0'*(length-len(result))+result
    return result


label_test = np.asarray(label_test)
label_train = np.concatenate([y_train, y_val, label_test, np.asarray([6]*attn_omicron.shape[0])])
# # # print(label_one_train)
# # print(label_train[:attn_train.shape[0]])
# # print(label_pca[:attn_train.shape[0]])
# for k in range(6):
#     my_members = label_pca == k
#     if k == 0:
#         member = pred_train_val == k
#         np.random.seed(100)
#         c = '#'+str(random_hex(6))
#         # FD9409
#         print(c)
#         # plt.scatter(pcavec[:attn_train.shape[0]][member][:, 0], pcavec[:attn_train.shape[0]][member][:, 1], label='PR', color='#C4E4B4')
#         plt.scatter(pcavec[:attn_train.shape[0]][member][:, 0], pcavec[:attn_train.shape[0]][member][:, 1], label='PR',
#                     color='#FD9409')
#         # pr_less = pcavec[:attn_train.shape[0]][member][:, 0] > 4
#         # print(label_train[:attn_train.shape[0]][member][pr_less])
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha].shape[0]), 100, replace=False)
#         np.random.seed(100)
#         c = '#'+str(random_hex(6))
#         # 0D48ED
#         print(c)
#         # plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha][seed][:, 0],
#         #             pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha][seed][:, 1], label='Alpha', color='firebrick')
#         plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha][seed][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha][seed][:, 1], label='Alpha', color='#0D48ED')
#         alpha_less = pcavec[attn_train.shape[0]:attn_train.shape[0] + attn_test.shape[0]][alpha][seed][:, 0] < 4
#         alpha_over = pcavec[attn_train.shape[0]:attn_train.shape[0] + attn_test.shape[0]][alpha][seed][:, 0] > 4
#         # print(label_test[alpha][seed][alpha_less])
#         # print(label_test[alpha][seed][alpha_over])
#         print(label_test[alpha][seed][alpha_less].shape)
#         print(label_test[alpha][seed][alpha_over].shape)
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta].shape[0]),
#                                 100, replace=False)
#         np.random.seed(100)
#         c = '#'+str(random_hex(6))
#         # A8DD1F
#         print(c)
#         # plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta][seed][:, 0],
#         #             pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta][seed][:, 1], label='Delta', color='midnightblue')
#         plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta][seed][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta][seed][:, 1], label='Delta', color='tomato')
#         delta_less = pcavec[attn_train.shape[0]:attn_train.shape[0] + attn_test.shape[0]][delta][seed][:, 0] < 4
#         delta_over = pcavec[attn_train.shape[0]:attn_train.shape[0] + attn_test.shape[0]][delta][seed][:, 0] > 4
#         print(label_test[delta][seed][delta_less])
#         print(label_test[delta][seed][delta_over])
#         print(label_test[delta][seed][delta_less].shape)
#         print(label_test[delta][seed][delta_over].shape)
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta].shape[0]),
#                                 50, replace=False)
#         np.random.seed(100)
#         c = '#'+str(random_hex(6))
#         # 6655BA
#         print(c)
#         # plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta][seed][:, 0],
#         #             pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta][seed][:, 1], label='Beta', color='y')
#         plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta][seed][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta][seed][:, 1], label='Beta', color='#6655BA')
#         # np.random.seed(100)
#         # seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma].shape[0]),
#         #                         100, replace=False)
#         np.random.seed(100)
#         c = '#'+str(random_hex(6))
#         # EAE290
#         print(c)
#         # plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma][:, 0],
#         #             pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma][:, 1], label='Gamma', color='darkorange')
#         plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma][:, 1], label='Gamma', color='#EAE290')
#         np.random.seed(100)
#         c = '#'+str(random_hex(6))
#         # 1AC626
#         print(c)
#         # plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][mu][:, 0],
#         #             pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][mu][:, 1], label='Mu', color='g')
#         plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][mu][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][mu][:, 1], label='Mu', color='darkgreen')
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]+attn_test.shape[0]:].shape[0]), 100, replace=False)
#         np.random.seed(1000)
#         c = '#'+str(random_hex(6))
#         # 394F58
#         print(c)
#         # plt.scatter(pcavec[attn_train.shape[0]+attn_test.shape[0]:][seed][:, 0],
#         #             pcavec[attn_train.shape[0]+attn_test.shape[0]:][seed][:, 1], label='Omicron', color='palevioletred')
#         plt.scatter(pcavec[attn_train.shape[0]+attn_test.shape[0]:][seed][:, 0],
#                     pcavec[attn_train.shape[0]+attn_test.shape[0]:][seed][:, 1], label='Omicron', color='purple')
#         print(pcavec[attn_train.shape[0]+attn_test.shape[0]:][pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 0] > 0].shape)
#         plt.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][0, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][0, 1], label='WIV04', color='r',
#                     s=60, marker='x')
#         # plt.scatter(mean_pr[0], mean_pr[1], color='black', s=60)
#         # plt.scatter(mean_ch[0], mean_ch[1], color='black', s=60)
#         # print(label_omicron[pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 0] > 0])
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:][others].shape[0]), 50, replace=False)
#         np.random.seed(1000)
#         c = '#'+str(random_hex(6))
#         print(c)
#         # C485F7
#         # plt.scatter(pcavec[attn_train.shape[0]:][others][seed][:, 0],
#         #             pcavec[attn_train.shape[0]:][others][seed][:, 1], label='Others', color='cyan')
#         plt.scatter(pcavec[attn_train.shape[0]:][others][seed][:, 0],
#                     pcavec[attn_train.shape[0]:][others][seed][:, 1], label='Others', color='saddlebrown')
#     if k == 1:
#         np.random.seed(100)
#         c = '#'+str(random_hex(6))
#         print(c)
#         # 260018
#         # plt.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], label='CH', color='#0F99B2')
#         plt.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], label='CH', color='#260018')
#         member = pred_train_val == k
#         # plt.scatter(pcavec[:attn_train.shape[0]][member][:, 0], pcavec[:attn_train.shape[0]][member][:, 1], label='CH', color='#0F99B2')
#         # pr_less = pcavec[:attn_train.shape[0]][member][:, 1] > 0.75
#         # print(label_train[:attn_train.shape[0]][member][pr_less])
#         # print(label_pca[:attn_train.shape[0]][member][pr_less])
#         # print(pred_train_val[:attn_train.shape[0]][member][pr_less])
#
#         # np.random.seed(100)
#         # seed = np.random.choice(np.arange(0, pcavec_after_nonadapt.shape[0]), 1000, replace=False)
#         # plt.scatter(pcavec_after_nonadapt[seed][:, 0], pcavec_after_nonadapt[seed][:, 1], color='purple', label='Nonadapt')
#         # np.random.seed(100)
#         # seed = np.random.choice(np.arange(0, pcavec_before_nonadapt.shape[0]), 1000, replace=False)
#         # plt.scatter(pcavec_before_nonadapt[seed][:, 0], pcavec_before_nonadapt[seed][:, 1], color='purple')
#         # print(pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][gamma][pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][gamma][:, 1] > 2].shape)
#         # print(pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][lambda_after][pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][lambda_after][:, 1] > 2].shape)
#         # print(pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][mu][pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][mu][:, 1] > 2].shape)
# plt.legend()
# # plt.ylim([-8, 2])
# plt.show()
# # plt.savefig('./figure/pca_trying_new_revised.png', format='png', dpi=600, bbox_inches='tight')

# fig = plt.figure()
# ax = Axes3D(fig)
# for k in range(7):
#     my_members = label_pca == k
#     if k == 0:
#         member = pred_train_val == k
#         ax.scatter(pcavec[:attn_train.shape[0]][member][:, 0], pcavec[:attn_train.shape[0]][member][:, 1],
#                    pcavec[:attn_train.shape[0]][member][:, 2], label='PR', color='#FD9409')
#         pr_less = pcavec[:attn_train.shape[0]][member][:, 2] > -2
#         print(label_train[:attn_train.shape[0]][member][pr_less])
#
#         np.random.seed(100)
#         # filtered = pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][alpha][:, 1] < 2
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha].shape[0]), 200, replace=False)
#         ax.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha][seed][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha][seed][:, 1],
#                    pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha][seed][:, 2], label='Alpha', color='#0D48ED')
#         print(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][alpha].shape)
#         # filtered = pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][delta][:, 1] < 2
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta].shape[0]),
#                                         100, replace=False)
#         ax.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta][seed][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta][seed][:, 1],
#                    pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta][seed][:, 2], label='Delta', color='tomato')
#         print(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][delta].shape)
#         # filtered = pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][beta][:, 1] < 2
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta].shape[0]),50, replace=False)
#         ax.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta][seed][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta][seed][:, 1],
#                    pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta][seed][:, 2], label='Beta', color='#6655BA')
#         print(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][beta].shape)
#         # filtered = pcavec[attn_train.shape[0] + attn_before_total.shape[0]:][gamma][:, 1] < 2
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma].shape[0]),
#                                         50, replace=False)
#         ax.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma][seed][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma][seed][:, 1],
#                    pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma][seed][:, 2], label='Gamma', color='#EAE290')
#         print(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][gamma].shape)
#         ax.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][lambda_number][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][lambda_number][:, 1],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][lambda_number][:, 2], label='Lambda', color='slategray')
#         print(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][lambda_number].shape)
#         ax.scatter(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][mu][:, 0],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][mu][:, 1],
#                     pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][mu][:, 2], label='Mu', color='saddlebrown')
#         print(pcavec[attn_train.shape[0]:attn_train.shape[0]+attn_test.shape[0]][mu].shape)
#         np.random.seed(100)
#         seed = np.random.choice(np.arange(0, attn_omicron.shape[0]), 200, replace=False)
#         # filt1 = np.arange(0, attn_omicron.shape[0])[pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 2] < -3]
#         # filt2 = np.arange(0, attn_omicron.shape[0])[pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 0] > 0]
#         # ax.scatter(pcavec[attn_train.shape[0]+attn_test.shape[0]:][filt1][:, 0], pcavec[attn_train.shape[0]+attn_test.shape[0]:][filt1][:, 1],
#         #            pcavec[attn_train.shape[0]+attn_test.shape[0]:][filt1][:, 2], label='Omicron', color='purple')
#         # ax.scatter(pcavec[attn_train.shape[0]+attn_test.shape[0]:][filt2][:, 0], pcavec[attn_train.shape[0]+attn_test.shape[0]:][filt2][:, 1],
#         #            pcavec[attn_train.shape[0]+attn_test.shape[0]:][filt2][:, 2], label='Omicron', color='purple')
#         ax.scatter(pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 0], pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 1],
#                    pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 2], label='Omicron', color='purple')
#         # np.random.seed(100)
#         # seed = np.random.choice(np.arange(0, pcavec[others].shape[0]), 100, replace=False)
#         # ax.scatter(pcavec[others][seed][:, 0], pcavec[others][seed][:, 1], pcavec[others][seed][:, 2],
#         #            label='Others', color='cyan')
#         print(pcavec[others].shape)
#         print(np.arange(0, attn_omicron.shape[0])[pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 2] > -2][pcavec[attn_train.shape[0]+attn_test.shape[0]:][pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 2] > -2][:, 0] < -1])
#         # print(np.arange(0, attn_omicron.shape[0])[pcavec[attn_train.shape[0]+attn_test.shape[0]:][:, 2] > -2])
#     if k == 1:
#         ax.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], pcavec[my_members][:, 2], label='CH', color='#260018')
# ax.scatter(mean_ch[0], mean_ch[1], mean_ch[2], color='black', s=40)
# # ax.scatter(mean_pc[0], mean_pc[1], mean_pc[2], color='black', s=40)
# ax.scatter(mean_pr[0], mean_pr[1], mean_pr[2], color='black', s=40)
# ax.scatter(mean_sars_cov_2[0], mean_sars_cov_2[1], mean_sars_cov_2[2], color='black', s=40)
# distance_ch = np.linalg.norm(mean_ch-mean_sars_cov_2)
# distance_pr = np.linalg.norm(mean_pr-mean_sars_cov_2)
# # distance_pc = np.linalg.norm(mean_pc-mean_sars_cov_2)
# # ax.quiver3D(mean_ch[0], mean_ch[1], mean_ch[2], mean_pc[0]-mean_ch[0], mean_pc[1]-mean_ch[1], mean_pc[2]-mean_ch[2],
# #             color='black', lw=2, arrow_length_ratio=0.05)
# ax.quiver3D(mean_sars_cov_2[0], mean_sars_cov_2[1], mean_sars_cov_2[2], mean_ch[0]-mean_sars_cov_2[0],
#             mean_ch[1]-mean_sars_cov_2[1], mean_ch[2]-mean_sars_cov_2[2], color='black', lw=2, arrow_length_ratio=0.05)
# # ax.quiver3D(mean_sars_cov_2[0], mean_sars_cov_2[1], mean_sars_cov_2[2], mean_pc[0]-mean_sars_cov_2[0],
# #             mean_pc[1]-mean_sars_cov_2[1], mean_pc[2]-mean_sars_cov_2[2], color='black', lw=2, arrow_length_ratio=0.05)
# ax.quiver3D(mean_sars_cov_2[0], mean_sars_cov_2[1], mean_sars_cov_2[2], mean_pr[0]-mean_sars_cov_2[0],
#             mean_pr[1]-mean_sars_cov_2[1], mean_pr[2]-mean_sars_cov_2[2], color='black', lw=2, arrow_length_ratio=0.05)
# # Arrow3D((mean_ch[0], mean_pc[0]-mean_ch[0]), (mean_ch[1], mean_pc[1]-mean_ch[1]), (mean_ch[2], mean_pc[2]-mean_ch[2]),
# #             color='black', lw=2)
# # Arrow3D(mean_sars_cov_2[0], mean_sars_cov_2[1], mean_sars_cov_2[2], mean_pr[0]-mean_sars_cov_2[0],
# #             mean_pr[1]-mean_sars_cov_2[1], mean_pr[2]-mean_sars_cov_2[2], color='black', lw=2)
# ax.text((mean_ch[0]+mean_sars_cov_2[0])/2, (mean_ch[1]+mean_sars_cov_2[1])/2+1, (mean_ch[2]+mean_sars_cov_2[2])/2,
#         s='%.2f' % distance_ch, size=12)
# ax.text((mean_pr[0]+mean_sars_cov_2[0])/2-2, (mean_pr[1]+mean_sars_cov_2[1])/2, (mean_pr[2]+mean_sars_cov_2[2])/2,
#         s='%.2f' % distance_pr, size=12)
# # ax.text((mean_pc[0]+mean_sars_cov_2[0])/2+2, (mean_pc[1]+mean_sars_cov_2[1])/2, (mean_pc[2]+mean_sars_cov_2[2])/2,
# #         s='%.2f' % distance_pc, size=12)
# ax.set_xlabel('PCA_1')
# ax.set_ylabel('PCA_2')
# ax.set_zlabel('PCA_3')
# plt.legend()
# plt.show()

fig = plt.figure()
ax = Axes3D(fig)
for k in range(7):
    my_members = label_pca == k
    if k == 0:
        ax.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], pcavec[my_members][:, 2], label='PR', color='#FD9409')
    if k == 1:
        ax.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], pcavec[my_members][:, 2], label='CH', color='#260018')
    if k == 2:
        ax.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], pcavec[my_members][:, 2], label='CA',
                   color='#55A0FB')
    if k == 3:
        ax.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], pcavec[my_members][:, 2], label='AR',
                   color='#008040')
    if k == 4:
        ax.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], pcavec[my_members][:, 2], label='SU',
                   color='#440151')
    if k == 6:
        ax.scatter(pcavec[my_members][:, 0], pcavec[my_members][:, 1], pcavec[my_members][:, 2], label='SARS-CoV-2',
                   color='r')
ax.set_xlabel('PCA_1')
ax.set_ylabel('PCA_2')
ax.set_zlabel('PCA_3')
plt.legend()
plt.show()
