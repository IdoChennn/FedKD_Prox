import matplotlib.pyplot as plt

from codes.utilites.requirments import *


def plot_results(res_iid, res_noniid, n_iter, dataset_name):
    iid_avg_global_hist = res_iid["avg_global_hist_iid"]
    iid_prox_globals_hist = res_iid["prox_globals_hist_iid"]
    iid_FD_globals_hist = res_iid["FD_global_hist_iid"]

    non_iid_avg_global_hist = res_noniid["avg_global_hist_noniid"]
    non_iid_prox_globals_hist = res_noniid["prox_globals_hist_noniid"]
    non_iid_FD_globals_hist = res_noniid["FD_global_hist_noniid"]

    ### accuracy (no system and statistial heterogenity) ###

    avg_global_accuracy_iid = [i[1] for i in iid_avg_global_hist]
    prox_global_accuracy_iid = [i[1] for i in iid_prox_globals_hist]
    kd_global_accuracy_iid = [i[1] for i in iid_FD_globals_hist]

    plt.figure(figsize=(15, 7), facecolor='white')
    plt.title(f"Global model accuracy under homogeneous environment using {dataset_name} dataset", fontsize=32)
    plt.xlabel("global round", fontsize=32)
    plt.ylabel("Accuracy", fontsize=32)
    plt.plot(avg_global_accuracy_iid, label="FedAvg", color="black", linestyle="dashed", linewidth=4)
    plt.plot(prox_global_accuracy_iid, label="FedProx", color="red", linestyle="solid", linewidth=4)
    plt.plot(kd_global_accuracy_iid, label="FedKD_prox", color="blue", linestyle="solid", linewidth=4)
    plt.xticks(np.arange(0, n_iter, 10), fontsize=25, color='black')
    plt.xticks(np.arange(0, n_iter, 2), fontsize=25, color='black')
    plt.yticks(fontsize=25, color='black')
    ax = plt.gca()

    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.set_facecolor('white')
    plt.show()

    ### training loss (no system and statistial heterogenity) ###

    avg_global_loss_iid = [i[0] for i in iid_avg_global_hist]
    prox_global_loss_iid = [i[0] for i in iid_prox_globals_hist]
    kd_global_loss_iid = [i[0] for i in iid_FD_globals_hist]

    plt.figure(figsize=(15, 7), facecolor='white')
    print(f"Global model loss under homogeneous environment using {dataset_name} dataset")
    plt.xlabel("global round", fontsize=32)
    plt.ylabel("Loss", fontsize=32)
    plt.plot(avg_global_loss_iid, label="FedAvg", color="black", linestyle="dashed", linewidth=4)
    plt.plot(prox_global_loss_iid, label="FedProx", color="red", linestyle="solid", linewidth=4)
    plt.plot(kd_global_loss_iid, label="FedKD_Prox", color="blue", linestyle="solid", linewidth=4)
    plt.xticks(np.arange(0, n_iter, 10), fontsize=25, color='black')
    plt.xticks(np.arange(0, n_iter, 2), fontsize=25, color='black')
    plt.yticks(fontsize=25, color='black')
    ax = plt.gca()

    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.set_facecolor('white')
    plt.show()

    ### Accuracy (system and statistial heterogenity) ###
    avg_global_accuracy_niid = [i[1] for i in non_iid_avg_global_hist]
    prox_global_accuracy_niid = [i[1] for i in non_iid_prox_globals_hist]
    kd_global_accuracy_niid = [i[1] for i in non_iid_FD_globals_hist]

    plt.figure(figsize=(15, 7), facecolor='white')
    plt.title(f"Global model accuracy under heterogeneous environment using {dataset_name} dataset", fontsize=32)
    plt.xlabel("Epochs", fontsize=32)
    plt.ylabel("Accuracy", fontsize=32)
    plt.plot(avg_global_accuracy_niid, label='Fed_avg', color="black", linestyle="dashed", linewidth=4)

    plt.plot(prox_global_accuracy_niid, label='Fed_prox', color="red", linestyle="solid", linewidth=4)

    plt.plot(kd_global_accuracy_niid, label='FedKD-Prox', color="blue", linestyle="solid", linewidth=4)

    plt.xticks(np.arange(0, n_iter, 2), fontsize=25, color='black')
    plt.yticks(fontsize=25, color='black')
    ax = plt.gca()

    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.set_facecolor('white')
    plt.show()

    ### training loss (system and statistial heterogenity) ###

    avg_global_loss_niid = [i[0] for i in non_iid_avg_global_hist]
    prox_global_loss_niid = [i[0] for i in non_iid_prox_globals_hist]
    kd_global_loss_niid = [i[0] for i in non_iid_FD_globals_hist]

    plt.figure(figsize=(15, 7), facecolor='white')
    plt.title(f"Global model training loss under heterogeneous environment using {dataset_name} dataset", fontsize=32)
    plt.xlabel("Epochs", fontsize=32)
    plt.ylabel("Loss", fontsize=32)
    plt.plot(avg_global_loss_niid, label="FedAvg", color="black", linestyle="dashed", linewidth=4)
    plt.plot(prox_global_loss_niid, label="FedProx", color="red", linestyle="solid", linewidth=4)
    plt.plot(kd_global_loss_niid, label="Fed_KD", color="blue", linestyle="solid", linewidth=4)

    plt.xticks(np.arange(0, n_iter, 2), fontsize=25, color='black')
    plt.yticks(fontsize=25, color='black')
    # plt.legend(loc='upper right', fontsize='x-large')
    ax = plt.gca()

    # Set the spines (axes line) color to black
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Hide the top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set the background color for the axes area to white
    ax.set_facecolor('white')
    # Set the axes background color to white
    # ax.set_facecolor('white')
    plt.show()
