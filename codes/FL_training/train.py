from codes.utilites.model import client_node_ANN
from codes.FL_training.FL_algorithms import FedAvg, FedProx, Fed_KD
from codes.utilites.requirments import *


def training_without_heterogeneity(n_iter, model_params, training_sets_iid, testing_sets_iid, server_test_loader,
                                   percent_of_stragglers, mu, epochs, lr, decay, server_train_loader,
                                   KD_server_loader):

    model = client_node_ANN(num_classes=model_params[0], input_size=model_params[1]).to(device)
    avg_loss_hist, avg_acc_hist, avg_server_acc, avg_server_loss, avg_models_hist, avg_server_hist, avg_client_hist_iid, avg_global_hist_iid, avg_iid_f1 = FedAvg(
        model, model_params, percent_of_stragglers, training_sets_iid
        , n_iter, testing_sets_iid, server_test_loader,
        mu=0, epochs=epochs, lr=lr, decay=decay, noSysHeter=True)

    model = client_node_ANN(num_classes=model_params[0], input_size=model_params[1]).to(device)
    print(f"\nTraining with mu: {mu}")
    prox_loss_hist, prox_acc_hist, prox_server_acc \
        , prox_server_loss, prox_models_hist, prox_server_hist, prox_client_hist_iid, prox_global_hist_iid, prox_iid_f1 = FedProx(
        model, model_params, percent_of_stragglers,
        training_sets_iid,
        n_iter,
        testing_sets_iid,
        server_test_loader,
        mu=mu,
        epochs=epochs,
        lr=lr,
        decay=decay, noSysHeter=True)

    print(f"\nTraining with FedKD_prox: {mu}")
    model = client_node_ANN(num_classes=model_params[0], input_size=model_params[1]).to(device)
    kd_loss_hist, kd_acc_hist, kd_server_acc, kd_server_loss, kd_models_hist, kd_server_hist, kd_client_hist_iid, kd_global_hist_iid, kd_iid_f1 = Fed_KD(
        model, model_params, percent_of_stragglers, server_train_loader, KD_server_loader, training_sets_iid
        , n_iter, testing_sets_iid, server_test_loader,
        mu=mu, epochs=epochs, lr=lr, decay=decay, noSysHeter=True)

    res_iid = {

        "avg_global_hist_iid": avg_global_hist_iid,
        "prox_globals_hist_iid": prox_global_hist_iid,
        "FD_global_hist_iid": kd_global_hist_iid,

        "avg_iid_f1": avg_iid_f1,
        "prox_iid_f1": prox_iid_f1,
        "kd_iid_f1": kd_iid_f1
    }
    return res_iid


def training_with_heterogeneity(n_iter, model_params, training_sets_niid, testing_sets_niid, server_test_loader,
                                percent_of_stragglers, mu, epochs, lr, decay, server_train_loader, KD_server_loader):

    model = client_node_ANN(num_classes=model_params[0], input_size=model_params[1]).to(device)
    avg_loss_hist_niid, avg_acc_hist_niid, avg_server_acc_niid, avg_server_loss_niid, avg_models_hist_niid, avg_server_hist_niid, avg_client_hist_noniid, avg_global_hist_noniid, avg_niid_f1 = FedAvg(
        model, model_params, percent_of_stragglers, training_sets_niid
        , n_iter, testing_sets_niid, server_test_loader,
        mu=0, epochs=epochs, lr=lr, decay=1, noSysHeter=False)

    model = client_node_ANN(num_classes=model_params[0], input_size=model_params[1]).to(device)
    print(f"\nTraining with mu: {mu}")

    prox_loss_hist_niid, prox_acc_hist_niid, prox_server_acc_iid \
        , prox_server_loss_niid, prox_models_hist_niid, prox_server_hist_niid, prox_client_hist_niid, prox_global_hist_niid, prox_niid_f1 = FedProx(
        model, model_params, percent_of_stragglers,
        training_sets_niid,
        n_iter,
        testing_sets_niid,
        server_test_loader,
        mu=mu,
        epochs=epochs,
        lr=lr,
        decay=decay, noSysHeter=False)

    print(f"\nTraining with FedKD_prox: {mu}")
    model = client_node_ANN(num_classes=model_params[0], input_size=model_params[1]).to(device)
    kd_loss_hist_niid, kd_acc_hist_niid, kd_server_acc_niid, kd_server_loss_niid, kd_models_hist_niid, kd_server_hist_niid, kd_client_hist_iid_niid, kd_global_hist_niid, kd_niid_f1 = Fed_KD(
        model, model_params, percent_of_stragglers, server_train_loader, KD_server_loader, training_sets_niid
        , n_iter, testing_sets_niid, server_test_loader,
        mu=mu, epochs=epochs, lr=lr, decay=decay, noSysHeter=False)

    res_noniid = {

        "avg_global_hist_noniid": avg_global_hist_noniid,
        "prox_globals_hist_noniid": prox_global_hist_niid,
        "FD_global_hist_noniid": kd_global_hist_niid,

        "avg_niid_f1": avg_niid_f1,
        "prox_niid_f1": prox_niid_f1,
        "kd_niid_f1": kd_niid_f1

    }

    return res_noniid
