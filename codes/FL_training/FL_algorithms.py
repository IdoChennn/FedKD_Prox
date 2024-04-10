from codes.FL_training.local_trainer import *
from codes.FL_training.server_aggregation import *


def FedAvg(model, model_param, percent_of_stragglers, training_sets: list, n_iter: int, testing_sets: list,
           server_data: list, mu=0, epochs=5,
           lr=10 ** -2, decay=1, noSysHeter=True):
    loss_f = loss_classifier
    K = len(training_sets)

    weights = [1 / K for i in range(K)]
    client_hist = {i: [] for i in range(K)}
    global_hist = []
    f1_score_list = []

    print("\n ############### FedAvg #################")
    print(f"No. of clients: {K}")
    print(f"Rounds: {n_iter}, Epochs: {epochs}, lr: {lr}, decay: {decay}")
    # print(
    #     f"Server testing samples : {len(server_data.dataset)}, No. of labels: {len(server_data.dataset.tensors[1].unique())}")
    for i in range(K):
        print(
            f"Client {i}\t\t Training samples: {len(training_sets[i].dataset)}, Testing samples: {len(testing_sets[i].dataset)}, No. of labels: {len(training_sets[i].dataset.tensors[1].unique())}")

    loss_hist, acc_hist, server_hist, models_hist = [], [], [], []

    server_loss, server_acc, correct, f1_score = loss_dataset(model, server_data, loss_f)
    print(
        f'\n Initial Server Testing, Loss: {server_loss:.4f}, Accuracy: {server_acc:.2f}%, Correct predictions: {correct}/{len(server_data.dataset)}, F1_score: {f1_score[2]:.2f}')

    aggregated_model = client_node_ANN(num_classes=model_param[0], input_size=model_param[1]).to(device)

    for i in range(n_iter):
        print(
            "\n\n====================================================================================================")
        print(f"Round: {i + 1}")
        clients_params, clients_losses, clients_models = [], [], []

        for k in range(K):
            print(
                f"\n{k}: Client {k} Starts, Training size: {len(training_sets[k].dataset)}, Testing size: {len(testing_sets[k].dataset)}")

            local_model = deepcopy(aggregated_model).to(device)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
            local_model, local_loss, local_acc, local_hist, systemHeterogeneity = local_learning(local_model,
                                                                                                 percent_of_stragglers,
                                                                                                 mu,
                                                                                                 local_optimizer,
                                                                                                 training_sets[k],
                                                                                                 testing_sets[k],
                                                                                                 epochs,
                                                                                                 loss_f, "FedAvg", True,
                                                                                                 noSysHeter)

            if systemHeterogeneity < 13:
                continue

            else:

                clients_losses.append(local_loss)
                client_hist[k].append(local_hist)

                list_params = list(local_model.parameters())
                list_params = [tens_param.detach().cpu() for tens_param in list_params]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

        # aggregated_model = weighted_average_models(deepcopy(model), clients_params, weights=weights)
        aggregated_model = average_models_aggregation(deepcopy(aggregated_model), clients_params)
        server_loss, server_acc, correct, f1_score = loss_dataset(aggregated_model, server_data, loss_f)
        global_hist.append([server_loss, server_acc])

        loss_hist.append(server_loss)
        acc_hist.append(server_acc)
        f1_score_list.append(f1_score)

        print(
            f'\n Server Testing, Loss: {server_loss:.4f}, Accuracy: {server_acc:.2f}%, Correct predictions: {correct}/{len(server_data.dataset)}, F1_score: {f1_score[2]:.2f}')

        models_hist.append(clients_models)
        server_hist.append([tens_param.detach().cpu().numpy() for tens_param in list(aggregated_model.parameters())])
        aggregated_model.load_state_dict(aggregated_model.state_dict())
        lr *= decay
    print("\n ############### Training Completed #################\n")
    return loss_hist, acc_hist, server_acc, server_loss, models_hist, server_hist, client_hist, global_hist, f1_score_list


def FedProx(model, model_param,percent_of_stragglers, training_sets: list, n_iter: int, testing_sets: list,
            server_data: list, mu=0, epochs=5,
            lr=10 ** -2, decay=1, noSysHeter=True):
    loss_f = loss_classifier
    K = len(training_sets)

    weights = [1 / K for i in range(K)]
    client_hist = {i: [] for i in range(K)}
    global_hist = []
    f1_score_list = []
    print("\n ############### FedProx #################")
    print(f"No. of clients: {K}")
    print(f"Rounds: {n_iter}, Epochs: {epochs}, lr: {lr}, decay: {decay}")
    # print(
    #     f"Server testing samples : {len(server_data.dataset)}, No. of labels: {len(server_data.dataset.tensors[1].unique())}")
    for i in range(K):
        print(
            f"Client {i}\t\t Training samples: {len(training_sets[i].dataset)}, Testing samples: {len(testing_sets[i].dataset)}, No. of labels: {len(training_sets[i].dataset.tensors[1].unique())}")

    loss_hist, acc_hist, server_hist, models_hist = [], [], [], []

    server_loss, server_acc, correct, f1_score = loss_dataset(model, server_data, loss_f)
    print(
        f'\n Initial Server Testing, Loss: {server_loss:.4f}, Accuracy: {server_acc:.2f}%, Correct predictions: {correct}/{len(server_data.dataset)}, F1_score: {f1_score[2]:.2f}')

    aggregated_model = client_node_ANN(num_classes=model_param[0], input_size=model_param[1]).to(device)
    for i in range(n_iter):
        print(
            "\n\n====================================================================================================")
        print(f"Round: {i + 1}")
        clients_params, clients_losses, clients_models = [], [], []

        for k in range(K):
            print(
                f"\n{k}: Client {k} Starts, Training size: {len(training_sets[k].dataset)}, Testing size: {len(testing_sets[k].dataset)}")

            local_model = deepcopy(aggregated_model).to(device)

            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
            local_model, local_loss, local_acc, local_hist, systemHeterogeneity = local_learning(local_model,
                                                                                                 percent_of_stragglers,
                                                                                                 mu,
                                                                                                 local_optimizer
                                                                                                 , training_sets[k],
                                                                                                 testing_sets[k],
                                                                                                 epochs,
                                                                                                 loss_f, "FedProx",
                                                                                                 True, noSysHeter)

            clients_losses.append(local_loss)
            client_hist[k].append(local_hist)

            list_params = list(local_model.parameters())
            list_params = [tens_param.detach().cpu() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))

        # aggregated_model = weighted_average_models(deepcopy(model), clients_params, weights=weights)
        aggregated_model = average_models_aggregation(deepcopy(aggregated_model), clients_params)
        server_loss, server_acc, correct, f1_score = loss_dataset(aggregated_model, server_data, loss_f)
        global_hist.append([server_loss, server_acc])

        loss_hist.append(server_loss)
        acc_hist.append(server_acc)
        f1_score_list.append(f1_score)
        # aggregated_model.load_state_dict(aggregated_model.state_dict())
        print(
            f'\n Server Testing, Loss: {server_loss:.4f}, Accuracy: {server_acc:.2f}%, Correct predictions: {correct}/{len(server_data.dataset)}, F1_score: {f1_score[2]:.2f}')

        models_hist.append(clients_models)
        server_hist.append([tens_param.detach().cpu().numpy() for tens_param in list(aggregated_model.parameters())])
        aggregated_model.load_state_dict(aggregated_model.state_dict())
        lr *= decay
    print("\n ############### FedProx Completed #################\n")
    return loss_hist, acc_hist, server_acc, server_loss, models_hist, server_hist, client_hist, global_hist, f1_score_list


def Fed_KD(model, model_param, percent_of_stragglers, server_train_loader, KD_server_loader, training_sets: list,
           n_iter: int, testing_sets: list, server_data: list, mu=0, epochs=5,
           lr=10 ** -2, decay=1, noSysHeter=True):
    loss_f = loss_classifier
    K = len(training_sets)

    weights = [1 / K for i in range(K)]
    client_hist = {i: [] for i in range(K)}
    global_hist = []
    f1_score_list = []

    print("\n ############### FedKD-Prox #################")
    print(f"No. of clients: {K}")
    print(f"Rounds: {n_iter}, Epochs: {epochs}, lr: {lr}, decay: {decay}")
    # print(
    #     f"Server testing samples : {len(server_data.dataset)}, No. of labels: {len(server_data.dataset.tensors[1].unique())}")
    for i in range(K):
        print(
            f"Client {i}\t\t Training samples: {len(training_sets[i].dataset)}, Testing samples: {len(testing_sets[i].dataset)}, No. of labels: {len(training_sets[i].dataset.tensors[1].unique())}")

    loss_hist, acc_hist, server_hist, models_hist = [], [], [], []

    server_loss, server_acc, correct, f1_score = loss_dataset(model, server_data, loss_f)
    print(
        f'\n Initial Server Testing, Loss: {server_loss:.4f}, Accuracy: {server_acc:.2f}%, Correct predictions: {correct}/{len(server_data.dataset)}, F1_score: {f1_score[2]:.2f}')

    outliers_client = [[], []]
    mu_dict = {}
    for i in range(K):
        mu_dict[i] = mu

    aggregated_model = client_node_ANN(num_classes=model_param[0], input_size=model_param[1]).to(device)
    KD_server_complex_model = ServerANN(num_classes=model_param[0], input_size=model_param[1]).to(device)
    client_initial_model = client_node_ANN(num_classes=model_param[0], input_size=model_param[1]).to(device)
    client_mode_lists = []

    for k in range(K):
        client_mode_lists.append(deepcopy(client_initial_model))

    server_teacher_logits, server_trained_model = server_training_for_knowledge_extraction(KD_server_complex_model,
                                                                                           server_train_loader,
                                                                                           KD_server_loader)
    for i in range(n_iter):

        print(
            "\n\n====================================================================================================")
        print(f"Round: {i + 1}")
        clients_params, clients_losses, clients_models, = [], [], []

        if i == 0:

            for k in range(K):
                print(
                    f"\n{k}: Client {k} Starts, Training size: {len(training_sets[k].dataset)}, Testing size: {len(testing_sets[k].dataset)}")

                local_model = client_mode_lists[k]
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
                local_model_trained, local_loss, local_acc, local_hist, systemHeterogeneity = local_learning(
                    local_model, percent_of_stragglers, mu, local_optimizer
                    , training_sets[k], testing_sets[k], epochs,
                    loss_f, "FedProx", True, noSysHeter)

                clients_losses.append(local_loss)
                client_hist[k].append(local_hist)

                list_params = list(local_model_trained.parameters())
                list_params = [tens_param.detach().cpu() for tens_param in list_params]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model_trained))
                client_mode_lists[k] = deepcopy(local_model_trained)

            aggregated_model = average_models_aggregation(deepcopy(aggregated_model), clients_params)
            server_loss, server_acc, correct, f1_score = loss_dataset(aggregated_model, server_data, loss_f)
            global_hist.append([server_loss, server_acc])

            loss_hist.append(server_loss)
            acc_hist.append(server_acc)
            # aggregated_model.load_state_dict(aggregated_model.state_dict())
            f1_score_list.append(f1_score)
            print(
                f'\n Server Testing, Loss: {server_loss:.4f}, Accuracy: {server_acc:.2f}%, Correct predictions: {correct}/{len(server_data.dataset)}, F1_score: {f1_score[2]:.2f}')

        else:

            for k in range(K):
                print(
                    f"\n{k}: Client {k} Starts, Training size: {len(training_sets[k].dataset)}, Testing size: {len(testing_sets[k].dataset)}")

                if k in outliers_client[0]:
                    mu_dict[k] = mu_dict[k] * 2

                elif k in outliers_client[1]:
                    mu_dict[k] = mu_dict[k] / 2

                student_logits, client_model_kd, local_loss, local_acc, local_hist, systemHeterogeneity = local_learning_kd(
                    client_mode_lists[k], percent_of_stragglers, mu_dict[k], lr,
                    training_sets[k], server_teacher_logits, testing_sets[k], epochs,
                    loss_f, noSysHeter)

                clients_losses.append(local_loss)
                client_hist[k].append(local_hist)
                clients_params.append(student_logits)
                client_mode_lists[k] = client_model_kd

            server_trained_model, outliers_client, correct, server_acc, server_loss, f1_score = kd_model_aggregation(
                server_trained_model, server_train_loader, KD_server_loader, clients_params)
            global_hist.append([server_loss, server_acc])

            loss_hist.append(server_loss)
            acc_hist.append(server_acc)
            f1_score_list.append(f1_score)

            print(
                f'\n Server Testing, Loss: {server_loss:.4f}, Accuracy: {server_acc:.2f}%, F1_score: {f1_score[2]:.2f}')

        if i == 0:
            aggregated_model.load_state_dict(aggregated_model.state_dict())
            for serverTeacher_features, serverTeacher_labels in KD_server_loader:
                if serverTeacher_features.size(0) != 128:
                    continue
                serverTeacher_features, serverTeacher_labels = serverTeacher_features.to(
                    device), serverTeacher_labels.to(device)
                server_teacher_logits = aggregated_model(serverTeacher_features)
            server_teacher_logits = server_teacher_logits.detach()
        else:
            server_teacher_logits, server_trained_model = server_training_for_knowledge_extraction(server_trained_model,
                                                                                                   server_train_loader,
                                                                                                   KD_server_loader)
    lr *= decay

    print("\n ############### FedKD-Prox Completed #################\n")
    return loss_hist, acc_hist, server_acc, server_loss, models_hist, server_hist, client_hist, global_hist, f1_score_list
