from codes.data_loader.data_loader import *
from codes.data_preprocessing.Server_data_split import *
from codes.data_preprocessing.client_data_partitioning import *
from codes.FL_training.train import *
from codes.utilites.plotting_results import *
import argparse


def main(dataset_name, n_iter, epochs, lr, decay, mu, percent_of_stragglers):
    if dataset_name == "CICIDS2017":

        dataset, y_column, all_labels, class_feature_number = IDS()

    elif dataset_name == "CICIoMT2024":
        dataset, y_column, all_labels, class_feature_number = IoMT()

    else:

        dataset, y_column, all_labels, class_feature_number = IoT()

    iid_data, niid_data = loading_dataset(10, dataset, y_column, all_labels)
    server_train_data, server_test_data, KD_server_data = obtain_server_data(iid_data)
    training_sets_iid, testing_sets_iid = IID_data_partitioning(iid_data)
    training_sets_niid, testing_sets_niid = Non_IID_data_partitioning(niid_data)

    print("\n################### Training without statistical heterogeneity... ###################\n")
    results_without_heterogeneity = training_without_heterogeneity(n_iter, class_feature_number, training_sets_iid,
                                                                   testing_sets_iid, server_test_data,
                                                                   percent_of_stragglers, mu, epochs, lr, decay,
                                                                   server_train_data,
                                                                   KD_server_data)

    print("\n################### Training with statistical heterogeneity... ###################\n")
    results_with_heterogeneity = training_with_heterogeneity(n_iter, class_feature_number, training_sets_niid,
                                                             testing_sets_niid, server_test_data,
                                                             percent_of_stragglers,
                                                             mu, epochs, lr, decay, server_train_data,
                                                             KD_server_data)

    save_logs(results_without_heterogeneity, results_with_heterogeneity)
    plot_results(results_without_heterogeneity, results_with_heterogeneity, n_iter,dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Training Parameters')
    parser.add_argument('--dataset', type=str, default="CICIoT2023", help='select dataset')
    parser.add_argument('--n_iter', type=int, default=30, help='Number of iterations')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0.9, help='Decay rate')
    parser.add_argument('--mu', type=float, default=0.1, help='Momentum')
    parser.add_argument('--percent_of_stragglers', type=int, default=8, help='Percentage of stragglers')

    args = parser.parse_args()

    # Pass the command-line arguments to the main function
    main(args.dataset, args.n_iter, args.epochs, args.lr, args.decay, args.mu, args.percent_of_stragglers)
