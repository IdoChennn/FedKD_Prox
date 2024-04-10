from codes.utilites.requirments import *


def IID_data_partitioning(iid_data_set):
    server_split_iid = 0.2
    testing_split_iid = 0.2

    print("\nIID data Partitioning for FL clients...\n")

    clients_data_iid = []

    client_x_iid_train = []
    client_y_iid_train = []

    client_x_iid_test = []
    client_y_iid_test = []

    for class_data in tqdm(iid_data_set, desc="prepare client training data ... "):
        x_client_iid = class_data[0][int(len(class_data[0]) * server_split_iid):]
        x_client_iid = x_client_iid[:int(len(x_client_iid) * (1 - testing_split_iid))]
        y_client_iid = class_data[1][int(len(class_data[1]) * server_split_iid):]
        y_client_iid = y_client_iid[:int(len(y_client_iid) * (1 - testing_split_iid))]

        # print(y_client_iid)
        client_x_iid_train.append(x_client_iid)
        client_y_iid_train.append(y_client_iid)

        x_client_test_iid = class_data[0][int(len(class_data[0]) * server_split_iid):]
        x_client_test_iid = x_client_test_iid[int(len(x_client_test_iid) * (1 - testing_split_iid)):]
        y_client_test_iid = class_data[1][int(len(class_data[1]) * server_split_iid):]
        y_client_test_iid = y_client_test_iid[int(len(y_client_test_iid) * (1 - testing_split_iid)):]
        client_x_iid_test.append(x_client_test_iid)
        client_y_iid_test.append(y_client_test_iid)

    for items in tqdm(range(len(client_x_iid_train)), desc="Transform to tensor ..."):
        x_client_iid = client_x_iid_train[items]
        y_client_iid = client_y_iid_train[items]

        idx_iid = np.random.permutation(len(x_client_iid))
        x_client_iid = x_client_iid[idx_iid]
        y_client_iid = y_client_iid[idx_iid]

        x_client_test_iid = client_x_iid_test[items]
        y_client_test_iid = client_y_iid_test[items]
        idx_iid = np.random.permutation(len(x_client_test_iid))
        x_client_test_iid = x_client_test_iid[idx_iid]
        y_client_test_iid = y_client_test_iid[idx_iid]

        tr_iid = TensorDataset(torch.tensor(x_client_iid, dtype=torch.float32),
                               torch.tensor(y_client_iid, dtype=torch.float32).reshape(-1, 1).long())

        te_iid = TensorDataset(torch.tensor(x_client_test_iid, dtype=torch.float32),
                               torch.tensor(y_client_test_iid, dtype=torch.float32).reshape(-1, 1).long())

        clients_data_iid.append(
            [DataLoader(tr_iid, batch_size=128, shuffle=True), DataLoader(te_iid, batch_size=128, shuffle=True)])

    training_sets_iid = [dl[0] for dl in clients_data_iid]
    testing_sets_iid = [dl[1] for dl in clients_data_iid]

    print("IID data partitioning done\n")

    return training_sets_iid, testing_sets_iid


def Non_IID_data_partitioning(niid_data_set):
    server_split_iid = 0.2
    testing_split_iid = 0.2
    print("\nNIID Partitioning x and y for FL...\n")
    clients_data_niid = []

    client_x_niid_train = []
    client_y_niid_train = []

    client_x_niid_test = []
    client_y_niid_test = []

    for class_data in tqdm(niid_data_set, desc="prepare client training data ... "):
        x_client_niid = class_data[0][int(len(class_data[0]) * server_split_iid):]
        x_client_niid = x_client_niid[:int(len(x_client_niid) * (1 - testing_split_iid))]
        y_client_niid = class_data[1][int(len(class_data[1]) * server_split_iid):]
        y_client_niid = y_client_niid[:int(len(y_client_niid) * (1 - testing_split_iid))]

        # print(y_client_iid)
        client_x_niid_train.append(x_client_niid)
        client_y_niid_train.append(y_client_niid)

        x_client_test_niid = class_data[0][int(len(class_data[0]) * server_split_iid):]
        x_client_test_niid = x_client_test_niid[int(len(x_client_test_niid) * (1 - testing_split_iid)):]
        y_client_test_niid = class_data[1][int(len(class_data[1]) * server_split_iid):]
        y_client_test_niid = y_client_test_niid[int(len(y_client_test_niid) * (1 - testing_split_iid)):]
        client_x_niid_test.append(x_client_test_niid)
        client_y_niid_test.append(y_client_test_niid)

    for items in tqdm(range(len(client_x_niid_train)), desc="Transform to tensor ..."):
        x_client_niid = client_x_niid_train[items]
        y_client_niid = client_y_niid_train[items]

        idx_niid = np.random.permutation(len(x_client_niid))
        x_client_niid = x_client_niid[idx_niid]
        y_client_niid = y_client_niid[idx_niid]

        x_client_test_niid = client_x_niid_test[items]
        y_client_test_niid = client_y_niid_test[items]
        idx_niid = np.random.permutation(len(x_client_test_niid))
        x_client_test_niid = x_client_test_niid[idx_niid]
        y_client_test_niid = y_client_test_niid[idx_niid]

        tr_iid = TensorDataset(torch.tensor(x_client_niid, dtype=torch.float32),
                               torch.tensor(y_client_niid, dtype=torch.float32).reshape(-1, 1).long())

        te_iid = TensorDataset(torch.tensor(x_client_test_niid, dtype=torch.float32),
                               torch.tensor(y_client_test_niid, dtype=torch.float32).reshape(-1, 1).long())

        clients_data_niid.append(
            [DataLoader(tr_iid, batch_size=128, shuffle=True), DataLoader(te_iid, batch_size=128, shuffle=True)])

    training_sets_niid = [dl[0] for dl in clients_data_niid]
    testing_sets_niid = [dl[1] for dl in clients_data_niid]

    print("NIID data partitioning done\n")

    return training_sets_niid, testing_sets_niid

