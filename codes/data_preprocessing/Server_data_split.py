from codes.utilites.requirments import *


def obtain_server_data(loaded_data):
    server_split_iid = 0.2

    ### Server training and testing data split ###
    server_test_x = []
    server_test_y = []
    print("\nServer data preparation for FL...\n")
    for class_data in tqdm(loaded_data, desc="prepare global model testing data ... "):
        x_global_iid = class_data[0][:int(len(class_data[0]) * server_split_iid)].copy()
        y_global_iid = class_data[1][:int(len(class_data[1]) * server_split_iid)].copy()
        server_test_x.append(x_global_iid)
        server_test_y.append(y_global_iid)

    server_test_iid_x = np.concatenate(server_test_x)
    server_test_iid_y = np.concatenate(server_test_y)

    idx_iid = np.random.permutation(len(server_test_iid_x))
    server_test_iid_x = server_test_iid_x[idx_iid]
    server_test_iid_y = server_test_iid_y[idx_iid]

    ### KD Server data split ###
    x_server_test_tensor = torch.tensor(server_test_iid_x, dtype=torch.float32)
    y_server_test_tensor = torch.tensor(server_test_iid_y, dtype=torch.float32).reshape(-1, 1).long()
    server_dataset = TensorDataset(x_server_test_tensor, y_server_test_tensor)

    # Calculate the sizes of splits
    total_size = len(server_dataset)
    train_size = int(0.6 * total_size)
    test_size = total_size - train_size

    # Randomly split the dataset into train and test
    train_dataset, test_dataset = random_split(server_dataset, [train_size, test_size])
    Kd_size = int(0.2 * test_size)
    actual_test_size = test_size - Kd_size
    actual_test_dataset, KD_testing_dataset = random_split(test_dataset, [actual_test_size, Kd_size])

    # Create DataLoaders for train and test sets
    server_train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    server_test_loader = DataLoader(actual_test_dataset, batch_size=128, shuffle=True)
    KD_server_loader = DataLoader(KD_testing_dataset, batch_size=128, shuffle=True)

    unique_labels = set()
    for _, label in actual_test_dataset:
        unique_labels.add(label.item())

    print("------- Global model Testing Data points:", len(test_dataset))

    unique_labels = set()
    for _, label in train_dataset:
        unique_labels.add(label.item())

    print("-------- Server Training Data points for server teacher model:", len(train_dataset))

    unique_labels = set()
    for _, label in KD_testing_dataset:
        unique_labels.add(label.item())

    print("--------- Server Data points for server knowledge extraction", len(KD_testing_dataset))

    return server_train_loader, server_test_loader, KD_server_loader
