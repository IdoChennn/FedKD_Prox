from codes.utilites.requirments import *


def prep_data(df_list, labels):
    preprocessed_data = []
    encoder = LabelEncoder()
    encoder.fit(labels)
    for client_df in df_list:
        df = client_df.sample(frac=1).reset_index(drop=True)

        x, y = df.drop(columns=['label']).values, encoder.transform(df['label'])
        standard_scaler = StandardScaler()
        x = standard_scaler.fit_transform(x)
        print(f"Client data points: {len(x)}, Client labels: {len(set(y))}")

        preprocessed_data.append([x, y])
    return preprocessed_data


def set_to_zero_model_weights(model):
    for layer_weights in model.parameters():
        layer_weights.data.sub_(layer_weights.data)


def loss_classifier(predictions, labels):
    loss_function = nn.CrossEntropyLoss()
    return loss_function(predictions, labels.view(-1))


def loss_dataset(model, dataset, loss_f):
    loss = 0
    correct = 0
    total = 0
    TP, FP, FN = 0, 0, 0
    model_val = deepcopy(model).to(device)
    model_val.eval()
    for features, labels in dataset:
        features, labels = features.to(device), labels.to(device)
        predictions = model_val(features)
        batch_loss = loss_f(predictions, labels)
        loss += batch_loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()

        for i in range(labels.size(0)):
            if predicted.eq(labels.view_as(predicted))[i].item() == 1:
                TP += 1
            elif predicted.eq(labels.view_as(predicted))[i].item() == 0:
                FP += 1
            elif predicted.eq(labels.view_as(predicted))[i].item() == 0 and labels[i].item() == 1:
                FN += 1

    loss = loss / len(dataset)
    accuracy = 100 * correct / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return loss, accuracy, correct, [precision, recall, F1_score]


def difference_models_norm_2(model, model_0):
    tensor_1 = list(model.parameters())
    tensor_2 = list(model_0.parameters())
    sub_norm = []
    for i in range(len(tensor_1)):
        s = torch.norm(tensor_1[i].to(device) - tensor_2[i].to(device), p=2)
        sub_norm.append(s)
    return sum(sub_norm)


def calculate_model_byte_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        total_size += buffer.nelement() * buffer.element_size()
    return total_size


def server_training_for_knowledge_extraction(server_model, server_TrainingData, server_KDData):
    loss_f = loss_classifier
    optimizer = optim.SGD(server_model.parameters(), lr=0.01)

    server_model.train()
    for inputs, labels in server_TrainingData:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = server_model(inputs)
        batch_loss = loss_f(predictions, labels)
        batch_loss.backward()
        optimizer.step()

    server_model.eval()
    with torch.no_grad():
        for features, labels in server_KDData:
            features, labels = features.to(device), labels.to(device)
            if features.size(0) != 128:
                continue
            logits = server_model(features)

    return logits, server_model


def save_logs(results_no_heter, results_heter):
    with open("./Results(No heterogeneity)", "w") as outfile:
        json.dump(results_no_heter, outfile)

    with open("./Results(Heterogeneity)", "w") as outfile:
        json.dump(results_heter, outfile)
