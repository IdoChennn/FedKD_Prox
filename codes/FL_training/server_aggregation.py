from codes.utilites.utilites import *


def weighted_average_models_aggregation(model, clients_models_hist: list, weights: list):
    new_model = deepcopy(model).to(device)
    set_to_zero_model_weights(new_model)
    for k, client_hist in enumerate(clients_models_hist):
        for idx, layer_weights in enumerate(new_model.parameters()):
            contribution = client_hist[idx].data.to(device) * weights[k]
            layer_weights.data.add_(contribution)

    return new_model


def average_models_aggregation(model, clients_models_hist: list):
    new_model = deepcopy(model).to(device)
    set_to_zero_model_weights(new_model)

    total_clients = len(clients_models_hist)
    dropped_client = 0
    for client in clients_models_hist:
        if len(client) == 0:
            dropped_client += 1
    total_clients = total_clients - dropped_client

    if total_clients == 0:
        return new_model

    print(f"Total clients: {total_clients}")
    for idx, layer_weights in enumerate(new_model.parameters()):
        layer_sum = sum(client_hist[idx].data.to(device) for client_hist in clients_models_hist)
        average_layer = layer_sum / total_clients
        layer_weights.data.add_(average_layer)

    return new_model


def kd_model_aggregation(aggregated_model, server_train_loader, server_test_loader, clients_models_hist: list):
    aggregated_model.train()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(aggregated_model.parameters(), lr=0.01)
    T = 2
    soft_target_loss_weight = 0.5
    ce_loss_weight = 0.5
    running_loss = 0.0

    aggregated_logits = []
    for client_teacher_logits in clients_models_hist:
        logits_tensor = client_teacher_logits.float()
        aggregated_logits.append(logits_tensor)

    stacked_logits = torch.stack(aggregated_logits, dim=0)
    teacher_logits = torch.mean(stacked_logits, dim=0)

    # Calculating distances from the mean logits
    distances = []
    for client_logits in aggregated_logits:
        distance = torch.norm(client_logits - teacher_logits, p=2)  # Euclidean distance
        distances.append(distance.item())

    distances = np.array(distances)
    Q1, Q3 = np.percentile(distances, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices_above = []
    outlier_indices_below = []

    for i, distance in enumerate(distances):
        if distance > upper_bound:
            # Client's deviation is above the upper bound
            outlier_indices_above.append(i)
        elif distance < lower_bound:
            # Client's deviation is below the lower bound
            outlier_indices_below.append(i)
    print(f"Outlier indices above: {outlier_indices_above}")
    print(f"Outlier indices below: {outlier_indices_below}")

    for inputs, labels in server_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.size(0) != 128:
            continue

        optimizer.zero_grad()

        student_logits = aggregated_model(inputs)
        # Soften the student logits by applying softmax first and log() second

        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

        # Calculate the soft target loss. Scaled by T**2 as suggested by the authors of the paper
        # "Distilling the knowledge in a neural network"
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T ** 2)

        # Calculate the true label loss
        label_loss = ce_loss(student_logits, labels.view(-1))

        # Weighted sum of the two losses
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += loss.item()

    loss = 0
    correct = 0
    TP, FP, FN = 0, 0, 0
    aggregated_model.eval()
    with torch.no_grad():
        for features, labels in server_test_loader:
            features, labels = features.to(device), labels.to(device)
            predictions = aggregated_model(features)
            batch_loss = ce_loss(predictions, labels.view(-1))
            loss += batch_loss.item()
            _, predicted = torch.max(predictions.data, 1)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()

            for i in range(labels.size(0)):
                if predicted.eq(labels.view_as(predicted))[i].item() == 1:
                    TP += 1
                elif predicted.eq(labels.view_as(predicted))[i].item() == 0:
                    FP += 1
                elif predicted.eq(labels.view_as(predicted))[i].item() == 0 and labels[i].item() == 1:
                    FN += 1

    loss /= len(server_test_loader)
    accuracy = 100 * correct / len(server_test_loader.dataset)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return aggregated_model, [outlier_indices_above, outlier_indices_below], correct, accuracy, loss, [precision,
                                                                                                       recall, F1_score]