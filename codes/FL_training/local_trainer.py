from codes.utilites.utilites import *
from codes.utilites.model import *


def train_step(model, model_0, mu: float, optimizer, train_data, valid_data, loss_f):
    tr_loss, correct_tr = 0, 0
    val_loss, val_correct = 0, 0
    total_tr, val_total = 0, 0
    TP, FP, FN = 0, 0, 0
    model.train()


    for tr_features, tr_labels in train_data:
        optimizer.zero_grad()

        tr_features, tr_labels = tr_features.to(device), tr_labels.to(device)
        tr_predictions = model(tr_features)
        tr_batch_loss = loss_f(tr_predictions, tr_labels)
        tr_prox_term = (mu / 2) * difference_models_norm_2(model, model_0)
        loss = tr_batch_loss + tr_prox_term
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, tr_predicted = torch.max(tr_predictions.data, 1)
        total_tr += tr_labels.size(0)

        correct_tr += tr_predicted.eq(tr_labels.view_as(tr_predicted)).sum().item()


    val_model = deepcopy(model).to(device)
    val_model.eval()

    with torch.no_grad():
        for val_features, val_labels in valid_data:
            val_features, val_labels = val_features.to(device), val_labels.to(device)
            val_predictions = val_model(val_features)
            val_batch_loss = loss_f(val_predictions, val_labels)
            val_loss += val_batch_loss.item()

            _, val_predicted = torch.max(val_predictions.data, 1)
            val_total += val_labels.size(0)

            val_correct += val_predicted.eq(val_labels.view_as(val_predicted)).sum().item()

            for i in range(val_labels.size(0)):
                if val_predicted.eq(val_labels.view_as(val_predicted))[i].item() == 1:
                    TP += 1
                elif val_predicted.eq(val_labels.view_as(val_predicted))[i].item() == 0:
                    FP += 1
                elif val_predicted.eq(val_labels.view_as(val_predicted))[i].item() == 0 and val_labels[
                    i].item() == 1:
                    FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    tr_loss = tr_loss / len(train_data)
    tr_accuracy = 100 * correct_tr / total_tr
    val_loss = val_loss / len(valid_data)
    val_accuracy = 100 * val_correct / val_total
    return model, tr_loss, tr_accuracy, val_loss, val_accuracy, F1_score



def local_learning(model, percent_of_stragglers,mu: float, optimizer, train_data, test_data, epochs: int, loss_f, algorithm,
                   first_time_learning=True, noSysHeter=True):
    valid_loss, valid_accuracy = 0, 0
    total = 0
    model_0 = deepcopy(model).to(device)
    best_model = deepcopy(model).to(device)

    tr_loss_list, tr_acc_list, val_loss_list, val_acc_list = [], [], [], []

    random_chance = random.randint(0, 10)

    if noSysHeter:
        systemHeterogeneity = epochs

        for e in range(systemHeterogeneity):
            trained_model, local_loss, accuracy, valid_loss, valid_accuracy, f1_score = train_step(model, model_0,
                                                                                                   mu, optimizer,
                                                                                                   train_data,
                                                                                                   test_data,
                                                                                                   loss_f,
                                                                                                   )
            tr_loss_list.append(local_loss)
            tr_acc_list.append(accuracy)
            val_loss_list.append(valid_loss)
            val_acc_list.append(valid_accuracy)

            ep = "0" + str(e + 1) if e + 1 < 10 else str(e + 1)
            print(
                f'Epoch {ep}\t Tr_Loss: {local_loss:.4f}\t Tr_Acc: {accuracy:.2f}%\t Val_Loss: {valid_loss:.4f}\t Val_Acc: {valid_accuracy:.2f}%\t, F1_score: {f1_score:.2f}')

        return trained_model, valid_loss, valid_accuracy, [tr_loss_list, tr_acc_list, val_loss_list,
                                                           val_acc_list], systemHeterogeneity



    else:
        if random_chance <= percent_of_stragglers:
            systemHeterogeneity = random.randint(1, 20)
        else:
            systemHeterogeneity = epochs

        if systemHeterogeneity < 13 and algorithm == "FedAvg":
            print("client dropped")
            return best_model, valid_loss, valid_accuracy, [tr_loss_list, tr_acc_list, val_loss_list,
                                                            val_acc_list], systemHeterogeneity

        else:


            for e in range(systemHeterogeneity):
                trained_model, local_loss, accuracy, valid_loss, valid_accuracy, F1_score = train_step(model,
                                                                                                       model_0, mu,
                                                                                                       optimizer,
                                                                                                       train_data,
                                                                                                       test_data,
                                                                                                       loss_f,
                                                                                                       )
                tr_loss_list.append(local_loss)
                tr_acc_list.append(accuracy)
                val_loss_list.append(valid_loss)
                val_acc_list.append(valid_accuracy)

                # best_model = deepcopy(model).to(device)
                ep = "0" + str(e + 1) if e + 1 < 10 else str(e + 1)
                print(
                    f'Epoch {ep}\t Tr_Loss: {local_loss:.4f}\t Tr_Acc: {accuracy:.2f}%\t Val_Loss: {valid_loss:.4f}\t Val_Acc: {valid_accuracy:.2f}%\t F1_score: {F1_score:.2f}')

            return trained_model, valid_loss, valid_accuracy, [tr_loss_list, tr_acc_list, val_loss_list,
                                                               val_acc_list], systemHeterogeneity




def train_step_kd(model, mu: float, optimizer, train_data, valid_data, loss_f, server_teacher_logits):
    tr_loss, correct_tr = 0, 0
    total_tr, val_total = 0, 0
    TP, FP, FN = 0, 0, 0
    model.train()

    # client_teacher_logits = torch.empty(0)
    soft_target_loss_weight = 0.25
    ce_loss_weight = 0.75
    model.train()

    for tr_features, tr_labels in train_data:

        optimizer.zero_grad()
        tr_features, tr_labels = tr_features.to(device), tr_labels.to(device)
        if tr_features.size(0) != 128:
            continue
        client_student_logits = model(tr_features)

        soften_targets = nn.functional.softmax(server_teacher_logits / 2, dim=-1)
        soft_prob = nn.functional.log_softmax(client_student_logits / 2, dim=-1)

        soft_targets_loss = -torch.sum(soften_targets * soft_prob) / soft_prob.size()[0] * (2 ** 2)
        label_loss = loss_f(client_student_logits, tr_labels)

        s = torch.norm(server_teacher_logits.to(device) - client_student_logits.to(device), p=2)
        tr_prox_term = (mu / 2) * s

        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * (tr_prox_term + label_loss)

        loss.backward(retain_graph=True)
        optimizer.step()
        tr_loss += loss.item()

        _, tr_predicted = torch.max(client_student_logits.data, 1)
        total_tr += tr_labels.size(0)
        correct_tr += tr_predicted.eq(tr_labels.view_as(tr_predicted)).sum().item()
        for i in range(tr_labels.size(0)):
            if tr_predicted.eq(tr_labels.view_as(tr_predicted))[i].item() == 1:
                TP += 1
            elif tr_predicted.eq(tr_labels.view_as(tr_predicted))[i].item() == 0:
                FP += 1
            elif tr_predicted.eq(tr_labels.view_as(tr_predicted))[i].item() == 0 and tr_labels[i].item() == 1:
                FN += 1
    # print(tr_prox_term, mu)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    tr_loss = tr_loss / len(train_data)
    training_accuracy = 100 * correct_tr / total_tr

    total_params = sum(p.nonzero().size(0) for p in model.parameters())

    returned_model = deepcopy(model).to(device)
    val_model = deepcopy(model).to(device)
    val_model.eval()

    for val_features, val_labels in valid_data:
        if val_features.size(0) != 128:
            continue
        val_features, val_labels = val_features.to(device), val_labels.to(device)
        client_teacher_logits = val_model(val_features)

    return client_teacher_logits, total_params, tr_loss, training_accuracy, returned_model, F1_score


def local_learning_kd(client_model_to_train, percent_of_stragglers,mu: float, lr, train_data, server_teacher_logits, test_data, epochs: int,
                      loss_f, noSysHeter=True):
    valid_loss, valid_accuracy = 0, 0
    total = 0

    optimizer_kd = optim.SGD(client_model_to_train.parameters(), lr=lr)
    tr_loss_list, tr_acc_list, val_loss_list, val_acc_list = [], [], [], []

    random_chance = random.randint(0, 10)

    if noSysHeter:
        systemHeterogeneity = epochs
        val_correct = 0
        model_size = 0
        for e in range(systemHeterogeneity):

            student_logits, model_size, training_loss, training_accuracy, returned_model, f1_score = train_step_kd(
                client_model_to_train, mu, optimizer_kd, train_data, test_data, loss_f, server_teacher_logits)

            returned_model.eval()
            with torch.no_grad():
                for val_features, val_labels in test_data:
                    val_features, val_labels = val_features.to(device), val_labels.to(device)
                    val_predictions = returned_model(val_features)
                    val_batch_loss = loss_f(val_predictions, val_labels)
                    valid_loss += val_batch_loss.item()

                    _, val_predicted = torch.max(val_predictions.data, 1)
                    total += val_labels.size(0)
                    val_correct += val_predicted.eq(val_labels.view_as(val_predicted)).sum().item()

            valid_loss = valid_loss / len(test_data)
            valid_accuracy = 100 * val_correct / total
            ep = "0" + str(e + 1) if e + 1 < 10 else str(e + 1)
            print(
                f'Epoch {ep}\t Tr_Loss: {training_loss:.4f}\t Tr_Acc: {training_accuracy:.2f}%\t Val_Loss: {valid_loss:.4f}\t Val_Acc: {valid_accuracy:.2f}%\t, F1_score: {f1_score:.2f}, mu: {mu}')

        print("Client updates size before KD: ", model_size)
        print("Client updates size after KD: ", student_logits.numel())

        val_loss_list.append(valid_loss)
        val_acc_list.append(valid_accuracy)

        return student_logits, returned_model, valid_loss, valid_accuracy, [tr_loss_list, tr_acc_list, val_loss_list,
                                                                            val_acc_list], systemHeterogeneity

    else:
        if random_chance <= percent_of_stragglers:
            systemHeterogeneity = random.randint(1, 20)
        else:
            systemHeterogeneity = epochs

        # systemHeterogeneity = epochs
        val_correct = 0
        model_size = 0
        for e in range(systemHeterogeneity):

            student_logits, model_size, training_loss, training_accuracy, returned_model, f1_score = train_step_kd(
                client_model_to_train, mu, optimizer_kd, train_data, test_data, loss_f, server_teacher_logits)

            returned_model.eval()
            with torch.no_grad():
                for val_features, val_labels in test_data:
                    val_features, val_labels = val_features.to(device), val_labels.to(device)
                    val_predictions = returned_model(val_features)
                    val_batch_loss = loss_f(val_predictions, val_labels)
                    valid_loss += val_batch_loss.item()

                    _, val_predicted = torch.max(val_predictions.data, 1)
                    total += val_labels.size(0)
                    val_correct += val_predicted.eq(val_labels.view_as(val_predicted)).sum().item()

            valid_loss = valid_loss / len(test_data)
            valid_accuracy = 100 * val_correct / total
            ep = "0" + str(e + 1) if e + 1 < 10 else str(e + 1)
            print(
                f'Epoch {ep}\t Tr_Loss: {training_loss:.4f}\t Tr_Acc: {training_accuracy:.2f}%\t Val_Loss: {valid_loss:.4f}\t Val_Acc: {valid_accuracy:.2f}%\t, F1_score: {f1_score:.2f}, mu: {mu}')

        print("Client updates size before KD: ", model_size)
        print("Client updates size after KD: ", student_logits.numel())

        val_loss_list.append(valid_loss)
        val_acc_list.append(valid_accuracy)

        return student_logits, returned_model, valid_loss, valid_accuracy, [tr_loss_list, tr_acc_list, val_loss_list,
                                                                            val_acc_list], systemHeterogeneity