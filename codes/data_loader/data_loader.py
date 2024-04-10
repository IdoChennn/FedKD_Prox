from codes.utilites.utilites import *


### IoT dataset ###

def IoT():
    DATASET_DIRECTORY = './data/CICIoT2023'
    X_columns = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
        'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
        'ece_flag_number', 'cwr_flag_number', 'ack_count',
        'syn_count', 'fin_count', 'urg_count', 'rst_count',
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
        'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
        'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
        'Radius', 'Covariance', 'Variance', 'Weight',
    ]

    ddos_labels = ['DDoS-RSTFINFlood',
                   'DDoS-PSHACK_Flood',
                   'DDoS-SYN_Flood', 'DDoS-UDP_Flood',
                   'DDoS-TCP_Flood', 'DDoS-ICMP_Flood',
                   'DDoS-SynonymousIP_Flood', 'DDoS-ACK_Fragmentation', 'DDoS-UDP_Fragmentation',
                   'DDoS-ICMP_Fragmentation', 'DDoS-SlowLoris', 'DDoS-HTTP_Flood']
    dos_labels = ['DoS-UDP_Flood', 'DoS-SYN_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood']
    mirai_labels = ['Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain']
    recon_labels = ['Recon-PingSweep', 'Recon-OSScan', 'Recon-PortScan', 'VulnerabilityScan', 'Recon-HostDiscovery']
    spoofing_labels = ['DNS_Spoofing', 'MITM-ArpSpoofing']
    web_labels = ['BrowserHijacking', 'Backdoor_Malware', 'XSS', 'Uploading_Attack', 'SqlInjection', 'CommandInjection']
    bruteforce_labels = ['DictionaryBruteForce']
    benign_labels = ['BenignTraffic']
    all_labels = ddos_labels + dos_labels + mirai_labels + recon_labels + spoofing_labels + web_labels + bruteforce_labels + benign_labels

    y_column = 'label'
    df_sets = [os.path.join(DATASET_DIRECTORY, k) for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')][0:4]
    df_sets.sort()
    dataset = []
    for i, csv_file in tqdm(enumerate(df_sets), total=len(df_sets)):
        raw_df = pd.read_csv(csv_file, usecols=X_columns + [y_column])

        dataset.append(raw_df)

    dataset = pd.concat(dataset, ignore_index=True)
    len_data = len(dataset)
    print(f"Total data before SMOTE: {len_data}")

    encoder = LabelEncoder()
    encoder.fit(all_labels)
    y_encoded = encoder.transform(dataset[y_column])
    class_counts = np.bincount(y_encoded)
    target_samples = 3000
    sampling_strategy = {class_label: target_samples for class_label, count in enumerate(class_counts) if
                         0 < count < target_samples}
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    x, y = smote.fit_resample(dataset.drop(columns=[y_column]), y_encoded)
    y = encoder.inverse_transform(y)

    dataset = pd.DataFrame(x, columns=X_columns)
    dataset[y_column] = y

    len_data = len(dataset[y_column])
    print(f"Total data after SMOTE: {len_data}")

    feature_class_number = [34, 46]
    return dataset, y_column, all_labels, feature_class_number


def IoMT():
    DATASET_DIRECTORY = './data/CICIoMT2024/'

    X_columns = ['Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate', 'fin_flag_number',
                 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
                 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS',
                 'Telnet',
                 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min',
                 'Max',
                 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
                 'label']

    all_labels = ['MQTT-DoS-Publish_Flood', 'TCP_IP-DDoS-UDP', 'TCP_IP-DoS-TCP',
                  'Recon-Ping_Sweep', 'TCP_IP-DDoS-TCP', 'TCP_IP-DoS-SYN', 'Recon-VulScan',
                  'TCP_IP-DoS-ICMP', 'TCP_IP-DoS-UDP', 'MQTT-DDoS-Publish_Flood',
                  'TCP_IP-DDoS-ICMP', 'MQTT-Malformed_Data', 'ARP_Spoofing',
                  'MQTT-DoS-Connect_Flood', 'TCP_IP-DDoS-SYN', 'Benign', 'Recon-Port_Scan',
                  'MQTT-DDoS-Connect_Flood', 'Recon-OS_Scan']

    y_column = 'label'

    df_sets = [os.path.join(DATASET_DIRECTORY, k) for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
    all_dfs = []
    for file_path in df_sets:
        raw_df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)
        label = re.sub(r'\d*_train\.pcap\.csv', '', file_name)
        df = raw_df.sample(frac=0.15, random_state=42)
        df['label'] = label
        all_dfs.append(df)

    dataset = pd.concat(all_dfs, ignore_index=True)
    len_data = len(dataset)
    print(f"Total data before SMOTE: {len_data}")

    encoder = LabelEncoder()
    encoder.fit(all_labels)
    y_encoded = encoder.transform(dataset[y_column])
    class_counts = np.bincount(y_encoded)
    target_samples = 3000
    sampling_strategy = {class_label: target_samples for class_label, count in enumerate(class_counts) if
                         0 < count < target_samples}
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    x, y = smote.fit_resample(dataset.drop(columns=[y_column]), y_encoded)
    y = encoder.inverse_transform(y)

    dataset = pd.DataFrame(x, columns=X_columns)
    dataset[y_column] = y

    len_data = len(dataset[y_column])
    print(f"Total data after SMOTE: {len_data}")

    feature_class_number = [19, 45]
    return dataset, y_column, all_labels, feature_class_number


def IDS():
    DATASET_DIRECTORY = './data/CICIDS2017_RFE_binary_multiclass_balanced.csv'

    X_columns = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
                 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
                 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
                 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
                 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Mean', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
                 'Packet Length Variance', 'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
                 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Subflow Fwd Packets',
                 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Idle Max', 'Idle Min']

    all_labels = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'FTP-Patator', 'SSH-Patator',
                  'DoS slowloris',
                  'DoS Slowhttptest', 'Bot', 'Brute Force', 'XSS', 'Infiltration', 'Sql Injection', 'Heartbleed']

    y_column = 'Label'
    df_sets = pd.read_csv(DATASET_DIRECTORY, usecols=X_columns + [y_column])
    df_sets.rename(columns={'Label': 'label'}, inplace=True)
    df_sets = df_sets.sample(frac=0.35, random_state=42)
    y_column = 'label'

    dataset = df_sets
    len_data = len(dataset)
    print(f"Total data before SMOTE: {len_data}")

    encoder = LabelEncoder()
    encoder.fit(all_labels)
    y_encoded = encoder.transform(dataset[y_column])
    class_counts = np.bincount(y_encoded)
    target_samples = 3000
    sampling_strategy = {class_label: target_samples for class_label, count in enumerate(class_counts) if
                         0 < count < target_samples}
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    x, y = smote.fit_resample(dataset.drop(columns=[y_column]), y_encoded)
    y = encoder.inverse_transform(y)

    dataset = pd.DataFrame(x, columns=X_columns)
    dataset[y_column] = y

    len_data = len(dataset[y_column])
    print(f"Total data after SMOTE: {len_data}")
    feature_class_number = [15, 49]
    return dataset, y_column, all_labels, feature_class_number


def loading_dataset(client_num, dataset, y_column, all_labels):
    ######### This is IID setting #########
    # for each client in IID setting, they will have 10% of each class data
    client_data_iid = []
    for i in range(client_num):
        local_data = pd.DataFrame()
        for label in all_labels:
            temp = dataset[dataset[y_column] == label].sample(frac=1 / client_num)
            local_data = pd.concat([temp, local_data], ignore_index=True)

        client_data_iid.append(local_data)

    ######### This is non-IID setting #########

    # for each client, they will have random classes data but maximum 5 classes and each class can only be selected once
    client_data_niid = []
    shuffled_classes = np.random.permutation(all_labels)

    for i in range(client_num):
        client_data_niid.append([])  # Initialize each client's class list

    # Initial distribution to ensure each class is assigned at least once
    for class_index, class_name in enumerate(shuffled_classes):
        client_index = class_index % client_num
        client_data_niid[client_index].append(class_name)

    # Random allocation to fill up to 5 classes per client
    for client_classes in client_data_niid:
        while len(client_classes) < 4:
            additional_classes = [cls for cls in shuffled_classes if cls not in client_classes]
            if not additional_classes:  # Break if there are no more classes to add
                break
            additional_class = np.random.choice(additional_classes)
            client_classes.append(additional_class)

    # Filtering dataset for each client based on the selected classes and storing the data

    print()
    print("#############################################")
    print("IID data partitioning ...")
    print("#############################################")
    print()
    iid_data_set = prep_data(client_data_iid, all_labels)
    print()
    print("#############################################")
    print("NIID data partitioning ...")
    print("#############################################")
    print()
    for i, client_classes in enumerate(client_data_niid):
        local_data = dataset[dataset[y_column].isin(client_classes)]
        client_data_niid[i] = local_data  # Replace class list with actual data for each client
        print(f"Client {i} has {len(local_data)} samples from classes: {client_classes}")
    niid_data_set = prep_data(client_data_niid, all_labels)

    return iid_data_set, niid_data_set
