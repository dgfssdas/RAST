import math
from modules.RAST import RAST
from utils.util import *
from time import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def train(args, data, device):
    train_dataset = prepare_data_for_dataloader(data.train_data)
    valid_dataset = prepare_data_for_dataloader(data.valid_data)
    test_dataset = prepare_data_for_dataloader(data.test_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    """load model"""
    logging.info("begin load model ...")
    model = RAST(args, data, device)
    logging.info("model parameters: " + get_total_parameters(model))
    model.to(device)
    logging.info(model)

    """prepare optimizer"""
    logging.info("begin prepare optimizer ...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """start training"""
    logging.info("start training ...")

    for epoch in range(1, args.epoch + 1):
        total_loss = 0
        torch.cuda.empty_cache()
        model.train()
        time_start = time()

        for inputs, labels, node_indexes in tqdm(train_loader, desc="Train processing..."):
            optimizer.zero_grad()

            inputs, labels, node_indexes = inputs.to(device), labels.to(device), node_indexes.to(device)
            batch_loss = model(inputs, labels, node_indexes, mode="train")

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        time_train = time() - time_start

        if epoch % args.eval_step == 0 or epoch == 1:
            model.eval()

            time_start = time()
            valid_auc, valid_acc, valid_f1 = risk_eval(model, valid_loader, device)
            test_auc, test_acc, test_f1 = risk_eval(model, test_loader, device)

            time_eval = time() - time_start

            logging.info('Epoch: %d  loss: %.4f  train_time: %.2fs  eval_time: %.2fs' % (epoch, total_loss / len(train_loader), time_train, time_eval))
            logging.info('valid auc: %.4f  acc: %.4f  f1: %.4f  |  test auc: %.4f  acc: %.4f  f1: %.4f' % (valid_auc, valid_acc, valid_f1, test_auc, test_acc, test_f1))
            logging.info("-")


def prepare_data_for_dataloader(data_dict):
    node_indexes = []
    X_all = []
    y_all = []

    for node_id, data in data_dict.items():
        X_all.append(data['X'])
        y_all.append(data['y'])
        node_indexes.append(np.full(len(data['X']), node_id))

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    node_indexes = np.concatenate(node_indexes, axis=0)

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)
    node_indexes_tensor = torch.tensor(node_indexes, dtype=torch.long)  # 节点索引的张量

    dataset = TensorDataset(X_tensor, y_tensor, node_indexes_tensor)
    return dataset


def risk_eval(model, data_loader, device):
    auc_list = []
    acc_list = []
    f1_list = []
    for inputs, labels, node_indexes in data_loader:
        inputs, labels, node_indexes = inputs.to(device), labels.to(device), node_indexes.to(device)
        with torch.no_grad():
            scores = model(inputs, labels, node_indexes, mode="eval")

        scores = scores.cpu().numpy()

        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        acc = np.mean(np.equal(scores, labels))

        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)

    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))
