import numpy as np
import torch
from tqdm import tqdm


def print_result(result):
    """
    결과를 print하는 함수 입니다.
    :param result: list를 input으로 받아 print합니다.
    :return:
    """
    epoch, train_loss, valid_loss, train_acc, valid_acc = result
    print(
        f"[epoch{epoch}] train_loss: {round(train_loss, 3)}, valid_loss: {round(valid_loss, 3)}, train_acc: {train_acc}%, valid_acc: {valid_acc}%"
    )


def get_accuracy(y_hat, target, name='MAE'):

    acc = None
    diff = torch.abs(y_hat - target)
    mae = torch.mean(diff)
    mse = torch.mean(torch.pow(diff, 2))
    rmse = torch.sqrt(mse)


    return acc

def split_dataset(df, config):
    from sklearn.model_selection import train_test_split
    """
    학습 데이터셋과 검증 데이터셋으로 나누는 함수입니다.
    :param data_path:
    :param test_size:
    :param seed:
    :param target_column:
    :return:
    """

    if config['PARTITION'] == 2:
        train, test = train_test_split(
            df,
            test_size=0.2,
            shuffle=True,
            random_state=config['SEED']
        )

        # 데이터 로더
        return train, test

    else:
        train, test = train_test_split(
            df,
            test_size=0.4,
            shuffle=True,
            random_state=config['SEED']
        )

        valid, test = train_test_split(
            test,
            test_size=0.5,
            random_state=config['SEED']
        )

        return train, valid, test


def share_loop(epoch=10,
               model=None,
               data_loader=None,
               criterion=None,
               optimizer=None,
               mode="train"):
    """
    학습과 검증에서 사용하는 loop 입니다. mode를 이용하여 조정합니다.
    :param epoch:
    :param model:
    :param data_loader:
    :param criterion:
    :param optimizer:
    :param mode: 'train', 'valid' 중 하나의 값을 받아 loop를 진행합니다.
    :return: average_loss(float64), total_losses(list), accuracy(float)
    """
    count = 0
    correct = 0
    total_losses = []

    opt_name = optimizer[1]
    optimizer = optimizer[0]

    mode = mode.lower()
    progress_bar = tqdm(data_loader, desc=f"{mode} {epoch}")
    if mode == "train":
        model.train()
        for data, label in progress_bar:
            out = model(data).float()
            # label = label.float()

            loss = criterion(out, label)
            # 역전파
            optimizer.zero_grad()
            loss.backward()

            if opt_name == "SAM":
                optimizer.first_step(zero_grad=True)
                criterion(
                    model(data).logits, label
                ).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            total_losses.append(loss.item())

            # accuracy 계산
            acc = RMSE()


            progress_bar.set_postfix({'loss': loss.item(), 'acc': acc})

    elif mode == 'valid':
        model.eval()
        with torch.no_grad():
            for data, label in progress_bar:
                # data, label = batch
                out = model(data).logits

                loss = criterion(out, label)

                total_losses.append(loss.item())

                progress_bar.set_postfix({'loss': loss.item(), 'acc': acc})

    elif mode == 'test':
        model.eval()
        with torch.no_grad():
            all_preds = []
            for batch in tqdm(data_loader, desc=f"{mode}"):
                data = batch
                out = model(data).logits

                predicted = torch.argmax(torch.softmax(out, dim=1), dim=1)
                all_preds.append(predicted.detach())
        return all_preds

    elif mode == 'ensemble':
        all_preds = []
        for batch in tqdm(data_loader, desc=f"{mode}"):
            tmp_out = 0
            for model_ in model:
                model_.eval()
                with torch.no_grad():
                    data = batch
                    out = model_(data).logits
                    soft_out = torch.softmax(out, dim=1)
                    tmp_out += soft_out

            predicted = torch.argmax(soft_out, dim=1)
            all_preds.append(predicted.detach())
        return all_preds

    else:
        raise Exception(f'mode는 train, valid 중 하나여야 합니다. 현재 mode값 -> {mode}')

    avg_loss = np.average(total_losses)
    accuracy = 100 * correct / count  # Accuracy 계산
    return avg_loss, total_losses, accuracy


if __name__ == '__main__':
    pass
