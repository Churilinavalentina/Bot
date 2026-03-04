import torch
import pickle # для сохранения объектов.

from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from params import START_DATE, END_DATE, DELTA_T, DELTA_T_FUTURE, K, TRAIN_PART, BATCH_SIZE, EPOCHS, LOSS, FOLDER
from read_data import MyCSVDataset, split_data, check_balance
from model import MyModel, calc_metrics
from pathlib import Path

# get_values(get_figi(TICKERS).figi, INTERVAL, START_DATE, END_DATE, num_workers=1)

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    full_ds = MyCSVDataset(DELTA_T, DELTA_T_FUTURE, K, START_DATE, END_DATE)
    train_ds, test_ds = split_data(full_ds, train_part=TRAIN_PART)

    check_balance(full_ds)
    check_balance(train_ds)
    check_balance(test_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MyModel(input_dim=DELTA_T).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LOSS, weight_decay=0)
    criterion = torch.nn.BCELoss()

    # Цикл обучения по батчам
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in tqdm(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

        print(f'Эпоха [{epoch+1}/{EPOCHS}], Ошибка: {loss.item():.4f}')

        # validation
        model.eval()

        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                all_preds.extend(outputs)
                all_true.extend(batch_y)

        all_preds = torch.cat(all_preds)
        all_true = torch.cat(all_true)


        # print(all_preds)
        # print(all_true)
        calc_metrics(all_preds, all_true)

    model_path = f"{FOLDER}models_{START_DATE.date()}-{END_DATE.date()}"
    torch.save(model.state_dict(), model_path)
    
if __name__ == '__main__':
    main()