from argparse import ArgumentParser
import logging
import os
import statistics

import numpy as np
from PIL import Image
import torch.utils
import torch.nn as nn
import torch.optim as optim

from tech_drawing_correction import data, network


LOGGER = logging.getLogger(__name__)


def main():
    argp = ArgumentParser()
    argp.add_argument("--batch-size", type=int, default=6)
    argp.add_argument("--epochs", type=int, default=60)
    argp.add_argument("--limit", type=int, default=2**20)
    args = argp.parse_args()

    LOGGER.info("Preparing model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.Network()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.99)

    LOGGER.info("Loading data")
    trainset = data.TrainDataset(limit=args.limit)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = data.TestDataset(limit=args.limit)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    LOGGER.info("Training")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for i, batch in enumerate(trainloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        print('Epoch %03d avg. loss: %.4f' % (epoch, statistics.mean(losses)))

        if epoch % 5 == 4:
            with torch.no_grad():
                model.eval()
                losses = []
                for i, batch in enumerate(testloader):
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    prediction = model(x)
                    losses.append(criterion(prediction, y).item())

                    if i != 0:
                        continue

                    for j in range(3):
                        for t, name in zip((x, y, prediction), ("dirty", "orig", "fixed")):
                            img_np = np.maximum(np.minimum(t[j][0].cpu().detach().numpy() * 255,
                                                           255), 0)
                            img = Image.fromarray(img_np.astype(np.uint8))
                            os.makedirs("output", exist_ok=True)
                            img.convert("RGB").save("output/test_epoch_%04d_%d_%s.png" % (
                                epoch, j, name))

                print("Test set loss: ", statistics.mean(losses))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
