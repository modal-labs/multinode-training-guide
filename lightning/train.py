
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.demos import Transformer, WikiText2
from torch.utils.data import DataLoader


def main():
    L.seed_everything(42)

    fabric = L.Fabric()

    # Data
    with fabric.rank_zero_first():
        dataset = WikiText2(download=True)

    train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    # Model
    model = Transformer(vocab_size=dataset.vocab_size)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    max_steps = len(train_dataloader)
    for batch_idx, batch in enumerate(train_dataloader):
        input, target = batch
        output = model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            fabric.print(f"iteration: {batch_idx}/{max_steps} - loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
