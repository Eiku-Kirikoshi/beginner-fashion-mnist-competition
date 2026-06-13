
# uv run src/train.py

import pickle
from pathlib import Path

from load_fashion_mnist import load_train_data
from network import NetworkConfig, SimpleMLP

OUTPUT_PATH = Path("sample_weight.pkl")

# Dropoutの導入により収束が遅くなるためエポックを増加
EPOCHS = 40
HIDDEN_SIZE1 = 512
HIDDEN_SIZE2 = 256
LEARNING_RATE = 0.001
BATCH_SIZE = 64
SEED = 42

def main() -> int:
    (x_train, t_train), (x_valid, t_valid) = load_train_data()

    model = SimpleMLP(
        NetworkConfig(
            input_size=x_train.shape[1],
            hidden_size1=HIDDEN_SIZE1,
            hidden_size2=HIDDEN_SIZE2,
            output_size=10,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
    )

    for epoch in range(1, EPOCHS + 1):
        # 【追加3】学習の終盤で学習率を10分の1に下げる (Learning Rate Decay)
        if epoch == 30:
            model.config.learning_rate *= 0.1
            print(f"  -> Learning rate dynamically decayed to {model.config.learning_rate:.4f}")

        loss = model.train_epoch(x_train, t_train, epoch=epoch)
        train_acc = model.evaluate_accuracy(x_train, t_train)
        valid_acc = model.evaluate_accuracy(x_valid, t_valid)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={loss:.4f} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}"
        )

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(model.to_state(), f)

    print(f"Saved model: {OUTPUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())