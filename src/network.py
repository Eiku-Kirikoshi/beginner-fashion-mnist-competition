# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any

# import numpy as np


# def _softmax(x: np.ndarray) -> np.ndarray:
#     shifted = x - np.max(x, axis=1, keepdims=True)
#     exp_x = np.exp(shifted)
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
#     out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
#     out[np.arange(labels.shape[0]), labels] = 1.0
#     return out


# @dataclass
# class NetworkConfig:
#     input_size: int = 784
#     hidden_size: int = 1024
#     output_size: int = 10
#     learning_rate: float = 0.01
#     batch_size: int = 64
#     seed: int = 42


# class SimpleMLP:
#     def __init__(self, config: NetworkConfig) -> None:
#         self.config = config
#         rng = np.random.default_rng(config.seed)
#         self.params: dict[str, np.ndarray] = {
#             "W1": (rng.standard_normal((config.input_size, config.hidden_size)) * 0.01).astype(
#                 np.float32
#             ),
#             "b1": np.zeros(config.hidden_size, dtype=np.float32),
#             "W2": (rng.standard_normal((config.hidden_size, config.output_size)) * 0.01).astype(
#                 np.float32
#             ),
#             "b2": np.zeros(config.output_size, dtype=np.float32),
#         }

#     def predict_proba(self, x: np.ndarray) -> np.ndarray:
#         z1 = np.tanh(np.dot(x, self.params["W1"]) + self.params["b1"])
#         logits = np.dot(z1, self.params["W2"]) + self.params["b2"]
#         return _softmax(logits)

#     def predict(self, x: np.ndarray) -> np.ndarray:
#         return np.argmax(self.predict_proba(x), axis=1)

#     def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
#         correct = 0
#         total = x.shape[0]
#         batch_size = self.config.batch_size
#         for i in range(0, total, batch_size):
#             x_batch = x[i : i + batch_size]
#             y_batch = y[i : i + batch_size]
#             pred = self.predict(x_batch)
#             correct += int(np.sum(pred == y_batch))
#         return float(correct) / float(total)

#     def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
#         rng = np.random.default_rng(self.config.seed + epoch)
#         indices = rng.permutation(x.shape[0])
#         total_loss = 0.0
#         steps = 0
#         batch_size = self.config.batch_size

#         for start in range(0, x.shape[0], batch_size):
#             batch_idx = indices[start : start + batch_size]
#             x_batch = x[batch_idx]
#             y_batch = y[batch_idx]

#             z1_linear = np.dot(x_batch, self.params["W1"]) + self.params["b1"]
#             z1 = np.tanh(z1_linear)
#             logits = np.dot(z1, self.params["W2"]) + self.params["b2"]
#             probs = _softmax(logits)

#             y_one_hot = _one_hot(y_batch, self.config.output_size)
#             loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
#             total_loss += float(loss)
#             steps += 1

#             d_logits = (probs - y_one_hot) / x_batch.shape[0]
#             dW2 = np.dot(z1.T, d_logits)
#             db2 = np.sum(d_logits, axis=0)

#             d_z1 = np.dot(d_logits, self.params["W2"].T)
#             d_z1_linear = d_z1 * (1.0 - np.square(z1))
#             dW1 = np.dot(x_batch.T, d_z1_linear)
#             db1 = np.sum(d_z1_linear, axis=0)

#             lr = self.config.learning_rate
#             self.params["W1"] -= lr * dW1.astype(np.float32)
#             self.params["b1"] -= lr * db1.astype(np.float32)
#             self.params["W2"] -= lr * dW2.astype(np.float32)
#             self.params["b2"] -= lr * db2.astype(np.float32)

#         return total_loss / max(steps, 1)

#     def to_state(self) -> dict[str, object]:
#         return {
#             "model_type": "SimpleMLP",
#             "config": {
#                 "input_size": self.config.input_size,
#                 "hidden_size": self.config.hidden_size,
#                 "output_size": self.config.output_size,
#                 "learning_rate": self.config.learning_rate,
#                 "batch_size": self.config.batch_size,
#                 "seed": self.config.seed,
#             },
#             "params": self.params,
#         }

#     @classmethod
#     def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
#         config_obj = state.get("config")
#         if not isinstance(config_obj, dict):
#             raise ValueError("Invalid state: 'config' must be a dict")
#         config_dict: dict[str, Any] = config_obj

#         config = NetworkConfig(
#             input_size=int(config_dict["input_size"]),
#             hidden_size=int(config_dict["hidden_size"]),
#             output_size=int(config_dict["output_size"]),
#             learning_rate=float(config_dict.get("learning_rate", 0.1)),
#             batch_size=int(config_dict.get("batch_size", 128)),
#             seed=int(config_dict.get("seed", 42)),
#         )

#         params_obj = state.get("params")
#         if not isinstance(params_obj, dict):
#             raise ValueError("Invalid state: 'params' must be a dict")
#         params: dict[str, np.ndarray] = {}
#         for key, value in params_obj.items():
#             if not isinstance(key, str) or not isinstance(value, np.ndarray):
#                 raise ValueError("Invalid state: params must be dict[str, np.ndarray]")
#             params[key] = value

#         model = cls(config)
#         model.params = params
#         return model

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any

# import numpy as np


# def _softmax(x: np.ndarray) -> np.ndarray:
#     shifted = x - np.max(x, axis=1, keepdims=True)
#     exp_x = np.exp(shifted)
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
#     out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
#     out[np.arange(labels.shape[0]), labels] = 1.0
#     return out


# # 【修正1】NetworkConfigのパラメータを精度が出やすい値に変更
# @dataclass
# class NetworkConfig:
#     input_size: int = 784
#     hidden_size: int = 256        # 変更: 64 -> 256 (表現力の向上)
#     output_size: int = 10
#     learning_rate: float = 0.05   # 変更: 0.1 -> 0.05 (学習の安定化)
#     batch_size: int = 64          # 変更: 128 -> 64 (汎化性能の向上)
#     seed: int = 42


# class SimpleMLP:
#     def __init__(self, config: NetworkConfig) -> None:
#         self.config = config
#         rng = np.random.default_rng(config.seed)
        
#         # 【修正2】Heの初期化の導入 (ReLUに最適な初期化手法)
#         # 以前は標準正規分布に一律 0.01 を掛けていましたが、ノード数に応じた係数にします。
#         he_init_W1 = np.sqrt(2.0 / config.input_size)
#         he_init_W2 = np.sqrt(2.0 / config.hidden_size)

#         self.params: dict[str, np.ndarray] = {
#             "W1": (rng.standard_normal((config.input_size, config.hidden_size)) * he_init_W1).astype(
#                 np.float32
#             ),
#             "b1": np.zeros(config.hidden_size, dtype=np.float32),
#             "W2": (rng.standard_normal((config.hidden_size, config.output_size)) * he_init_W2).astype(
#                 np.float32
#             ),
#             "b2": np.zeros(config.output_size, dtype=np.float32),
#         }

#     def predict_proba(self, x: np.ndarray) -> np.ndarray:
#         z1_linear = np.dot(x, self.params["W1"]) + self.params["b1"]
#         # 【修正3】活性化関数を tanh から ReLU に変更
#         z1 = np.maximum(0, z1_linear)
#         logits = np.dot(z1, self.params["W2"]) + self.params["b2"]
#         return _softmax(logits)

#     def predict(self, x: np.ndarray) -> np.ndarray:
#         return np.argmax(self.predict_proba(x), axis=1)

#     def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
#         correct = 0
#         total = x.shape[0]
#         batch_size = self.config.batch_size
#         for i in range(0, total, batch_size):
#             x_batch = x[i : i + batch_size]
#             y_batch = y[i : i + batch_size]
#             pred = self.predict(x_batch)
#             correct += int(np.sum(pred == y_batch))
#         return float(correct) / float(total)

#     def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
#         rng = np.random.default_rng(self.config.seed + epoch)
#         indices = rng.permutation(x.shape[0])
#         total_loss = 0.0
#         steps = 0
#         batch_size = self.config.batch_size

#         for start in range(0, x.shape[0], batch_size):
#             batch_idx = indices[start : start + batch_size]
#             x_batch = x[batch_idx]
#             y_batch = y[batch_idx]

#             # --- 順伝播 (Forward) ---
#             z1_linear = np.dot(x_batch, self.params["W1"]) + self.params["b1"]
#             # 【修正3】活性化関数を tanh から ReLU に変更
#             z1 = np.maximum(0, z1_linear)
#             logits = np.dot(z1, self.params["W2"]) + self.params["b2"]
#             probs = _softmax(logits)

#             y_one_hot = _one_hot(y_batch, self.config.output_size)
#             loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
#             total_loss += float(loss)
#             steps += 1

#             # --- 逆伝播 (Backward) ---
#             d_logits = (probs - y_one_hot) / x_batch.shape[0]
#             dW2 = np.dot(z1.T, d_logits)
#             db2 = np.sum(d_logits, axis=0)

#             d_z1 = np.dot(d_logits, self.params["W2"].T)
#             # 【修正4】tanhの微分から ReLUの微分 に変更
#             # ReLUの微分は「入力が0より大きい場合は1、それ以外は0」
#             d_z1_linear = d_z1 * (z1_linear > 0).astype(np.float32)
            
#             dW1 = np.dot(x_batch.T, d_z1_linear)
#             db1 = np.sum(d_z1_linear, axis=0)

#             lr = self.config.learning_rate
#             self.params["W1"] -= lr * dW1.astype(np.float32)
#             self.params["b1"] -= lr * db1.astype(np.float32)
#             self.params["W2"] -= lr * dW2.astype(np.float32)
#             self.params["b2"] -= lr * db2.astype(np.float32)

#         return total_loss / max(steps, 1)

#     # 以下の状態保存・復元メソッドは変更なし
#     def to_state(self) -> dict[str, object]:
#         return {
#             "model_type": "SimpleMLP",
#             "config": {
#                 "input_size": self.config.input_size,
#                 "hidden_size": self.config.hidden_size,
#                 "output_size": self.config.output_size,
#                 "learning_rate": self.config.learning_rate,
#                 "batch_size": self.config.batch_size,
#                 "seed": self.config.seed,
#             },
#             "params": self.params,
#         }

#     @classmethod
#     def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
#         config_obj = state.get("config")
#         if not isinstance(config_obj, dict):
#             raise ValueError("Invalid state: 'config' must be a dict")
#         config_dict: dict[str, Any] = config_obj

#         config = NetworkConfig(
#             input_size=int(config_dict["input_size"]),
#             hidden_size=int(config_dict["hidden_size"]),
#             output_size=int(config_dict["output_size"]),
#             learning_rate=float(config_dict.get("learning_rate", 0.05)), # ここのデフォルトも合わせて変更
#             batch_size=int(config_dict.get("batch_size", 64)),         # ここのデフォルトも合わせて変更
#             seed=int(config_dict.get("seed", 42)),
#         )

#         params_obj = state.get("params")
#         if not isinstance(params_obj, dict):
#             raise ValueError("Invalid state: 'params' must be a dict")
#         params: dict[str, np.ndarray] = {}
#         for key, value in params_obj.items():
#             if not isinstance(key, str) or not isinstance(value, np.ndarray):
#                 raise ValueError("Invalid state: params must be dict[str, np.ndarray]")
#             params[key] = value

#         model = cls(config)
#         model.params = params
#         return model

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any

# import numpy as np


# def _softmax(x: np.ndarray) -> np.ndarray:
#     shifted = x - np.max(x, axis=1, keepdims=True)
#     exp_x = np.exp(shifted)
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
#     out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
#     out[np.arange(labels.shape[0]), labels] = 1.0
#     return out


# @dataclass
# class NetworkConfig:
#     input_size: int = 784
#     # 【修正1】表現力を最大化するためユニット数を増加
#     hidden_size: int = 512        
#     output_size: int = 10
#     # 【修正2】Adamオプティマイザに最適な学習率へ変更 (0.05 -> 0.001)
#     learning_rate: float = 0.001  
#     batch_size: int = 64
#     seed: int = 42
#     # 【修正3】過学習を防ぐためのL2正則化（Weight Decay）係数を追加
#     weight_decay: float = 1e-4    


# class SimpleMLP:
#     def __init__(self, config: NetworkConfig) -> None:
#         self.config = config
#         rng = np.random.default_rng(config.seed)
        
#         he_init_W1 = np.sqrt(2.0 / config.input_size)
#         he_init_W2 = np.sqrt(2.0 / config.hidden_size)

#         self.params: dict[str, np.ndarray] = {
#             "W1": (rng.standard_normal((config.input_size, config.hidden_size)) * he_init_W1).astype(
#                 np.float32
#             ),
#             "b1": np.zeros(config.hidden_size, dtype=np.float32),
#             "W2": (rng.standard_normal((config.hidden_size, config.output_size)) * he_init_W2).astype(
#                 np.float32
#             ),
#             "b2": np.zeros(config.output_size, dtype=np.float32),
#         }

#         # 【修正4】Adamオプティマイザ用の状態変数（モーメンタムとRMSProp）を初期化
#         self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
#         self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
#         self.t = 0 # Adamのタイムステップ

#     def predict_proba(self, x: np.ndarray) -> np.ndarray:
#         z1_linear = np.dot(x, self.params["W1"]) + self.params["b1"]
#         z1 = np.maximum(0, z1_linear)
#         logits = np.dot(z1, self.params["W2"]) + self.params["b2"]
#         return _softmax(logits)

#     def predict(self, x: np.ndarray) -> np.ndarray:
#         return np.argmax(self.predict_proba(x), axis=1)

#     def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
#         correct = 0
#         total = x.shape[0]
#         batch_size = self.config.batch_size
#         for i in range(0, total, batch_size):
#             x_batch = x[i : i + batch_size]
#             y_batch = y[i : i + batch_size]
#             pred = self.predict(x_batch)
#             correct += int(np.sum(pred == y_batch))
#         return float(correct) / float(total)

#     def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
#         rng = np.random.default_rng(self.config.seed + epoch)
#         indices = rng.permutation(x.shape[0])
#         total_loss = 0.0
#         steps = 0
#         batch_size = self.config.batch_size

#         for start in range(0, x.shape[0], batch_size):
#             batch_idx = indices[start : start + batch_size]
#             x_batch = x[batch_idx]
#             y_batch = y[batch_idx]

#             # --- 順伝播 (Forward) ---
#             z1_linear = np.dot(x_batch, self.params["W1"]) + self.params["b1"]
#             z1 = np.maximum(0, z1_linear)
#             logits = np.dot(z1, self.params["W2"]) + self.params["b2"]
#             probs = _softmax(logits)

#             y_one_hot = _one_hot(y_batch, self.config.output_size)
#             loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
#             total_loss += float(loss)
#             steps += 1

#             # --- 逆伝播 (Backward) ---
#             d_logits = (probs - y_one_hot) / x_batch.shape[0]
#             dW2 = np.dot(z1.T, d_logits)
#             db2 = np.sum(d_logits, axis=0)

#             d_z1 = np.dot(d_logits, self.params["W2"].T)
#             d_z1_linear = d_z1 * (z1_linear > 0).astype(np.float32)
            
#             dW1 = np.dot(x_batch.T, d_z1_linear)
#             db1 = np.sum(d_z1_linear, axis=0)

#             # 【修正5】L2正則化（Weight Decay）を重みの勾配に適用（バイアスには適用しないのが一般的）
#             wd = self.config.weight_decay
#             dW2 += wd * self.params["W2"]
#             dW1 += wd * self.params["W1"]

#             # 【修正6】単純なSGDからAdamオプティマイザによる重み更新へ変更
#             grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
#             lr = self.config.learning_rate
#             beta1, beta2, eps = 0.9, 0.999, 1e-8
#             self.t += 1

#             for key in self.params:
#                 g = grads[key]
#                 self.m[key] = beta1 * self.m[key] + (1.0 - beta1) * g
#                 self.v[key] = beta2 * self.v[key] + (1.0 - beta2) * np.square(g)
                
#                 # バイアス補正
#                 m_hat = self.m[key] / (1.0 - beta1 ** self.t)
#                 v_hat = self.v[key] / (1.0 - beta2 ** self.t)
                
#                 self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

#         return total_loss / max(steps, 1)

#     def to_state(self) -> dict[str, object]:
#         return {
#             "model_type": "SimpleMLP",
#             "config": {
#                 "input_size": self.config.input_size,
#                 "hidden_size": self.config.hidden_size,
#                 "output_size": self.config.output_size,
#                 "learning_rate": self.config.learning_rate,
#                 "batch_size": self.config.batch_size,
#                 "seed": self.config.seed,
#                 "weight_decay": self.config.weight_decay, # 【修正7】追加したプロパティを保存
#             },
#             "params": self.params,
#         }

#     @classmethod
#     def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
#         config_obj = state.get("config")
#         if not isinstance(config_obj, dict):
#             raise ValueError("Invalid state: 'config' must be a dict")
#         config_dict: dict[str, Any] = config_obj

#         config = NetworkConfig(
#             input_size=int(config_dict["input_size"]),
#             hidden_size=int(config_dict["hidden_size"]),
#             output_size=int(config_dict["output_size"]),
#             learning_rate=float(config_dict.get("learning_rate", 0.001)), 
#             batch_size=int(config_dict.get("batch_size", 64)),         
#             seed=int(config_dict.get("seed", 42)),
#             weight_decay=float(config_dict.get("weight_decay", 1e-4)), # 【修正7】追加したプロパティを復元
#         )

#         params_obj = state.get("params")
#         if not isinstance(params_obj, dict):
#             raise ValueError("Invalid state: 'params' must be a dict")
#         params: dict[str, np.ndarray] = {}
#         for key, value in params_obj.items():
#             if not isinstance(key, str) or not isinstance(value, np.ndarray):
#                 raise ValueError("Invalid state: params must be dict[str, np.ndarray]")
#             params[key] = value

#         model = cls(config)
#         model.params = params
#         return model

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


@dataclass
class NetworkConfig:
    input_size: int = 784
    hidden_size1: int = 512
    hidden_size2: int = 256
    output_size: int = 10
    learning_rate: float = 0.001
    batch_size: int = 64
    seed: int = 42
    weight_decay: float = 1e-4
    # 【追加1】Dropoutの割合 (20%のニューロンをランダムに無効化)
    dropout_rate: float = 0.2
    # 【追加2】Label Smoothingの係数
    label_smoothing: float = 0.1


class SimpleMLP:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        
        he_init_W1 = np.sqrt(2.0 / config.input_size)
        he_init_W2 = np.sqrt(2.0 / config.hidden_size1)
        he_init_W3 = np.sqrt(2.0 / config.hidden_size2)

        self.params: dict[str, np.ndarray] = {
            "W1": (rng.standard_normal((config.input_size, config.hidden_size1)) * he_init_W1).astype(np.float32),
            "b1": np.zeros(config.hidden_size1, dtype=np.float32),
            "W2": (rng.standard_normal((config.hidden_size1, config.hidden_size2)) * he_init_W2).astype(np.float32),
            "b2": np.zeros(config.hidden_size2, dtype=np.float32),
            "W3": (rng.standard_normal((config.hidden_size2, config.output_size)) * he_init_W3).astype(np.float32),
            "b3": np.zeros(config.output_size, dtype=np.float32),
        }

        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0 

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # 評価（推論）時はDropoutを適用しない
        z1 = np.maximum(0, np.dot(x, self.params["W1"]) + self.params["b1"])
        z2 = np.maximum(0, np.dot(z1, self.params["W2"]) + self.params["b2"])
        logits = np.dot(z2, self.params["W3"]) + self.params["b3"]
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        correct = int(np.sum(pred == y))
        return float(correct) / float(x.shape[0])

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
        rng = np.random.default_rng(self.config.seed + epoch)
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        steps = 0
        batch_size = self.config.batch_size

        y_one_hot_all = _one_hot(y, self.config.output_size)
        
        # 【追加2】Label Smoothingの適用
        ls = self.config.label_smoothing
        if ls > 0.0:
            y_one_hot_all = y_one_hot_all * (1.0 - ls) + (ls / self.config.output_size)

        rate = self.config.dropout_rate

        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x[batch_idx]
            y_one_hot = y_one_hot_all[batch_idx]

            # --- 順伝播 (Forward) ---
            # 隠れ層1
            z1_linear = np.dot(x_batch, self.params["W1"]) + self.params["b1"]
            z1_act = np.maximum(0, z1_linear)
            # 【追加1】Dropoutの適用 (学習時のみ)
            if rate > 0.0:
                mask1 = (rng.random(z1_act.shape, dtype=np.float32) > rate).astype(np.float32) / (1.0 - rate)
                z1_drop = z1_act * mask1
            else:
                z1_drop = z1_act

            # 隠れ層2
            z2_linear = np.dot(z1_drop, self.params["W2"]) + self.params["b2"]
            z2_act = np.maximum(0, z2_linear)
            # 【追加1】Dropoutの適用 (学習時のみ)
            if rate > 0.0:
                mask2 = (rng.random(z2_act.shape, dtype=np.float32) > rate).astype(np.float32) / (1.0 - rate)
                z2_drop = z2_act * mask2
            else:
                z2_drop = z2_act

            # 出力層
            logits = np.dot(z2_drop, self.params["W3"]) + self.params["b3"]
            probs = _softmax(logits)

            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
            total_loss += float(loss)
            steps += 1

            # --- 逆伝播 (Backward) ---
            d_logits = (probs - y_one_hot) / x_batch.shape[0]
            
            dW3 = np.dot(z2_drop.T, d_logits)
            db3 = np.sum(d_logits, axis=0)

            d_z2_drop = np.dot(d_logits, self.params["W3"].T)
            # Dropoutの逆伝播
            d_z2_act = d_z2_drop * mask2 if rate > 0.0 else d_z2_drop
            d_z2_linear = d_z2_act * (z2_linear > 0).astype(np.float32)
            
            dW2 = np.dot(z1_drop.T, d_z2_linear)
            db2 = np.sum(d_z2_linear, axis=0)

            d_z1_drop = np.dot(d_z2_linear, self.params["W2"].T)
            # Dropoutの逆伝播
            d_z1_act = d_z1_drop * mask1 if rate > 0.0 else d_z1_drop
            d_z1_linear = d_z1_act * (z1_linear > 0).astype(np.float32)
            
            dW1 = np.dot(x_batch.T, d_z1_linear)
            db1 = np.sum(d_z1_linear, axis=0)

            # L2正則化
            wd = self.config.weight_decay
            dW3 += wd * self.params["W3"]
            dW2 += wd * self.params["W2"]
            dW1 += wd * self.params["W1"]

            # --- Adam Optimizer 更新 ---
            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
            lr = self.config.learning_rate
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            self.t += 1

            for key in self.params:
                g = grads[key]
                self.m[key] *= beta1
                self.m[key] += (1.0 - beta1) * g
                
                self.v[key] *= beta2
                self.v[key] += (1.0 - beta2) * np.square(g)
                
                m_hat = self.m[key] / (1.0 - beta1 ** self.t)
                v_hat = self.v[key] / (1.0 - beta2 ** self.t)
                
                self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        return total_loss / max(steps, 1)

    def to_state(self) -> dict[str, object]:
        return {
            "model_type": "SimpleMLP",
            "config": {
                "input_size": self.config.input_size,
                "hidden_size1": self.config.hidden_size1,
                "hidden_size2": self.config.hidden_size2,
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
                "weight_decay": self.config.weight_decay,
                "dropout_rate": self.config.dropout_rate,       # 追加
                "label_smoothing": self.config.label_smoothing, # 追加
            },
            "params": self.params,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
        config_dict: dict[str, Any] = state.get("config", {}) # type: ignore

        config = NetworkConfig(
            input_size=int(config_dict["input_size"]),
            hidden_size1=int(config_dict.get("hidden_size1", 512)),
            hidden_size2=int(config_dict.get("hidden_size2", 256)),
            output_size=int(config_dict["output_size"]),
            learning_rate=float(config_dict.get("learning_rate", 0.001)), 
            batch_size=int(config_dict.get("batch_size", 64)),         
            seed=int(config_dict.get("seed", 42)),
            weight_decay=float(config_dict.get("weight_decay", 1e-4)),
            dropout_rate=float(config_dict.get("dropout_rate", 0.2)),       # 追加
            label_smoothing=float(config_dict.get("label_smoothing", 0.1)), # 追加
        )

        params_obj = state.get("params", {})
        params: dict[str, np.ndarray] = {k: v for k, v in params_obj.items()} # type: ignore

        model = cls(config)
        model.params = params
        return model