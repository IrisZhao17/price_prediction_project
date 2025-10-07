import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries

import yfinance as yf
import pandas as pd

print("All libraries loaded")

config = {
    "alpha_vantage": {
        "key": "YOUR_API_KEY", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 12, # 12 features
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

def download_data(config):
    import yfinance as yf
    import pandas as pd

    symbol = config["alpha_vantage"]["symbol"]  # e.g. "AAPL"
    df = yf.download(symbol, period="5y", interval="1d", auto_adjust=False, progress=False)

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    df = df.dropna()

    # 使用原始列：['Open','High','Low','Close','Adj Close','Volume']
    ohlc = df[["Open", "High", "Low", "Close"]].copy()

    data_open = ohlc["Open"].to_numpy()
    data_high = ohlc["High"].to_numpy()
    data_low  = ohlc["Low"].to_numpy()
    data_close = ohlc["Close"].to_numpy()
    data_date = [d.strftime("%Y-%m-%d") for d in ohlc.index]

    num_data_points = len(data_close)
    display_date_range = f"{data_date[0]} to {data_date[-1]}"
    return data_date, data_open, data_high, data_low, data_close, num_data_points, display_date_range


# 调用
data_date, data_open, data_high, data_low, data_close, num_data_points, display_date_range = download_data(config)


# plot

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close, color=config["plots"]["color_actual"])
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
plt.grid(which='major', axis='y', linestyle='--')
plt.show()

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

# normalize
scaler = Normalizer()
normalized_data_close = scaler.fit_transform(data_close)

# ===== 新增：技术指标构建 =====

def build_features_from_ohlc(open_np, high_np, low_np, close_np):
    import numpy as np
    import pandas as pd

    # ---- 保证每列是一维并且长度一致 ----
    o = np.asarray(open_np).ravel()
    h = np.asarray(high_np).ravel()
    l = np.asarray(low_np).ravel()
    c = np.asarray(close_np).ravel()

    n = len(c)
    assert len(o) == len(h) == len(l) == len(c) and n > 0, \
        (len(o), len(h), len(l), len(c))

    df = pd.DataFrame({
        "open": o,
        "high": h,
        "low":  l,
        "close": c,
    })

    # === 基础特征 & 指标（保持因果）===
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]
    df["ret1"] = df["close"].pct_change()
    df["logret1"] = np.log(df["close"]).diff()

    df["sma5"]  = df["close"].rolling(5).mean()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    ema26       = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]      = df["ema12"] - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_z"] = (df["close"] - mid) / std
    df["rv5"]  = df["logret1"].rolling(5).std() * np.sqrt(252)

    # 丢 NaN，得到有效 T'
    df = df.dropna().reset_index(drop=True)

    # feature_cols = [
    #     "open","high","low","close",
    #     "hl_range","oc_change","ret1","logret1",
    #     "sma5","ema12","macd","macd_hist","bb_z","rv5"
    # ]
    feature_cols = [
        "open","high","low","close"
    ]
    features = df[feature_cols].astype(np.float32).values  # [T', F]
    return features, feature_cols


def prepare_multifeat_windows(X_2d: np.ndarray, window_size: int):
    """
    输入: X_2d [T', F]
    输出: X_win [N, window, F], X_unseen [window, F]
    """
    T, F = X_2d.shape
    N = T - window_size
    X_win = np.stack([X_2d[i:i+window_size, :] for i in range(N)], axis=0)
    X_unseen = X_2d[-window_size:, :]
    return X_win, X_unseen

# 构建特征矩阵
features_raw, feature_cols = build_features_from_ohlc(data_open, data_high, data_low, data_close)

# 按列标准化
scaler_X = Normalizer()
X_norm = scaler_X.fit_transform(features_raw)   # shape [T', F]
F = X_norm.shape[1]
config["model"]["input_size"] = F

# y: 标准化的 close（目标值）
scaler_y = Normalizer()
normalized_close = scaler_y.fit_transform(np.array(data_close).reshape(-1, 1)).ravel()

# 对齐 offset
offset = len(data_close) - X_norm.shape[0]
y_close_cut = normalized_close[offset:]  # [T']

# 滑动窗口
data_x, data_x_unseen = prepare_multifeat_windows(X_norm, config["data"]["window_size"])
data_y = y_close_cut[config["data"]["window_size"]:]

# 训练/验证划分
split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
data_x_train, data_x_val = data_x[:split_index], data_x[split_index:]
data_y_train, data_y_val = data_y[:split_index], data_y[split_index:]


# prepare data for plotting
num_points = num_data_points  # 建议确保 = len(data_close_price)

to_plot_data_y_train = np.full(num_points, np.nan, dtype=float)
to_plot_data_y_val   = np.full(num_points, np.nan, dtype=float)

window_size = config["data"]["window_size"]

# 计算切片区间
train_start = offset + window_size
train_end   = train_start + len(data_y_train)

val_start   = offset + window_size + split_index
val_end     = val_start + len(data_y_val)

# 边界与长度自检（出问题会直接抛错，方便定位）
assert 0 <= train_start <= train_end <= num_points, (train_start, train_end, num_points)
assert 0 <= val_start   <= val_end   <= num_points, (val_start,   val_end,   num_points)

# 确保 inverse_transform 的输入是二维
y_train_inv = scaler.inverse_transform(np.asarray(data_y_train).reshape(-1, 1)).ravel()
y_val_inv   = scaler.inverse_transform(np.asarray(data_y_val).reshape(-1, 1)).ravel()

# 再次校验左右长度完全一致
assert (train_end - train_start) == len(y_train_inv)
assert (val_end   - val_start)   == len(y_val_inv)

# 赋值
to_plot_data_y_train[train_start:train_end] = y_train_inv
to_plot_data_y_val[val_start:val_end]       = y_val_inv

## plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] + " - showing training and validation data")
plt.grid(True, which='major', axis='y', linestyle='--')
plt.legend()
plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)   # [N, window, F]
        self.y = y.astype(np.float32)   # [N]
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return (self.x[idx], self.y[idx])


dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))


# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

# predict on the training data, to see how well the model managed to learn and memorize

predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

# predict on the validation data, to see how the model does

predicted_val = np.array([])

for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val = np.concatenate((predicted_val, out))

# === prepare data for plotting (predictions) ===
num_points = num_data_points
window_size = config["data"]["window_size"]

to_plot_data_y_train_pred = np.full(num_points, np.nan, dtype=float)
to_plot_data_y_val_pred   = np.full(num_points, np.nan, dtype=float)

# 起止索引（与真实 y 的对齐完全一致）
train_start = offset + window_size
train_end   = train_start + len(predicted_train)

val_start   = offset + window_size + split_index
val_end     = val_start + len(predicted_val)

# 防越界 + 保证长度匹配
assert 0 <= train_start <= train_end <= num_points
assert 0 <= val_start   <= val_end   <= num_points

# inverse_transform 需要二维；再 ravel 回一维
pred_train_inv = scaler.inverse_transform(np.asarray(predicted_train).reshape(-1, 1)).ravel()
pred_val_inv   = scaler.inverse_transform(np.asarray(predicted_val).reshape(-1, 1)).ravel()

# 再次校验左右长度一致
assert (train_end - train_start) == len(pred_train_inv)
assert (val_end   - val_start)   == len(pred_val_inv)

# 赋值
to_plot_data_y_train_pred[train_start:train_end] = pred_train_inv
to_plot_data_y_val_pred[val_start:val_end]       = pred_val_inv


# plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close, label="Actual prices", color=config["plots"]["color_actual"])
plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
plt.title("Compare predicted prices to actual prices")
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.grid(True, which='major', axis='y', linestyle='--')
plt.legend()
plt.show()

# === zoomed-in plotting: align dates with offset and lengths ===
window_size = config["data"]["window_size"]

# 反归一化 (确保二维输入再拉平)
y_val_inv = scaler.inverse_transform(np.asarray(data_y_val).reshape(-1, 1)).ravel()
pred_val_inv = scaler.inverse_transform(np.asarray(predicted_val).reshape(-1, 1)).ravel()

# 若预测数量和标签数量不等（例如 DataLoader(drop_last=True)），裁成相同长度
L = min(len(y_val_inv), len(pred_val_inv))
y_val_inv = y_val_inv[:L]
pred_val_inv = pred_val_inv[:L]

# 日期序列与 y 对齐（关键：加 offset）
val_start = offset + window_size + split_index
val_end   = val_start + L
dates_val = np.asarray(data_date)[val_start:val_end]

# 自检，防越界/长度不一致
assert len(dates_val) == L, (len(dates_val), L)

# --------- 画图（两种写法选一种）---------

# 写法 A：用日期做 x（更直观）
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(dates_val, y_val_inv,  label="Actual prices",    color=config["plots"]["color_actual"])
plt.plot(dates_val, pred_val_inv, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
plt.title("Zoom in to examine predicted price on validation data portion")
plt.grid(True, which='major', axis='y', linestyle='--')
plt.legend()

# 如果仍想稀疏显示 x 轴标签，可只抽取一部分位置：
step = max(1, int(config["plots"]["xticks_interval"] // 5) or 1)
tick_idx = [i for i in range(L) if (i % step == 0) or (i == L-1)]
plt.xticks([dates_val[i] for i in tick_idx],
           [dates_val[i] for i in tick_idx], rotation='vertical')

plt.show()

# （可选）写法 B：用索引作为 x，然后把日期当作刻度标签
# xs = np.arange(L)
# plt.plot(xs, y_val_inv,  ...)
# plt.plot(xs, pred_val_inv, ...)
# 同样按 step 取 tick_idx，然后 plt.xticks(xs[tick_idx], dates_val[tick_idx], rotation='vertical')


# predict the closing price of the next trading day

model.eval()

x = torch.tensor(data_x_unseen, dtype=torch.float32, device=config["training"]["device"]).unsqueeze(0)  # [1, window, F]
with torch.no_grad():
    prediction = model(x)              # 形状通常是 [1, out_dim] 或 [1, seq, out_dim]
prediction = prediction.cpu().numpy()


# === prepare plots (robust) ===
plot_range = 10                      # 想展示的最后N天 + 明天
assert plot_range >= 2               # 至少要有 (N-1) 个历史点 + 1 个明天

window_size = config["data"]["window_size"]

# 1) 反归一化（确保二维输入再拉平）
y_val_inv   = scaler.inverse_transform(np.asarray(data_y_val).reshape(-1, 1)).ravel()
pred_val_inv= scaler.inverse_transform(np.asarray(predicted_val).reshape(-1, 1)).ravel()

# 对“明天”的预测做反归一化；prediction 可能是 [1] / [1,1] / [seq,1]，统一成标量
pred_next_inv_arr = scaler.inverse_transform(np.asarray(prediction).reshape(-1, 1)).ravel()
pred_next_inv = float(pred_next_inv_arr[-1])  # 取最后一个作为“下一天”的点

# 2) 历史展示长度（N-1），与实际可用长度对齐
hist_len = min(plot_range - 1, len(y_val_inv), len(pred_val_inv))
assert hist_len >= 1, f"hist_len too small: {hist_len}, check data_y_val/predicted_val lengths"

# 3) 日期对齐：验证段起点 = offset + window + split_index
val_start = offset + window_size + split_index
dates_val_full = np.asarray(data_date)[val_start: val_start + len(y_val_inv)]
assert len(dates_val_full) == len(y_val_inv), "Dates and y_val_inv length mismatch"

# 取验证段最后 hist_len 天的日期
dates_hist = dates_val_full[-hist_len:]

# 4) 组装绘图数组：长度 = hist_len + 1（+1 是“明天”）
to_plot_data_y_val       = np.full(hist_len + 1, np.nan)
to_plot_data_y_val_pred  = np.full(hist_len + 1, np.nan)
to_plot_data_y_test_pred = np.full(hist_len + 1, np.nan)

# 填充最后 hist_len 天的真实值 & 过去预测
to_plot_data_y_val[:hist_len]      = y_val_inv[-hist_len:]
to_plot_data_y_val_pred[:hist_len] = pred_val_inv[-hist_len:]

# 最后一个位置放“明天”的预测
to_plot_data_y_test_pred[-1] = pred_next_inv

# 5) 构造 x 轴日期（在最后加上一个“tomorrow”占位标签）
plot_date_test = list(dates_hist) + ["tomorrow"]

# 6) 画图
fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))

plt.plot(plot_date_test, to_plot_data_y_val,       label="Actual prices",              marker=".", markersize=10, color=config["plots"]["color_actual"])
plt.plot(plot_date_test, to_plot_data_y_val_pred,  label="Past predicted prices",      marker=".", markersize=10, color=config["plots"]["color_pred_val"])
plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price (tomorrow)", marker=".", markersize=20, color=config["plots"]["color_pred_test"])

plt.title("Predicting the close price of the next trading day")
plt.grid(True, which='major', axis='y', linestyle='--')
plt.legend()
plt.show()

print("Predicted close price of the next trading day:",
      np.round(to_plot_data_y_test_pred[-1], 2))

