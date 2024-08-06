import numpy as np
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import LSM

np.random.seed(seed=0)
nest.ResetKernel()

# ローレンツ方程式の定義
def lorenz(x, y, z, sigma, rho, beta):
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt

# ルンゲ・クッタ法（RK4）による数値積分の関数
def rk4_method(lorenz_func, initial_conditions, params, dt, num_steps):
    x, y, z = initial_conditions
    sigma, rho, beta = params

    # 結果を保存するための配列
    xs = np.empty(num_steps)
    ys = np.empty(num_steps)
    zs = np.empty(num_steps)

    # 初期条件をセット
    xs[0], ys[0], zs[0] = x, y, z

    # RK4による数値積分
    for i in range(num_steps-1):
        k1x, k1y, k1z = lorenz_func(xs[i], ys[i], zs[i], sigma, rho, beta)
        k2x, k2y, k2z = lorenz_func(xs[i] + 0.5 * k1x * dt, ys[i] + 0.5 * k1y * dt, zs[i] + 0.5 * k1z * dt, sigma, rho, beta)
        k3x, k3y, k3z = lorenz_func(xs[i] + 0.5 * k2x * dt, ys[i] + 0.5 * k2y * dt, zs[i] + 0.5 * k2z * dt, sigma, rho, beta)
        k4x, k4y, k4z = lorenz_func(xs[i] + k3x * dt, ys[i] + k3y * dt, zs[i] + k3z * dt, sigma, rho, beta)

        xs[i + 1] = xs[i] + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        ys[i + 1] = ys[i] + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        zs[i + 1] = zs[i] + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)

    return xs, ys, zs


# 正規化関数の定義
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


if __name__ == '__main__':
    nest.SetKernelStatus({'print_time': True, 'local_num_threads': 11})
    
    # データの設定
    dt = 0.01  # タイムステップ
    T = 2000  # 訓練用
    T_test = 500  # 検証用
    sim_time = T + T_test # 総データ長
    t_eval = np.linspace(0, dt*sim_time, sim_time) # ステップ数
    
        
    # ローレンツ方程式のパラメータ
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # lorenz-初期条件
    initial_conditions = (0.0, 1.0, 1.05)
    params = (sigma, rho, beta)
    
    
    # オイラー法を使ってローレンツ方程式を解く
    xs, ys, zs = rk4_method(lorenz, initial_conditions, params, dt, sim_time)
    
    # NEST params
    pred_period = 5
    

    # 入力データの正規化
    x_normalized = normalize(xs)
    y_normalized = normalize(ys)
    z_normalized = normalize(zs)
    input_normalized = [x_normalized, y_normalized, z_normalized]
        

    # シミュレート
    model = LSM()
    model.inject_waveform(input_normalized,  model.nodes)
    
    spikes_binary = model.get_status(sim_time)
    
    print(spikes_binary.shape)
    # print('spikes_times', spikes_binary[1][:50])
    train_data = spikes_binary[:, :T-pred_period].T
    train_targets = x_normalized[pred_period:T]
    test_data = spikes_binary[:, T:sim_time - pred_period].T
    test_targets = x_normalized[T+pred_period:]
    
    # スパイクデータの前処理
    # Kerasモデルの入力に適した形に変換（ここでは2次元配列に変換）
    input_dim = train_data.shape[1]

    # モデルの定義
    readout = keras.Sequential([
        layers.Dense(1, activation='sigmoid', input_shape=(input_dim,))
    ])

    # 重みの初期化
    initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
    readout.layers[0].kernel_initializer = initializer

    # モデルのコンパイル
    readout.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['mean_squared_error'])

    readout.summary()


    # モデルの訓練
    readout.fit(train_data, train_targets, epochs=100)

    # モデルの評価
    loss, mse = readout.evaluate(test_data, test_targets)
    print(f"Loss: {loss}, Mean Squared Error: {mse}")
    
    # モデルによる予測
    predictions = readout.predict(test_data)
    



    # グラフ表示
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(t_eval[T+pred_period:], x_normalized[T+pred_period:], linestyle=':')
    ax1.plot(t_eval[T+pred_period:], predictions)
    
    # lorenzのプロット
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_normalized, y_normalized, z_normalized)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    plt.plot(t_eval, x_normalized, label='X')
    plt.legend()
    
    ax1 = fig.add_subplot(3, 1, 2)
    plt.plot(t_eval, y_normalized, label='Y')
    plt.legend()
    
    ax1 = fig.add_subplot(3, 1, 3)
    plt.plot(t_eval, z_normalized, label='Z')    
    plt.legend()

    plt.show()
