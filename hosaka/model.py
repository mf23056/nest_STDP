import numpy as np
import matplotlib.pyplot as plt
import nest
from scipy.integrate import solve_ivp
import nest.raster_plot



class LSM:
    def __init__(self):
        self.nodes_E, self.nodes_I = self.create_iaf_psc_exp()
        self.connect_stdp(self.nodes_E, self.nodes_I)


    def create_iaf_psc_exp(self, n_E = 1000, n_I = 250): #https://nest-simulator.readthedocs.io/en/v3.8_rc2/models/iaf_psc_alpha.html
        self.nodes = nest.Create('iaf_psc_alpha', n_E + n_I,
                            {'C_m': 30.0,  # 1.0,
                            'I_e': 13.5, #####
                            'tau_m': 30.0,  # Membrane time constant in ms
                            'E_L': 0.0,
                            'V_th': 15.0,  # Spike threshold in mV
                            'tau_syn_ex': 3.0,
                            'tau_syn_in': 2.0,
                            'V_reset': 13.8,
                            't_ref': 3.0})

        return self.nodes[:n_E], self.nodes[n_E:]
        
        
    def connect_stdp(self, nodes_E, nodes_I):
        # params
        connect_prob = 0.2
        W_EE = [0.0,1.0]
        W_EI = [0.0,1.0]
        W_IE = [-1.0,0.0]
        W_II = [-1.0,0.0]
        
        nest.SetDefaults('stdp_synapse_hom',{'mu_plus': 0.0, 'mu_minus': 0.0})
        
        def connect(src, trg, W):
            
            nest.Connect(src, trg,
                        conn_spec={'rule': 'pairwise_bernoulli', 'p': connect_prob},
                        syn_spec={'synapse_model': 'stdp_synapse_hom',
                                  'weight': nest.random.uniform(W[0],W[1]),
                                 })
            
        connect(nodes_E, nodes_E, W_EE)
        connect(nodes_E, nodes_I, W_EI)
        connect(nodes_I, nodes_E, W_IE)
        connect(nodes_I, nodes_I, W_II)
                
        

    def inject_waveform(self, amplitudes:list, neuron_target, w_min=0, w_max=1, N=200):
        '''
        param amplitudes: 入力データ(時系列)
        param neuron_target: ターゲットニューロン
        param w_min: シナプス重みの最小値
        param w_max: シナプス重みの最大値
        param N: リザバー層に接続するシナプス数
        '''
        generators = nest.Create('step_current_generator', len(amplitudes))
        
        for i, current in enumerate(amplitudes):
        
            times = [float(i+1) for i in range(len(current))]

            nest.SetStatus(generators[i] ,params={"amplitude_values":current, "amplitude_times":times})

            nest.Connect(generators[i], neuron_target,
                        {'rule': 'fixed_total_number', 'N': N},
                        {"synapse_model": "static_synapse", 
                        "weight": nest.random.uniform(min=w_min,max=w_max)})
            

    
    def visualize_networks(self, M):
        # extract position information, transpose to list of x, y and z positions
        xpos, ypos, zpos = zip(*nest.GetPosition(M))
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection="3d")
        ax1.scatter(xpos, ypos, zpos, s=15, facecolor="b")
        plt.show()


    def get_status(self, sim_time):        
        # スパイクデータの取得
        sr = nest.Create("spike_recorder")
        nest.Connect(self.nodes, sr)
        
        # SIMULATE
        nest.Simulate(sim_time)

        # スパイクデータの取得
        spike_events = sr.get("events")
        
        # スパイクイベントから送信者IDと時刻を取得
        senders = spike_events['senders']
        times = spike_events['times']

        # シミュレーションの最大時間とニューロンの最大IDを取得
        max_time = sim_time
        max_neuron_id = senders.max()

        # スパイク行列をゼロで初期化
        spike_binary = np.zeros((max_neuron_id, max_time))
        print('max_neuron_id', max_neuron_id)

        # スパイクイベントを行列に変換
        for sender, time in zip(senders, times):
            spike_binary[sender-1, int(np.floor(time-1))] = 1

        # ラスタープロットの生成
        nest.raster_plot.from_device(sr, hist=False)

        return spike_binary