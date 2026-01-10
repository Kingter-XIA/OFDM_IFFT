import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class QAM16_1Hz_Visualizer:
    def __init__(self, symbol_duration=1.0, sampling_rate=100):
        # 16QAM Gray 映射
        a = 1 / np.sqrt(10)
        self.qam16_map = {
            (0,0,0,0): (-3*a, -3*a),
            (0,0,0,1): (-3*a, -1*a),
            (0,0,1,1): (-3*a, +1*a),
            (0,0,1,0): (-3*a, +3*a),

            (0,1,0,0): (-1*a, -3*a),
            (0,1,0,1): (-1*a, -1*a),
            (0,1,1,1): (-1*a, +1*a),
            (0,1,1,0): (-1*a, +3*a),

            (1,1,0,0): (+1*a, -3*a),
            (1,1,0,1): (+1*a, -1*a),
            (1,1,1,1): (+1*a, +1*a),
            (1,1,1,0): (+1*a, +3*a),

            (1,0,0,0): (+3*a, -3*a),
            (1,0,0,1): (+3*a, -1*a),
            (1,0,1,1): (+3*a, +1*a),
            (1,0,1,0): (+3*a, +3*a),
        }
        
        self.symbol_duration = symbol_duration
        self.sampling_rate = sampling_rate
        self.carrier_freq = 1.0  # 1Hz载波频率
        
        # 单个符号的时间轴
        self.t_symbol = np.linspace(0, symbol_duration, int(sampling_rate * symbol_duration), endpoint=False)
        
        # 16个符号的连续时间轴
        self.t_total = np.linspace(0, 16 * symbol_duration, 16 * int(sampling_rate * symbol_duration), endpoint=False)
    
    def generate_16qam_symbol(self, bits):
        I, Q = self.qam16_map[bits]
        
        # 生成载波
        carrier_I = I * np.cos(2 * np.pi * self.carrier_freq * self.t_symbol)
        carrier_Q = Q * np.sin(2 * np.pi * self.carrier_freq * self.t_symbol)
        
        # 16QAM调制信号
        modulated_signal = carrier_I - carrier_Q
        
        return modulated_signal, carrier_I, carrier_Q, I, Q
    
    def generate_continuous_16qam(self, symbol_sequence):
        #生成连续的16QAM信号
        continuous_signal = np.array([])
        I_components = np.array([])
        Q_components = np.array([])
        
        for bits in symbol_sequence:
            modulated, carrier_I, carrier_Q, _, _ = self.generate_16qam_symbol(bits)
            continuous_signal = np.concatenate([continuous_signal, modulated])
            I_components = np.concatenate([I_components, carrier_I])
            Q_components = np.concatenate([Q_components, carrier_Q])
            
        return continuous_signal, I_components, Q_components
    
    def plot_continuous_16qam(self):
        #绘制连续的16个16QAM符号波形
        # 定义16个符号的序列 - 这里用自然顺序展示跳变，而非严格的Gray顺序...
        symbol_sequence = [(0, 0, 0, 0), 
                           (0, 0, 0, 1), 
                           (0, 0, 1, 0),
                           (0, 0, 1, 1), 
                           (0, 1, 0, 0), 
                           (0, 1, 0, 1), 
                           (0, 1, 1, 0),
                           (0, 1, 1, 1), 
                           (1, 0, 0, 0), 
                           (1, 0, 0, 1), 
                           (1, 0, 1, 0),
                           (1, 0, 1, 1), 
                           (1, 1, 0, 0), 
                           (1, 1, 0, 1), 
                           (1, 1, 1, 0),
                           (1, 1, 1, 1)]
                           
        symbol_labels = ["0000", "0001", "0010", "0011","0100", "0101", "0110", "0111","1000", "1001", "1010", "1011","1100", "1101", "1110", "1111"]
        
        # 生成连续信号
        continuous_signal, I_comp, Q_comp = self.generate_continuous_16qam(symbol_sequence)
        
        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1])
        
        # 主图：连续的16QAM调制信号
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.t_total, continuous_signal, 'b-', linewidth=2, label='16QAM调制信号')
        
        # 标记符号边界和符号类型
        for i in range(16):
            # 符号边界
            ax1.axvline(x=i * self.symbol_duration, color='red', linestyle='--', alpha=0.7)
            
            # 符号标签
            symbol_time = (i + 0.5) * self.symbol_duration
            ax1.text(symbol_time, ax1.get_ylim()[1] * 0.9, symbol_labels[i],
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
            
            # 相位信息
            I, Q = self.qam16_map[symbol_sequence[i]]
            angle = np.degrees(np.arctan2(Q, I))
            if angle < 0:
                angle += 360
            radius = np.sqrt(I**2 + Q**2)
            ax1.text(symbol_time, ax1.get_ylim()[0] * 0.9,  f'θ={angle:.0f}°\n|s|={radius:.2f}',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
        
        ax1.set_title('16QAM调制: 0000b~1111b的16个符号连续波形', fontsize=16, fontweight='bold')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('幅度')
        ax1.grid(False, alpha=0.3)
        ax1.legend()
        
        # 子图1：I路分量
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.t_total, I_comp, 'g-', linewidth=1.5, label='I路分量')
        
        # 标记符号边界
        for i in range(16):
            ax2.axvline(x=i * self.symbol_duration, color='red', linestyle='--', alpha=0.7)
            ax2.text((i + 0.5) * self.symbol_duration, ax2.get_ylim()[1] * 0.8, symbol_labels[i],
                    ha='center', va='center', fontsize=10)
        
        ax2.set_title('同相分量 (I)')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('幅度')
        ax2.grid(False, alpha=0.3)
        ax2.legend()
        
        # 子图2：Q路分量
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(self.t_total, Q_comp, 'r-', linewidth=1.5, label='Q路分量')
        
        # 标记符号边界
        for i in range(16):
            ax3.axvline(x=i * self.symbol_duration, color='red', linestyle='--', alpha=0.7)
            ax3.text((i + 0.5) * self.symbol_duration, ax3.get_ylim()[1] * 0.8, symbol_labels[i],
                    ha='center', va='center', fontsize=10)
        
        ax3.set_title('正交分量 (Q)')
        ax3.set_xlabel('时间 (秒)')
        ax3.set_ylabel('幅度')
        ax3.grid(False, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
    
    
    def plot_constellation(self):
        fig, ax = plt.subplots(figsize=(8, 8))


        for bits, (I, Q) in self.qam16_map.items():
            label = "".join(str(b) for b in bits)
            ax.scatter(I, Q, s=120)
            ax.text(I, Q + 0.05, label, ha='center', fontsize=9)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title("16QAM 星座图", fontsize=16, fontweight='bold')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        plt.show()


if __name__ == "__main__":
    visualizer = QAM16_1Hz_Visualizer(symbol_duration=1.0, sampling_rate=1000)
    
    print("16QAM调制波形可视化 (1Hz载波)")
    print("=" * 50)
    print("每个符号持续1秒，使用1Hz载波，每个符号包含一个完整的正弦周期")
    
    # 显示星座图
    print("显示16QAM星座图...")
    visualizer.plot_constellation()
    
    # 显示连续波形
    print("显示16个符号的连续波形...")
    visualizer.plot_continuous_16qam()
    
    