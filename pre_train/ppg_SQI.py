import numpy as np
from scipy import signal, stats
import wfdb
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ppg_SQI(ppg, fs=125):

    # 预处理：去趋势和归一化
    ppg = signal.detrend(ppg)
    ppg_normalized = (ppg - np.mean(ppg)) / np.std(ppg)

    try:
        # 1. 信号功率质量指标
        f, Pxx = signal.welch(ppg_normalized, fs=fs, nperseg=min(len(ppg), 256))
        signal_power = np.sum(Pxx[(f > 0.5) & (f < 5)])  # PPG主要频率范围
        noise_power = np.sum(Pxx[(f > 5) & (f < 25)])  # 高频噪声范围
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        snr_score = 1 / (1 + np.exp(-0.5 * (snr - 5)))


        peaks, _ = signal.find_peaks(ppg_normalized, distance=fs // 2, prominence=0.5)  # 增加prominence阈值
        if len(peaks) < 2:
            return 0.0

        # 2. 灌注指数
        ac_component = np.max(ppg) - np.min(ppg)
        dc_component = np.mean(ppg)
        perfusion_index = ac_component / dc_component
        perfusion_score = np.clip(perfusion_index * 10, 0, 1)  # 归一化到0-1

        # 3. 信号偏度
        skewness = stats.skew(ppg_normalized)
        skewness_score = 1 / (1 + np.exp(-5 * (abs(skewness) - 1)))  # 理想PPG偏度接近0

        # 4. 相对功率
        total_power = np.sum(Pxx[(f > 0.1) & (f < 50)])
        rel_power = signal_power / (total_power + 1e-10)
        rel_power_score = np.clip(rel_power * 2, 0, 1)  # 归一化

        # 5. 节律性
        rr_intervals = np.diff(peaks) / fs * 1000
        rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
        rhythm_score = 1 / (1 + rr_cv)

        # 6. 幅度变化
        peak_amplitudes = ppg_normalized[peaks]
        amp_cv = np.std(peak_amplitudes) / np.mean(peak_amplitudes)
        amp_score = 1 / (1 + amp_cv)

        # 7. 自相关
        autocorr = np.correlate(ppg_normalized, ppg_normalized, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr_score = np.mean(autocorr[:fs] / autocorr[0])

        # 8. 熵
        hist, _ = np.histogram(ppg_normalized, bins=20)
        hist = hist / np.sum(hist)
        entropy = stats.entropy(hist)
        entropy_score = 1 - (entropy / np.log(20))

        # 综合所有指标
        quality_score = (
                0.2 * snr_score +
                0.2 * rhythm_score +
                0.1 * amp_score +
                0.1 * autocorr_score +
                0.1 * entropy_score +
                0.1 * perfusion_score +
                0.1 * skewness_score +
                0.1 * rel_power_score
        )

        print(f"SNR Score: {snr_score}")
        print(f"Rhythm Score: {rhythm_score}")
        print(f"Amplitude Score: {amp_score}")
        print(f"Autocorrelation Score: {autocorr_score}")
        print(f"Entropy Score: {entropy_score}")
        print(f"Perfusion Score: {perfusion_score}")
        print(f"Skewness Score: {skewness_score}")
        print(f"Relative Power Score: {rel_power_score}")

        return np.clip(quality_score, 0, 1)

    except Exception as e:
        print(f"Error in quality assessment: {e}")
        return 0.0


# 测试代码
# path = "mimic3wdb/30/3000393"
# segment = "3000393_0002"

# # 读取PPG信号
# seg_sig = wfdb.rdrecord(segment, pn_dir=path)
# sig_ppg_index = seg_sig.sig_name.index('PLETH')
# sig_ppg = seg_sig.p_signal[:, sig_ppg_index][7500:7500+3750]

# # 绘制信号
# plt.figure(figsize=(12, 4))
# plt.plot(sig_ppg)
# plt.title('PPG Signal')
# plt.close()

# # 评估信号质量
# quality = ppg_SQI(sig_ppg)
# print(f"该PPG信号综合质量为: {quality:.3f}")