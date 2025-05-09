import matplotlib.pyplot as plt
import numpy as np
from rrSQI import rrSQI
from ppg_SQI import ppg_SQI
import wfdb
from processing import scale_ppg_score
from QRS import peak_detection


path = "mimic3wdb/30/3000393"
segment = "3000393_0045"
start = 7500

seg_sig = wfdb.rdrecord(segment, pn_dir=path)
sig_ppg_index = seg_sig.sig_name.index('PLETH')
sig_ppg = seg_sig.p_signal[:, sig_ppg_index][start:start+3750]



sig_ecg_index = seg_sig.sig_name.index('II')
sig_ecg = seg_sig.p_signal[:, sig_ecg_index][start:start+3750]

qua_ppg = ppg_SQI(sig_ppg)
qua_ppg = scale_ppg_score(qua_ppg)

peaks = peak_detection(sig_ecg, 125)
_, _, qua_ii = rrSQI(sig_ecg, peaks, 125)

qua = (qua_ii + qua_ppg) / 2

fs = 125
t = np.arange(3750) / fs

plt.figure(figsize=(12, 6))

plt.suptitle(f"Average Signal Quality: {qua:.3f}", fontsize=16)

# 绘制 PPG 信号
plt.subplot(2, 1, 1)
plt.plot(t, sig_ppg, color='blue')
plt.title(f'PPG Signal (PLETH) - Quality: {qua_ppg:.3f}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# 绘制 ECG 信号
plt.subplot(2, 1, 2)
plt.plot(t, sig_ecg, color='red')
plt.title(f'ECG Signal (Lead II) - Quality: {qua_ii:.3f}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.savefig('ecg_ppg_plot.png')

print(f"该PPG信号综合质量为: {qua:.3f}")


