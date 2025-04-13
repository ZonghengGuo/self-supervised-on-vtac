import matplotlib.pyplot as plt
import numpy as np

losses = np.load("losses_list.npy")  # 或 json.load 打开
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SimSiam Training Loss Curve")
plt.grid()
plt.show()
