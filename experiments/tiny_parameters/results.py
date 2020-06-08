import numpy as np
import matplotlib.pyplot as plt
Adam_res = []
SGD_res = []
for i in range(1,10,1):
    res = np.loadtxt(f"Adam{i}/scores_Adam_{i}.txt")
    Adam_res.append(res)
for i in range(1,10,1):
    res = np.loadtxt(f"SGD{i}/scores_SGD_{i}.txt")
    SGD_res.append(res)

#print(Adam_res[0][:,0])
plt.figure(figsize=(16,8))
for i in range(9):
    plt.plot(Adam_res[i][:,0], Adam_res[i][:,1], label=f"Adam{i}")
plt.title("Adam - mAP")
plt.legend()
plt.savefig("Adam_mAP.png")
plt.figure(figsize=(16,8))
for i in range(9):
    plt.plot(Adam_res[i][:,0], Adam_res[i][:,2], label=f"Adam{i}")
plt.title("Adam - Loss")
plt.legend()
plt.savefig("Adam_Loss.png")

plt.figure(figsize=(16,8))
for i in range(9):
    plt.plot(SGD_res[i][:,0], SGD_res[i][:,1], label=f"SGD{i}")
plt.title("SGD - mAP")
plt.legend()
plt.savefig("SGD_mAP.png")

plt.figure(figsize=(16,8))
for i in range(9):
    plt.plot(SGD_res[i][:,0], SGD_res[i][:,2], label=f"SGD{i}")
plt.title("SGD - Loss")
plt.legend()
plt.savefig("SGD_Loss.png")