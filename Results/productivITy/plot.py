import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_excel('ddpg3.xlsx', index_col=0)

# print(df)
font = {'family': 'normal',
        'size': 14}

matplotlib.rc('font', **font)

df.plot(color=['C0', 'C1', 'C2', 'C7'], marker='o', markersize=6, linewidth=3, figsize=(17, 7))

plt.grid()
plt.xlim([0, 1000])
plt.ylim([0, 80])

plt.legend(loc=2)

plt.xlabel("Anzahl Trainingsiterationen", fontsize=14)
plt.ylabel("Anzahl fehlerfreier Aktionen", fontsize=14)

plt.tight_layout()
plt.savefig("Plot3.png", dpi=600)


plt.show()
