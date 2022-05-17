import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

cwd = os.path.join(os.getcwd(), 'SEEM2460_Project_RNN')
os.chdir(cwd)
whole = pd.read_csv('BTC-USD.csv')
dataframe = pd.read_csv('BTC-USD.csv').iloc[:, 1:]
sns.heatmap(dataframe.corr(), annot=True)
plt.show()

plt.title('BTC-USD')
plt.plot(range(len(whole.iloc[:, 0])), whole.iloc[:, 5])
plt.xlabel('Days')
plt.ylabel('Adj Price')
plt.show()

plt.title('BTC-USD')
plt.plot(range(len(whole.iloc[:361, 0])), whole.iloc[:361, 5], label='training')
plt.plot(range(365, 365 + len(whole.iloc[366:, 0])), whole.iloc[366:, 5], label='testing', color='g')
plt.xlabel('Days')
plt.ylabel('Adj Price')
plt.legend(['Training', 'Testing'])
plt.show()
