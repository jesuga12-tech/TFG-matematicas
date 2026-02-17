import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import numpy as np

resultados = pd.read_excel("results/metrics/resultados_estadisticos_30_semillas_TEST.xlsx", index_col=0)

rsmle_LSTM = resultados.loc['lstm',"rmsle"]
rsmle_CNN = resultados.loc['cnn',"rmsle"]
rsmle_TCN = resultados.loc['tcn',"rmsle"]

statistic, p_value = friedmanchisquare(rsmle_LSTM, rsmle_CNN, rsmle_TCN)
print(f"Friedman statistic: {statistic:.4f}, p-value: {p_value:.4f}")

data = np.array([rsmle_LSTM, rsmle_CNN, rsmle_TCN])

nemenyi_results = posthoc_nemenyi_friedman(data.T)


print("\nMatriz de p-valores del test de Nemenyi:")
print("(Filas y columnas: LSTM, CNN, TCN)")
print(nemenyi_results)