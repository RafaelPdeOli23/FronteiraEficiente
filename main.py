import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import yfinance as yf
import _datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#Pega o histórico de cotações e cria um DataFrame()
yf.pdr_override()

start = dt.datetime(2015, 1, 1)
assets = ['ITUB4.SA', 'WEGE3.SA', 'PETR4.SA', 'VALE3.SA', 'TAEE11.SA']
assets_prices = pd.DataFrame()

for a in assets:
    assets_prices[a] = wb.get_data_yahoo(a, start=start)['Adj Close']

#Cria uma matriz de retornos diários em log
log_returns = np.log(assets_prices / assets_prices.shift(1))

#Calcula a média de retornos anual e a covariância anual
mean_log_returns = log_returns.mean() * 250
cov_log_returns = log_returns.cov() * 250

#'return_risk_free' = Taxa selic (10,75%)
return_risk_free = 0.1075
num_assets = len(assets)
num_pfolios = 100000

#Cria matrizes dos retornos, volatilidades, índices sharp e pesos dos ativos de diferentes carteiras
array_returns = np.zeros(num_pfolios)
array_volatilities = np.zeros(num_pfolios)
array_sharpe = np.zeros(num_pfolios)
array_weights = np.zeros((num_pfolios, num_assets))

for p in range(num_pfolios):
    #Pesos dos ativos em cada carteira são definidos randomicamente
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    array_weights[p, :] = weights

    array_returns[p] = np.sum(weights * mean_log_returns)
    array_volatilities[p] = np.sqrt(np.dot(weights.T, np.dot(cov_log_returns, weights)))

    array_sharpe[p] = (array_returns[p] - return_risk_free) / array_volatilities[p]

#Matrizes com pesos de cada ativo em diferentes carteiras
itau_weights = []
wege_weights = []
petro_weights = []
vale_weights = []
taee_weights = []

for i in range(num_pfolios):
    itau_weights.append(array_weights[i][0])
    wege_weights.append(array_weights[i][1])
    petro_weights.append(array_weights[i][2])
    vale_weights.append(array_weights[i][3])
    taee_weights.append(array_weights[i][4])

#Transforma o retorno log em retorno art
array_returns_art = np.exp(array_returns) - 1

#Cria um DataFrame() contendo o índice sharp, retorno, volatilidade, e pesos de cada carteira
portifolios = pd.DataFrame({'Sharp Ratio': array_sharpe, 'Returns': array_returns_art, 'Volatilities': array_volatilities,
                            'Itau': itau_weights, 'Wege': wege_weights, 'Petro': petro_weights, 'Vale': vale_weights,
                            'Taee': taee_weights})

#Cria um arquivo Excel com todoas as carteiras criadas
portifolios.to_excel('Portifolios.xlsx', sheet_name='Portifolios')
efficient_frontier = portifolios[['Returns', 'Volatilities']]

#Pega a carteira ideal com base no maior índice Sharp
sharp_index_max = array_sharpe.argmax()
ideal_sharp = array_sharpe[sharp_index_max]
ideal_weights = array_weights[sharp_index_max]
ideal_return = array_returns_art[sharp_index_max]
ideal_volatility = array_volatilities[sharp_index_max]

#Printa a carteira ideal
print(f'\nSharp Ratio: {ideal_sharp:.2f}\n'
      f'Return: {(ideal_return * 100):.2f} %\n'
      f'Volratilitiy: {ideal_volatility:.2f}')

for i in range(num_assets):
    stock = assets[i]
    weight = ideal_weights[i]

    print(f'{stock}: {(weight * 100):.2f} %')

#Cria um gráfico de dispersão contendo todas as carteiras obtidas, formando a fronteira eficiente e destaca a carteira ideal em vermelho
efficient_frontier.plot(x='Volatilities', y='Returns', kind='scatter', figsize=(10, 6), c=array_sharpe)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.subplot().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.subplot().scatter(ideal_volatility, ideal_return, c='red')
plt.savefig('Efficient Frontier')
plt.show()
