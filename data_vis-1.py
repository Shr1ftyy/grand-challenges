# %%
from cProfile import label
import matplotlib.pyplot as plt
import matplotlib.ticker as pltTicker
import imgkit
import pandas
import numpy as np
from datetime import datetime as dt
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests 

totalTweets = 0
tickers = ["AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "NKE", "GS", "IBM", "INTC", "KO", "MMM", "MSFT", "V", "XOM"]
# 'MRK', 'NKE', 'PFE', 'PG', 'TRV', "UNH", "UTX", "VZR", 'WMT', 'WBA', 'DIS']
p_p = {}
avgVolumeP = 0
avgSentimentP = 0
allPolarities = []
earningsCount = 0
spikeCount = 0
POS_CAR = [0 for _ in range(16)]
NEUTRAL_CAR = [0 for _ in range(16)]
NEG_CAR = [0 for _ in range(16)]
CUM_VARIANCE = 0
VAR_CAR= 0

f = open('pearsons.csv', 'w')
f.write('"Ticker", "p(T_d, |R_d|)", "p(P_d, R_d)", "T_d, |R_d|", "P_d, R_d"\n')
f.close()
  # %% [markdown]
  # # STOCK PRICES

  
  # %%
for ticker in tickers:
  # c = pandas.read_csv('prices/KO Historical Data.csv')
  # c = pandas.read_csv('S&P 500 Historical Data.csv')
  t = yf.Ticker(ticker)
  fullYr = t.history(start="2019-02-28", end="2019-12-29")
  prevYr = t.history(start="2018-02-28", end="2018-12-19")

  prevYr.insert(1, "Change %", [x for x in prevYr['Close']], True)
  prevYr["Change %"] = 100 * (prevYr["Close"].pct_change(periods=1))
  prevYr = prevYr.fillna(0)

  fullYr.insert(1, "Change %", [x for x in fullYr['Close']], True)
  fullYr["Change %"] = 100 * (fullYr["Close"].pct_change(periods=1))
  fullYr = fullYr.fillna(0)
  fullYr = fullYr.reindex(pandas.date_range("2019-02-28", "2019-12-29"), fill_value=0)

  c = fullYr["2019-02-28":"2019-12-17"]



  try: 
    c = c.drop('2019-02-27')
    prevYr = prevYr.drop('2019-02-27')
  except:
    pass

  meanRet = prevYr["Change %"].mean()
  varRet = (prevYr["Change %"] - meanRet).var()
  CUM_VARIANCE += float(varRet)
  print("HERE: ", CUM_VARIANCE)


  pcts = list(c['Change %'])
  vols = list(c['Volume'])
  earnings = []
  stockDates = [x.strftime("%Y-%m-%d") for x in c.index]
  print(stockDates)


  # pcts = [float(x.replace("%", "").replace(",","")) for x in pct_change]


  # %% [markdown]
  # # TWITTER DATA

  # %%
  tweets = pandas.read_json(open(f'/Users/shr1ftyy/Desktop/Uni/Y1/SEM2/GRAND_CHALLENGES/Project1/data/tweet{ticker}RNN.json', 'r', encoding='utf8'))
  tweets = tweets.transpose()
  # tweets.reset_index(drop=True, inplace=True)
  print(tweets)

  tweets = tweets.reset_index(drop=True)
  print(tweets)
    
  td = list(tweets['Date'])
  maxLen = len(td)
  d = 0 
  while(d < maxLen):
    if td[d] not in stockDates:
      print(td[d])
      print(stockDates[d])
      # stockDates.pop(d)
      stockDates.insert(d, tweets['Date'][d])
      pcts.insert(d, 0)
      vols.insert(d, 0)
      # pcts.pop(d)
      d = d - 1
      # maxLen += 1
    d += 1
  

  print([pcts])

  print(f"Ticker: {ticker}")
  rawDates = t.earnings_dates.index
  earnDates = [x.strftime("%Y-%m-%d") for x in rawDates]
  earnings = []
  for x in stockDates:
    if x in earnDates:
      earnings.append(x)

  convVolume = tweets.set_index('Date')

  # Highlight Earnings Events
  earnMarkers = []
  for x in earnings:
    try:
      earnMarkers.append(float(convVolume['volume'][x]))
    except:
      earnMarkers.append(0)

  earningsCount += len(earnMarkers)
  
  tweets.insert(1, "abs. price %", [abs(float(x)) for x in pcts])
  tweets.insert(1, "price %", [float(x) for x in pcts])
  tweets.insert(1, "trade volume", [float(x) for x in vols])
  tweets['volume'] = [float(x) for x in tweets['volume']]
  tweets['sentiment'] = [float(x) for x in tweets['sentiment']]
  print(tweets)

  # %%
  # Spike Events
  meanVol = tweets['volume'].rolling(window=14).median()
  stdVol = tweets['volume'].rolling(window=14).std()
  mask = tweets['volume'] > meanVol + 3 * stdVol
  print(mask)
  spikes = tweets[mask]
  spikes.reset_index(drop=True, inplace=True)
  numSpikes = len(list(spikes))
  spikeCount += numSpikes
  polarities = list(spikes['sentiment'])

  posReturns = [0 for _ in range(16)]
  neuReturns = [0 for _ in range(16)]
  negReturns = [0 for _ in range(16)]

  tweets.reset_index(drop=True, inplace=True)

  numPos = 0
  numNeu = 0
  numNeg = 0

  # Categorization of Events
  dateList = list(fullYr.index) 
  dateSpikes = [pandas.Timestamp(x) for x in spikes['Date']]
  for d in dateSpikes:
    di = list(dateSpikes).index(d)
    polar = spikes['sentiment'][di]
    cumAR = 0
    if polar < 0.2:
      numNeg += 1
    elif polar > 0.4:
      numPos += 1
    else:
      numNeu += 1

    ind = 0
    for offset in range(-5, 11):
      dateIndex = dateList[dateList.index(d) + offset]
      cumAR += fullYr['Change %'][dateIndex] - meanRet
      print(cumAR)
      if polar < 0.2:
        negReturns[ind] += cumAR
      elif polar > 0.4:
        posReturns[ind] += cumAR
      else:
        neuReturns[ind] += cumAR

      ind += 1

  if numPos > 0 :
    posReturns = [x/numPos for x in posReturns]

  if numNeu > 0 :
    neuReturns = [x/numNeu for x in neuReturns]

  if numNeg > 0 :
    negReturns = [x/numNeg for x in negReturns]

  print("RETURN STATS:")
  print(numNeg)
  print(negReturns)

  for i in range(len(posReturns)):
    POS_CAR[i] += posReturns[i]
    NEG_CAR[i] += negReturns[i]
    NEUTRAL_CAR[i] += neuReturns[i]
  

  # Plotting
  fig, (ax, ax3) = plt.subplots(2, 1)
  # fig2, ax2 = plt.subplots(1, 1)
  ax.plot_date(stockDates, pcts, linestyle="-", marker="", color="red", label="% Change (Price)")
  ax2 = ax.twinx()
  ax4 = ax3.twinx()
  ax2.plot_date(tweets['Date'], tweets['sentiment'], linestyle="-", marker="", color="deepskyblue", label="Polarity")
  ax3.plot_date(tweets['Date'], tweets['volume'], linestyle="-", marker="", color="lightgreen", label="Tweet Volume")
  ax4.plot_date(stockDates, vols, linestyle="-", marker="", color="orange", label="Trade Vol.")
  ax3.plot_date(earnings, earnMarkers, marker="o", color="k", label="EA events")
  ax3.plot_date(spikes['Date'], spikes['volume'], marker="*", color="red", label="Spikes")
  tick_spacing = 60
  # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
  plt.setp( ax.get_xticklabels(), visible=False)
  ax3.xaxis.set_major_locator(pltTicker.MultipleLocator(tick_spacing))
  ax4.xaxis.set_major_locator(pltTicker.MultipleLocator(tick_spacing))
  ax2.xaxis.set_major_locator(pltTicker.MultipleLocator(tick_spacing))
  # plt.yticks(np.arange(-1, 1, 25)) 
  ax.set_title(f"(${ticker}) Tweet Sentiment Polarity vs Daily % Change")
  ax.set_ylabel("Daily % Change")
  ax3.set_xlabel("Date (yy-mm-dd)")
  ax2.set_ylabel("Polarity")
  ax3.set_ylabel("Tweet Volume")
  ax4.set_ylabel("Trading Volume")
  ax.legend()
  ax2.legend()
  ax3.legend()
  ax4.legend(loc='upper left')
  fig.set_size_inches(10,5)
  # plt.show()

  print(pcts)
  print(len(pcts))

  # Granger Cause Test
  resVol = grangercausalitytests(tweets[['volume', 'abs. price %']], maxlag=[3])
  vol_p_val = resVol[3][0]['ssr_ftest'][1]

  resPolar = grangercausalitytests(tweets[['sentiment', 'price %']], maxlag=[3])
  polar_p_val = resPolar[3][0]['ssr_ftest'][1]

  coeff = tweets[['abs. price %', 'price %', 'volume', 'sentiment']].corr()
  coeff.to_csv(f'imgs/{ticker}_coeff.csv') 

  # Pearson's Coeefficient
  renderPass = coeff.style.background_gradient(cmap='coolwarm').set_precision(3)
  rendered = renderPass.render()
  imgkit.from_string(rendered, f"imgs/{ticker}_coeff.png")
  print(coeff)
  p_p[ticker] = coeff
  plt.savefig(f"imgs/{ticker}.png", format="png", dpi=200)
  avgVolumeP += coeff['volume'][0]
  avgSentimentP += coeff['sentiment'][1]
  f = open('pearsons.csv', 'a')
  f.write(f"{ticker}, {coeff['volume'][0]}, {coeff['sentiment'][1]}, {vol_p_val}, {polar_p_val}\n")
  f.close()
  for x in polarities:
    allPolarities.append(round(x, 2))

allPolarities = pandas.DataFrame(allPolarities)
allPolarities.to_csv('polarities.csv')
avgVolumeP = avgVolumeP/len(tickers)
avgSentimentP = avgSentimentP/len(tickers)
print(avgVolumeP)
print(avgSentimentP)

polarFreq = {}
for x in allPolarities:
  if x not in list(polarFreq.keys()):
    polarFreq[x] = 0
  else:
    polarFreq[x] += 1

plt.show()
plt.figure(69)
plt.bar(list(polarFreq.keys()), list(polarFreq.values()))
plt.savefig('imgs/polar.png')

# Event stats
POS_CAR = [x/len(tickers) for x in POS_CAR]
NEG_CAR = [x/len(tickers) for x in NEG_CAR]
NEUTRAL_CAR = [x/len(tickers) for x in NEUTRAL_CAR]
print(CUM_VARIANCE)
# CAR Variance
CAR_VAR = (len(POS_CAR)) * (CUM_VARIANCE)/(spikeCount**2)


# Visualisation of CAR vs Lag days
plt.figure(420)
plt.plot([x for x in range(-5, 11)], POS_CAR, color='green', linestyle='-', marker='.', label="Pos. Event")
plt.plot([x for x in range(-5, 11)], NEUTRAL_CAR, color='blue', linestyle='-', marker='.', label="Neu. Event")
plt.plot([x for x in range(-5, 11)], NEG_CAR, color='red', linestyle='-', marker='.', label="Neg. Event")
plt.xlabel("Lag (Days from spike event)")
plt.ylabel("CAR (%)")
plt.legend(loc="upper left")
plt.savefig('CAR.png')
plt.show()

print(POS_CAR)
print(NEG_CAR)
print(CAR_VAR)

print(f"{earningsCount}")
print(f"{spikeCount}")
print(f"earnings spike rate: {(earningsCount/spikeCount) * 100}%")
print(totalTweets)
# %%
