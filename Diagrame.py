import gensim
import pandas as pd
import time
import numpy as np
from matplotlib.cbook import get_sample_data
import matplotlib.pyplot as plt

data_frame = pd.read_csv('train_electricity.csv', encoding="ISO-8859-1")

# print(data_frame.head(10))

d = data_frame["Date"].values.tolist()
c = data_frame["Consumption_MW"].values.tolist()
p = data_frame["Production_MW"].values.tolist()
coal = data_frame["Coal_MW"].values.tolist()
gas = data_frame["Gas_MW"].values.tolist()
hidro = data_frame["Hidroelectric_MW"].values.tolist()
nuclear = data_frame["Nuclear_MW"].values.tolist()
wind = data_frame["Wind_MW"].values.tolist()
solar = data_frame["Solar_MW"].values.tolist()
biomass = data_frame["Biomass_MW"].values.tolist()
d = sorted(d, reverse=True)

# for i in range(0, 10):
#   date = time.gmtime(time.time() - x[i])
#  print(time.strftime("%d/%m/%Y - %H:%M:%S", date))
date = time.gmtime(time.time() - min(d))
end_year = int(time.strftime("%Y", date))
date = time.gmtime(time.time() - max(d))
start_year = int(time.strftime("%Y", date))
print(start_year)
print(end_year)
print(min(p))
print(min(c))
print(max(p))
print(max(c))

prod_per_year = []
cons_per_year = []
coal_per_year = []
gas_per_year = []
nuclear_per_year = []
hidro_per_year = []
biomass_per_year = []
solar_per_year = []
wind_per_year = []
current_year = start_year
sp = 0
sc = 0
scoal = 0
sgas = 0
snuclear = 0
shidro = 0
sbiomass = 0
ssolar = 0
swind = 0
for i in range(0, len(d)):
    if int(time.strftime("%Y", time.gmtime(time.time() - d[i]))) == current_year:
        sp += p[i]
        sc += c[i]
        scoal += coal[i]
        sgas += gas[i]
        snuclear += nuclear[i]
        shidro += hidro[i]
        sbiomass += biomass[i]
        swind += wind[i]
    else:
        current_year = int(time.strftime("%Y", time.gmtime(time.time() - d[i])))
        prod_per_year.append(sp)
        cons_per_year.append(sc)
        coal_per_year.append(scoal)
        gas_per_year.append(sgas)
        nuclear_per_year.append(snuclear)
        hidro_per_year.append(shidro)
        biomass_per_year.append(sbiomass)
        solar_per_year.append(ssolar)
        wind_per_year.append(swind)
        sp = p[i]
        sc = c[i]
        scoal = coal[i]
        sgas = gas[i]
        snuclear = nuclear[i]
        shidro = hidro[i]
        sbiomass = biomass[i]
        swind = wind[i]
        ssolar = solar[i]
prod_per_year.append(sp)
cons_per_year.append(sc)
coal_per_year.append(scoal)
gas_per_year.append(sgas)
nuclear_per_year.append(snuclear)
hidro_per_year.append(shidro)
biomass_per_year.append(sbiomass)
solar_per_year.append(ssolar)
wind_per_year.append(swind)

print(max(max(prod_per_year), max(cons_per_year)))
print(prod_per_year)
print(cons_per_year)
print(min(min(coal_per_year), min(solar_per_year), min(gas_per_year), min(nuclear_per_year), min(biomass_per_year),
          min(hidro_per_year), min(wind_per_year)))
print(max(max(coal_per_year), max(solar_per_year), max(gas_per_year), max(nuclear_per_year), max(biomass_per_year),
          max(hidro_per_year), max(wind_per_year)))

# Data for plotting
t = np.arange(0.0, 9.0, 1)
s0 = np.arange(0.0, 378900000.0, 42100000)

fig, ax = plt.subplots()
l0, = ax.plot(t, s0, visible=False)
l1, = ax.plot(t, prod_per_year, lw=2, color='r', label='Production_MW')
l2, = ax.plot(t, cons_per_year, lw=2, color='g', label='Consumption_MW')
plt.subplots_adjust(left=0.2)
labels = [i for i in range(start_year, end_year + 1)]
plt.xticks(t, labels, rotation='horizontal')
lines = [l0, l1, l2]
ax.legend()
ax.set_title("Production and Consumption per years")
plt.show()

# Data for plotting
t = np.arange(0.0, 9.0, 1)
s0 = np.arange(0.0, 378900000.0, 42100000)

fig, ax = plt.subplots()
l0, = ax.plot(t, s0, visible=False)
l1, = ax.plot(t, coal_per_year, lw=2, color='r', label='Coal_MW')
l2, = ax.plot(t, gas_per_year, lw=2, color='g', label='Gas_MW')
l3, = ax.plot(t, hidro_per_year, lw=2, color='b', label='Hidroelectric_MW')
l4, = ax.plot(t, nuclear_per_year, lw=2, color='c', label='Nuclear_MW')
l5, = ax.plot(t, solar_per_year, lw=2, color='y', label='Solar_MW')
l6, = ax.plot(t, wind_per_year, lw=2, color='m', label='Wind_MW')
l7, = ax.plot(t, biomass_per_year, lw=2, color='w', label='Biomass_MW')
plt.subplots_adjust(left=0.2)
labels = [i for i in range(start_year, end_year + 1)]
plt.xticks(t, labels, rotation='horizontal')
lines = [l0, l1, l2, l3, l4, l5, l6, l7]
ax.legend()
ax.set_title("Particular production per year")
plt.show()

# Data for plotting
t = np.arange(0.0, 9.0, 1)
s0 = np.arange(0.0, 378900000.0, 42100000)

real_total_production = []
for i in range(0, len(coal_per_year)):
    mw1 = coal_per_year[i] + solar_per_year[i] + wind_per_year[i] + nuclear_per_year[i]
    mw2 = biomass_per_year[i] + gas_per_year[i] + hidro_per_year[i]
    mw = mw1 + mw2
    real_total_production.append(mw)

fig, ax = plt.subplots()
l0, = ax.plot(t, s0, visible=False)
l1, = ax.plot(t, prod_per_year, lw=2, color='r', label='Production_MW')
l2, = ax.plot(t, real_total_production, lw=2, color='g', label='Sum_Of_Particular_MW')
plt.subplots_adjust(left=0.2)
labels = [i for i in range(start_year, end_year + 1)]
plt.xticks(t, labels, rotation='horizontal')
lines = [l0, l1, l2]
ax.legend()
ax.set_title("Production difference per year")
plt.show()

# Data for plotting
real_total_production = []
for i in range(0, len(d)):
    mw1 = coal[i] + solar[i] + wind[i] + nuclear[i]
    mw2 = biomass[i] + gas[i] + hidro[i]
    mw = mw1 + mw2
    real_total_production.append(mw)

s0 = np.arange(0.0, max(real_total_production), max(real_total_production) / len(d))

fig, ax = plt.subplots()
l0, = ax.plot(d, s0, visible=False)
l1, = ax.plot(d, p, lw=2, color='r', label='Production_MW')
l2, = ax.plot(d, real_total_production, lw=2, color='g', label='Sum_Of_Particular_MW')
plt.subplots_adjust(left=0.2)
# labels = [i for i in range(start_year, end_year + 1)]
# plt.xticks(t, labels, rotation='horizontal')
lines = [l0, l1, l2]
ax.legend()
ax.set_title("Production difference")
plt.show()

# Data for plotting
s0 = np.arange(0.0, max(max(c), max(p)), max(max(c), max(p)) / len(d))

fig, ax = plt.subplots()
l0, = ax.plot(d, s0, visible=False)
l1, = ax.plot(d, p, lw=2, color='r', label='Production_MW')
l2, = ax.plot(d, c, lw=2, color='g', label='Consumption_MW')
plt.subplots_adjust(left=0.2)
lines = [l0, l1, l2]
ax.legend()
ax.set_title("Production and Consumption")
plt.show()
