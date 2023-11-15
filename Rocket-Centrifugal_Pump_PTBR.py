# -*- coding: utf-8 -*-
"""

TCC BOMBA FOGUETE.ipynb
# PACKS

"""

pip install CoolProp

from CoolProp import AbstractState
from CoolProp.CoolProp import PhaseSI, PropsSI, get_global_param_string
import CoolProp.CoolProp as CoolProp
from CoolProp.HumidAirProp import HAPropsSI
import CoolProp.Plots as CPP

import numpy as np
from numpy.linalg import solve
from numpy import matrix

import pandas as pd
import pandas_datareader.data as web

import math
from math import sin

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import statistics

# Materiais (matweb)

class Al_6063_O: # https://www.matweb.com/search/DataSheet.aspx?MatGUID=bcd1abbd8d6d47b1b9896af80a3759c6&ckck=1
  Sy = 48.3*1e6   # Limite de escoamento do material (Pa)
  Su = 89.6*1e6    # Limite de Resistência à tração (Pa)
  Sf_ = 0.4*89.6*1e6   # Resistência à fadiga não corrigida (Pa) [17]

class Al_6061_T6:
  Sy = 276*1e6   # Limite de escoamento do material (Pa)
  Su = 310*1e6    # Limite de Resistência à tração (Pa)
  Poison = 0.33   # Posion ratio
  Sf_ = 0.4*310*1e6   # Resistência à fadiga não corrigida (Pa) [17]

class Al_7075_T6:
  Sy = 503*1e6   # Limite de escoamento do material (Pa)
  Su = 572*1e6    # Limite de Resistência à tração (Pa)
  Poison = 0.33   # Posion ratio
  Sf_ = 159*1e6   # Resistência à fadiga não corrigida (Pa) [17]

class Steel_1020:   # https://www.matweb.com/search/DataSheet.aspx?MatGUID=a2eed65d6e5e4b66b7315a1b30f4b391
  Sy = 330*1e6   # Limite de escoamento do material (Pa)
  Su = 450*1e6    # Limite de Resistência à tração (Pa)
  Poison = 0.26   # Posion ratio
  Sf_ = 0.5*450*1e6   # Resistência à fadiga não corrigida (Pa) [17]

class Steel_1045:   # [17, p1017]
  Sy = 531*1e6   # Limite de escoamento do material (Pa)
  Su = 627*1e6    # Limite de Resistência à tração (Pa)
  Sf_ = 0.5*627*1e6   # Resistência à fadiga não corrigida (Pa) [17]

# Função para escolha do usuario:

def get_user_choice(prompt, options):
  while True:
    print(prompt)
    for i, option in enumerate(options):
      print(f"[{i + 1}] {option}")

    try:
      choice = int(input("Digite o número da sua escolha: "))
      if 1 <= choice <= len(options):
          print('\n---\n')
          return options[choice - 1]
      else:
          print("\n*** Escolha inválida. Selecione uma opção válida. ***\n")
    except ValueError:
      print("\n*** Entrada inválida. Insira o número da sua escolha. ***\n")

#


# Função para escolha do material da bomba:

def select_material():
  while True:
    options =  ['Aluminio 6063-O', 'Aluminio 6061-T6', 'Aluminio 7075-T6', 'Aço 1020', 'Aço 1045']
    print("* Selecione o material para a estrutura da bomba:\n")
    for i in range(len(options)):
        print(f"[ {i + 1} ] {options[i]}")
    try:
      material_choice = int(input("\n")) - 1
      if 0 <= material_choice < len(options):
          material = options[material_choice]
          if material == 'Aluminio 6063-O':
              c_material = Al_6063_O
          elif material == 'Aluminio 6061-T6':
              c_material = Al_6061_T6
          elif material == 'Aluminio 7075-T6':
              c_material = Al_7075_T6
          elif material == 'Aço 1020':
              c_material = Steel_1020
          elif material == 'Aço 1045':
              c_material = Steel_1045
          print('-> ', material, '\n\n---\n\n')
          return c_material
      else:
          print("\n*** Escolha inválida. Selecione uma opção válida. ***\n")
    except ValueError:
        print("\n*** Entrada inválida. Insira o número da sua escolha. ***\n")


# Função para escolha do fluido propelente:

def select_fluid():
  while True:
    options = [ 'H2', 'O2', 'N2O', 'H2O', 'CH4', 'Methanol', 'C2H6', 'Ethanol']
    print("* Selecione o fluido propelente:\n")
    for i in range(len(options)):
        print(f"[ {i + 1} ] {options[i]}")

    try:
      fprop_choice = int(input("\n")) - 1
      if 0 <= fprop_choice < len(options):
        fprop = options[fprop_choice]

        if fprop in ['H2', 'H2O', 'CH4', 'Methanol', 'C2H6', 'Ethanol']:
          ForO = 'fuel'
        else:
          ForO = 'oxidizer'

        print('-> ',fprop,'\n\n---\n\n')
        return fprop, ForO
      else:
          print("\n*** Escolha inválida. Selecione uma opção válida. ***\n")
    except ValueError:
        print("\n*** Entrada inválida. Insira o número da sua escolha. ***\n")

#


# Propriedades do fluido:

def fluid_properties(fluid, T):
  rho = PropsSI("Dmass","Q",0, "T", T, fluid)
  pmin = PropsSI("P","Q",0, "T", T, fluid)
  print("* Pressão mínima de armazenamento no tanque",T,"K =",pmin/1e5,"bar\n")
  print("* Densidade líquido",T,"K =",rho,"kg/m³\n")
  return rho, pmin

#


# Função para cálculo de parâmetros fundamentais para a bomba:

def rocketpump_param(Omegas, g, dp, rho, F, Isp, OF):

  ## Cálculo do Aumento de pressão, Altura de carga da bomba (m)
  dH = dp/(rho*g)    # Altura de carga da bomba para o instante inicial considerando apenas a pressão do tanque (m) [14]
  ##

  ## Cálculo da vazão volumétrica de oxidante:
  m_ = F/(Isp*g)   # Vazão mássica de propelente (oxidante + combustível) de projeto (kg/s) [2]
  m_fluid = OF*m_/(OF+1)    # Vazão mássica de oxidante de projeto (kg/s) [2][3]
  Q_fluid = m_fluid/rho   # Vazão volumétrica de oxidante de projeto (m³/s)
  #print("* Altura de carga (m) = ",dH,"\n* Vazão Mássica de Oxidante (kg/s) = ",m_fluid,"\n* Vazão Volumétrica de Oxidante (m³/s) = ",Q_fluid,"\n\n")
  ##

  ## Velocidade angular do rotor em rad/s
  N = (Omegas * ((g * dH) ** (3/4))) / (Q_fluid ** 0.5)
  ##

  ## Passando para rpm
  N_us = 9.5492965964254 * N
  ##

  ## Passando para gpm e ft
  Q_gpm = Q_fluid * 15850.3
  dH_ft = dH * 3.28084
  ##

  ## Velocidade Específica (US units)
  Ns = (N_us * (Q_gpm ** 0.5)) / (dH_ft ** (3/4))
  print("* Velocidade específica (US units) = ",Ns,"\n")
  ##

  ## Velocidade Específica (SI units)
  Ns_SI = (N_us * (Q_fluid ** 0.5)) / (dH ** (3/4))
  ##

  return dH, m_, m_fluid, Q_fluid, N, N_us, Ns, Ns_SI

###


# Função para escolha das constantes do rotor e voluta:

def constants_imp_vol():
  print("\n*** Consulte a velocidade específica em US units ' Ns ' para os parâmetros a seguir: ***\n")
  while True:
    try:
      print("---\n\n\n* Constantes do Rotor (Consulte o gráfico da Fig. 5.2 [13]):\n")
      Km1 = float(input("Constante Km1 = "))
      Km2 = float(input("Constante Km2 = "))
      print('\n')

      break
      return Km1, Km2, K3, D3ratio, alfav

    except ValueError:
      print("\n*** Entrada inválida. ***\n")

  while True:
    try:
      print("---\n\n\n* Constantes da Voluta (Consulte o gráfico da Fig. 7.5 [13]):\n")
      K3 = float(input("Constante K3 = "))
      D3ratio = float(input("Constante ((D3-D2)/D2)*100 = "))
      alfav = float(input("Constante alfav = "))
      print('\n---\n')

      return Km1, Km2, K3, D3ratio, alfav

    except ValueError:
      print("\n*** Entrada inválida. ***\n")

  #while True:

#


# Plot

def plot_dict_as_table(data_dict, title=''):
    df = pd.DataFrame(data_dict)

    table_fig = go.Figure(data=[go.Table(
        header=dict(values=df.columns, height=40, align=['center'],
                    line_color='darkslategray', fill_color='royalblue',
                    font=dict(color='white', size=16)),
        cells=dict(values=[df[col] for col in df.columns],
                   line_color='darkslategray', height=30,
                   fill_color=['lightskyblue', 'lightcyan'],
                   font=dict(color='darkslategray', size=14)))
    ], layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    ))

    table_fig.update_layout(title=title, titlefont=dict(color='royalblue', size=28), height=500)

    return table_fig

#

"""

# ENTRADA

Tabela 10-2: Obtenção dos dados de entrada da bomba (velocidade específica e eficiência) [2, p381]

"""

########################

""" ENTRADA DE DADOS """

########################


# Dados Iniciais para o Motor:

F = 1500   # Empuxo de projeto (N)
OF = 7.601    # Razão Oxidante/Combustível de projeto
Isp = 202.92    # Impulso Específico de projeto (s)
dp = 10e5    # Incremento de pressão da bomba (Pa)
T = 298    # Temperatura de armazenamento do fluido (K)
g = 9.806    # Aceleração da gravidade (m/s²)

#


# Dados Iniciais para a Bomba:

Omegas = 0.25   # Velocidade Específica (dimensionless) [2, Table 10-2]
np = 0.7    # Eficiência da bomba [2, Table 10-2]

#


fluid, ForO = select_fluid()
material = select_material()
rho, p = fluid_properties(fluid, T)
dH, m_, m_fluid, Q_fluid, N, N_us, Ns, Ns_SI = rocketpump_param(Omegas, g, dp, rho, F, Isp, OF)   # Calcular velocidade específica e velocidade do rotor
Kcm1, Kcm2, K3, D3ratio, alfav = constants_imp_vol()
#Kcm1 = 0.16   # Constante de capacidade de entrada em função de Ns [13]
#Kcm2 = 0.11   # Constante de capacidade de saída em função de Ns [13]
#K3 = 0.49   # Constante de velocidade da voluta [13][4][12]
#D3ratio = 6.5   # Igual a ((D3-D2)/D2)*100: Razão para obtenção do diâmetro de base da voluta [13][4][12]
#alfav = 5.5   # Ângulo da voluta (°) [13][4][12]

"""

Fig. 5.2: Obtenção da constante de capacidade (Kcm) para design do rotor [13, p79]

Fig. 7.5: Gráfico para obtenção de constantes para o design da voluta [13, p113]

# CÓDIGO

## Parâmetros Gerais

"""

##  * Para os cálculos de projeto, será considerada na entrada a pressão inicial do tanque, "p". *
##  * A bomba terá o objetivo de elevar essa "p"em "dp" Pa. Esse será o Best Efficiency Point (B.E.P) [18] *
##  * Será dimensionado uma bomba centrífuga de um estágio para esta aplicação. *

#

pump_data = {"Parâmetro": ['Altura de carga (m)', 'Vazão Volumétrica de Oxidante (m³/s)', 'Velocidade do rotor (rpm)', 'Torque do motor (N.m)', 'Potência mecânica do eixo (W)'],
             "Valor": [round(dH,2), round(Q_fluid*1000, 3), round(N_us)]}

print("\n* Altura de carga (m) = ",dH,"\n* Vazão Mássica de Oxidante (kg/s) = ",m_fluid,"\n* Vazão Volumétrica de Oxidante (m³/s) = ",Q_fluid)
print('\n* N (rad/s) = ',N,'\n* N (rpm) = ',N_us,'\n* Velocidade específica (US) = ', Ns)

#

"""## Dimensionamento do Rotor (Impeller)"""

# Obtenção do diâmetro de saída do rotor:

Ds = (3.72/(Omegas))**(1/1.1429)    # Diagrama de Cordier: Pela Eq da Ref. [5], Pelo gráfico da Ref. [4]

D2 = (Ds*(Q_fluid**(0.5)))/((g*dH)**(1/4))   # Diâmetro de saída (m) [4][5]

#


# Cálculo dos rendimentos e obtenção da potência de eixo:

##  Potência e eficiência total (* OBS * Verificar se essa eficiência não é a mesma que setei como 0.65 abaixo):

Ph = dp*Q_fluid    # Potência hidráulica da bomba (W) [4]

Pe = Ph/np    # Potência no eixo (W) [4]

Tor = Pe/N    # Torque do motor (N.m)

#np = Ph/Pe    # Eficiência total da bomba [4][14], localiza-se entre 50-80% para radial impellers [2, Table 10-2]

print('Eficiência total da bomba = ',np)
print('Torque do motor (N.m) = ',Tor)
print('Potência mecânica do eixo (W) = ',Pe)

pump_data["Valor"].append(round(Tor,2))
pump_data["Valor"].append(round(Pe,2))

###


u2 = N*D2/2   # Velocidade de saída (m/s) [2][5, p7]

Cm1 = Kcm1*((2*g*dH)**0.5)   # Velocidade vertical de entrada (Vertical velocity) (m/s), assumindo alfa = 90° [12][14 (*Cm1=Vn1)][15][13]

Cm2 = Kcm2*((2*g*dH)**0.5)   # Velocidade vertical de saída (Vertical velocity) (m/s) [15][13]

Cu2 = (g*dH)/(np*u2)   # Componente tangencial da velocidade absoluta (m/2) [12] (*Cu1 = 0, alfa = 90°)

psi = u2/((2*g*dH)**0.5)   # Head Coefficient, geralmente entre 0,9 e 1,10 para diferentes designs de bombas [2]

print('\nHead Coefficient (Ψ) = ',psi)

print("Cm1, Cm2, Cu2, u2:",Cm1, Cm2, Cu2, u2)

"""### Dimensionamento do Eixo"""

# Diâmetro mínimo de eixo segundo Gulich [18] e Lobanoff *OBS.: A Resistência ao escoamento sob cisalhamento Sys = 0,577*Sy [17]:

Dshaft_min =  ((16*Pe)/(math.pi*N*material.Sy*0.577))**(1/3)     # Diâmetro mínimo do eixo (m), considerando Ma = 0 inicialmente e coeficiente de segurança de 2,5 [17]

print("\nDiâmetro mínimo de eixo (Gulich e Lobanoff) = ", Dshaft_min*1000, "mm")


Dshaft = 2*Dshaft_min   # Diâmetro mínimo de eixo com coeficiente de segurança = 2
print("\nDiâmetro mínimo de eixo com coeficiente de segurança = ", Dshaft*1000, "mm")


if Dshaft < Dshaft_min:
  Dshaft = Dshaft_min


Dshaft = round(Dshaft*1000,0)/1000    # Diâmetro de eixo arredondado (m)

#

print("\nDiâmetro de eixo cosiderado (arredondado) = ", Dshaft*1000, "mm")

"""### Dimensões do Rotor"""

impeller_data = {"Parâmetro": ['D eixo (mm)', 'D hub (mm)', 'Diâmetro de entrada (mm)', 'Diâmetro de saída (mm)', 'Beta 1 (°)', 'Beta 2 (°)', 'Número de pás', 'Espessura da pá na entrada (mm)','Espessura da pá na saída (mm)', 'Largura entrada (mm)', 'Largura saída (mm)', 'Raio de curvatura da pá (método single-arc) (mm)'],
                 "Valor": []}


# Cálculo do diâmetro de hub [17]:

## Dhub considerando altura de chaveta padrão [17]

if round(Dshaft) > 8e-3:

  if round(Dshaft) <= 10e-3:
    H_chaveta = 3e-3

  elif round(Dshaft) <= 12e-3:
    H_chaveta = 4e-3

  elif round(Dshaft) <= 17e-3:
    H_chaveta = 5e-3    # Altura Padrão Chaveta Paralela para 17>=Dshaft>12 mm [17 (pag. 571)]

  elif round(Dshaft) <= 22e-3:
    H_chaveta = 6e-3

  elif round(Dshaft) <= 30e-3:
    H_chaveta = 7e-3

  elif round(Dshaft) <= 44e-3:
    H_chaveta = 8e-3

  elif round(Dshaft) <= 50e-3:
    H_chaveta = 9e-3

  elif round(Dshaft) <= 58e-3:
    H_chaveta = 10e-3

  elif round(Dshaft) <= 65e-3:
    H_chaveta = 11e-3

  elif round(Dshaft) <= 75e-3:
    H_chaveta = 12e-3

  elif round(Dshaft) > 75e-3:
    H_chaveta = 14e-3

else:
  print("\n* Consultar tabela da referência 17 para o diâmetro de eixo =",Dshaft*1000,"mm *\n")
  H_chaveta = 3e-3

Dhub = Dshaft+(H_chaveta*2)    # Diâmetro do núcleo de fixação do eixo (hub), núcleo de fixação do rotor ao eixo - considerando [(H_chaveta/2) x 2] x 2  [17 (pag. 571)]

###


D1 = 2*(((Q_fluid/(math.pi*Cm1))+((Dhub/2)**2))**0.5)   # Diâmetro de entrada [12][14 (*Cm1=Vn1)]

u1 = N*D1/2   # Velocidade de entrada (m/s)


print("Dshaft (mm) = ",Dshaft*1000,"\nDhub (mm) = ",Dhub*1000, "\nD1 (mm) = ",D1*1000,"\nD2 (mm) = ",D2*1000)

impeller_data["Valor"].append(round(Dshaft*1000,2))
impeller_data["Valor"].append(round(Dhub*1000,2))
impeller_data["Valor"].append(round(D1*1000,2))
impeller_data["Valor"].append(round(D2*1000,2))

# Configuração das pás do rotor - Método do arco da circunferência (single-arc), Hen (2012) [8][9][4]

## Ângulos de entrada e de saída, assumindo que há zero "swirl" na entrada e alfa = 90° [referências]:

beta1 = round(math.degrees(math.atan(Cm1/(u1))))   # Ângulo de entrada em ° considerando o fator de estrangulamento [12][11][13]

beta2 = round(math.degrees(math.atan(Cm2/(u2-Cu2))))   # Ângulo de saída em ° [12]

##


Z = round(6.5*((D2+D1)/(D2-D1))*math.sin(math.radians((beta1+beta2)/2)))   # Número de pás, para Kn = 6.5 (rotores fundidos) [9][4]

t1 = (math.pi*D1)/Z   # Passo de entrada para as pás (m) [9]

t2 = (math.pi*D2)/Z   # Passo de saída para as pás (m) [9]


## Largura de entrada e de saída, de acordo com a lei da conservação de massa (coeficiente de constrição = 1)

b1 = Q_fluid/(math.pi*D1*Cm1)   # Largura de entrada (m) [15]

b2 = Q_fluid/(math.pi*D2*Cm2)  # Largura de saída (m) [4][12][14]

##


## Espessura e raio de curvatura das pás

e = ((D2*1000*b2*1000)**(1/3))*0.3    # Fixação da espessura da pá medida segundo uma normal para bombas com rotor fundido (mm) [9]

et1 = e/(math.sin(math.radians(beta1)))   # Espessura tangencial da pás na entrada (mm) [9]

et2 = e/(math.sin(math.radians(beta2)))   # Espessura tangencial das pás na saída (mm) [9]

Rc = (((D2/2)**2)-((D1/2)**2))/(2*(((D2/2)*math.cos(math.radians(beta2)))-((D1/2)*math.cos(math.radians(beta1)))))    # Raio de curvatura da pá (método single-arc) (m) [9][18]

####


print("Passo de entrada para as pás (mm) = ",t1*1000, "\nPasso de saída para as pás (mm) = ",t2*1000,"\nEspessura da pá na entrada (mm) = ",et1,"\nEspessura da pá na saída (mm) = ",et2,"\nRaio de curvatura da pá (método single-arc) (mm) = ",Rc*1000,"\nBeta 1 (°) = ",beta1,"\nBeta 2 (°) = ",beta2,"\nLargura entrada (mm) = ",b1*1000,"\nLargura saída (mm) = ",b2*1000,"\nNúmero de pás = ",Z)


impeller_data["Valor"].append(round(beta1))
impeller_data["Valor"].append(round(beta2))
impeller_data["Valor"].append(round(Z))
impeller_data["Valor"].append(round(t1*1000,2))
impeller_data["Valor"].append(round(t2*1000,2))
impeller_data["Valor"].append(round(b1*1000,2))
impeller_data["Valor"].append(round(b2*1000,2))
impeller_data["Valor"].append(round(Rc*1000,2))

"""## Dimensionamento da Voluta"""

# Dimensões principais da voluta:

D3 = (D3ratio*D2/100)+D2    # Diâmetro da circunferência de base do Volute (m) [13][4][12]

C3 = K3*((2*g*dH)**0.5)   # Velocidade média no Volute (m/s) [13][4][12]

b3 = 2*b2    # Largura do volute (m), considerando que é uma bomba pequena dimensão [13][12]

Av = Q_fluid/C3    # Área de seção da garganta [4][13]

print("\nDiâmetro da circunferência de base do Volute (mm) =",D3*1000)
print("\nLargura do Volute (mm) =",b3*1000,"\n\n")

#


# Áreas de seção da voluta em intervalos de 45° (Área de seção da garganta aos 360°) [4][15] - Espiral de Arquimedes

Ax = []   # Área na seção x (m²) [4][15]
Rvx = []    # Raio da seção x do volute (m), considerando a seção circular [4][15][12]
tetav = 0   # Ángulo a partir da garganta correspondente à área de seção do volute (°).  [4][15]
volute_data = {'Seção': []}


while tetav <= 360:

  volute_data['Seção'].append(str(tetav)+' °')
  Ax.append(tetav/360*Av)   # Área de seção x, calculada em relação ao tetav [4][15]
  print("Ângulo de ",tetav," °: ",Ax[-1]," m²")

  Rvx.append(((Ax[-1])/math.pi)**0.5)
  print("Raio de seção: ",Rvx[-1]," m\n")

  tetav = tetav + 45

#

volute_data['A (mm²)'] = [round(numero*1e6, 2) for numero in Ax]    # Área seção (m²)
volute_data['R (mm)'] = [round(numero*1e3, 2) for numero in Rvx]    # Raio seção (m)

"""### Difusor Cônico"""

dif_data = {'Parâmetro': ['Comprimento do difusor (mm)', 'Semi-ângulo do difusor (°)'],
            'Valor': []}

# Na prática, difusores são geralmente projetados para um comprimento específico com AR < 3 [18]

ARdif = 2    # Desse modo, será assumido AR = 2 como parâmetro inicial

L_R1 = (ARdif-1.05)/0.184    # Razão comprimento/raio de entrada, por meio da mesma figura na qual cp é obtido

cpdif = 0.36*(L_R1)**0.26    # Para um dado AR, cp ótimo pode ser estimado por essa equação ou figura abaixo [18][12]

Ldif =  Rvx[-1]*L_R1   # Comprimento do difusor, considerando raio de entrada igual ao raio da última seção do volute (m)

print("\ncp ideal =",cpdif,"\nL/R1 =",L_R1)
print("\nComprimento do difusor =",Ldif*1000, "mm")

tetadif = round(math.degrees(math.atan((1/L_R1)*((ARdif**0.5)-1))))   # Semi-ângulo do difusor, deve estar abaixo da linha a-a (°) [18]
print("Semi-ângulo do difusor =",tetadif, "°")

dif_data['Valor'].append(round(Ldif*1000,2))
dif_data['Valor'].append(round(tetadif))

pumplt = plot_dict_as_table(pump_data, title='PARÂMETROS GERAIS DA BOMBA')
impplt = plot_dict_as_table(impeller_data, title='ROTOR: DIMENSÕES')
volplt = plot_dict_as_table(volute_data, title='VOLUTA: DIMENSÕES POR SEÇÃO')
difplt = plot_dict_as_table(dif_data, title='DIFUSOR: DIMENSÕES')

"""# SAÍDA"""

######################################

""" DIMENSÕES DA BOMBA CENTRÍFUGA """

######################################

pumplt.show()
impplt.show()
volplt.show()
difplt.show()

"""

# CAVITAÇÃO (NÚMERO DE THOMA)

"""

########################

""" ENTRADA DE DADOS """

########################

pin = 70e5    # Pressão de armazenamento do fluido no tanque (Pa)

########################

"""## Código"""

# Cálculo Número de Thoma (deve estar abaixo do número crítico para prevenir cavitação):

Hsa = pin/(rho*g)    # Altura de carga de sucção da bomba disponível para pressão de armazenamento - NPSH: available net positve suction head (m) [2, p410][14, p696]

Hv = PropsSI("P","Q",0, "T", T, "N2O")/(rho*g)    # Altura de carga para pressão de vapor 298 K(m)

thoma = (Hsa - Hv)/dH   # Coeficiente de Thoma sem considerar fricção (298 K)

#

"""## Saída"""

print("\nCoeficiente de Thoma para N2O na temperatura ambiente =",round(thoma,2),'\n\n*** Número de Thoma deve estar abaixo do número crítico para prevenir cavitação: Consultar Fig. 9-7 de Round [15]. ***')

"""
Fig. 9-7: Número de Thoma crítico em função da velocidade específica [15]


# REFERÊNCIAS

1.   Coolprop. Disponível em: <http://www.coolprop.org/>
2.   SUTTON, G. P.; BIBLARZ, O. Rocket propulsion elements. Wiley. [S.l.]: Wiley, 2017.
3.   RACHOV, P. A. P.; TACCA, H.; LENTINI, D. Electric Feed Systems for Liquid-Propellant
Rockets. Journal of Propulsion and Power, v. 29, n. 5, p. 1171 – 1180, 2013. Disponível
em: https://doi:org/10:2514/1:B34714.
4.   KIM, H. I. et al. Development of Ultra-Low Specific Speed Centrifugal Pumps Design Method for Small Liquid Rocket Engines. Aerospace, v. 9, n. 9, 2022. ISSN 2226-4310. Disponível em: https://www:mdpi:com/2226-4310/9/9/477.
5.   LEE, J. et al. Performance Analysis and Mass Estimation of a Small-Sized Liquid Rocket Engine with Electric-Pump Cycle. International Journal of Aeronautical and Space Sciences, v. 22, 09 2020.
6.   LOBANOFF, V. S.; ROSS, R. R. 16 - Shaft Design and Axial Thrust. In: LOBANOFF, V. S.; ROSS, R. R. (Ed.). Centrifugal Pumps (Second Edition). Boston: Gulf Professional Publishing, 1992. p. 333 – 353. ISBN 978-0-08-050085-0.
7.   OVERVIEW OF INDUSTRIAL AND ROCKET TURBOPUMP INDUCER DESIGN, Japikse
8.   DIMENSIONAMENTO DE UM ROTOR PARA UMA BOMBA CENTRÍFUGA, LEONARDO NUNES SCHWARZ
9.   SILVA, J. B. C. Pré-Projeto de Rotores de Máquinas de Fluxo Geradoras Radiais. Universidade Estadual Paulista, Ilha Solteira, 2000.
10.   TEDESCHI, P. Proyecto de Máquinas, Editorial Universitária, Buenos Aires, 1969
11.    LIQUID ROCKET ENGINE CENTRIFUGAL FLOW TURBOPUMP, NASA
12.   SAKSERUD, N. M. K. Centrifugal Pump for a Rocket Engine. In: . [S.l.: s.n.], 2019.
13.   STEPANOFF, A. J. Centrifugal and Axial Flow Pumps. 2. ed. New York: John Wiley & Sons, 1957.
14.   FOX, R. W.; PRITCHARD, P. J.; MCDONALD, A. T. Introdução à Mecânica dos fluidos. 8. ed. Rio de Janeiro: LTC, 2014.
15.   ROUND, G. F. Incompressible Flow Turbomachines: Design, Selection, Applications, and
Theory. 1. ed. [S.l.]: Elsevier Science, 2004.
16.   Resistência dos Materiais 7ª Edição, Hibbeler
17.   NORTON, R. L. Projeto de Máquinas: Uma Abordagem Integrada. 4. ed. [S.l.]: Bookman, 2013. ISBN 8582600224, 978-8582600221.
18. GÜLICH, J. F. Centrifugal Pumps. 2. ed. [S.l.]: Springer Berlin, Heidelberg, 2010.
"""