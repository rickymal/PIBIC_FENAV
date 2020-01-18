#!/usr/bin/env python
# coding: utf-8

# # Análise, validação, calibração e previsão de dados
# ## bibliotecas utilizadas:
# * Pandas
# * NumPy
# * sklearn
# * pltoly
# * matplotlib
# 
# ## Objetivo
# * O objetivo deste script é realizar uma análise pelo Teste de Man kendall e pela correlação de Pearson entre as variáveis inseridas hidrológicas e também as variáveis Enos
# 
# ## Resumo do trabalho
# A interação oceânica-atmosférica interfere de sobremaneira forma no regime de precipitação das bacias
# amazônicas. Entender a relação entre causa e efeitos auxilia para que previsões dos dados hidrológicos
# possam ser xrealizados em função de variáveis aqui denominadas de “dados extremos”, que são as variáveis
# influenciadoras dos fenômenos Enos como temperatura, pressão e radiação. O presente trabalho busca
# correlacionar os dados extremos com os dados de lâmina d’água, que são utilizado para a modelagem de
# condições de navegabilidade. O modelo utilizado para a estimação dos dados hidrológicos em função dos
# fenômenos Enos são modelos Multi Layer Perceptron (MLP), com algoritmo de backpropagation. Foram
# desenvolvidos dois modelos MLP desse artigo. O primeiro modelo realiza uma relação dos dados de nível
# d’agua com apenas dados de temperatura da região El Niño 1 + 2. Está variável é definida como entrada de
# dados para uma rede cujas entradas recebem diferentes defasagens desta mesma variável, fazendo com
# que a rede possa analisar diferentes períodos predecessores, cuja relações lineares, obtidas por meio da
# correlação de Pearson, são altas. O segundo modelo MLP criado relaciona outros índices relacionados aos
# fenômenos Enos, onde se procura a melhor defasagem que consiga descrever o comportamento de nível
# d’água com base nos dados extremos. Uma análise utilizando o Método de Man Kendall é utilizada para se
# realizar uma avaliação de relação.
# Palavras-chave: Redes neurais, previsão, Correlação de Pearson, Mann Kendall.
# 
# > *Motivação é a arte de fazer as pessoas fazerem o que você quer que elas façam porque elas o querem fazer.*
# 
# ## Outras informações
# * As variaveis do tipo .dframe armazenam um dataframe
# * As variaveis do tipo .sav armazenam informações genéricas
# * As variaveis do tipo .sln armazenam uma hierarquia contento informações relevantes acerca de uma otimização
# provavelmente definida por um objeto do tipo optimizer que será criado

# In[154]:


# importação de bibliotecas
# importando todas as bibliotecas necessárias para o processo
#get_ipython().run_line_magic('matplotlib', 'inline')

#Configurando o diretório de execução para o MAC - DEDE



import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from copy import deepcopy
import os
import mk_test as mks
from pandas import ExcelWriter
from copy import deepcopy
idx = pd.IndexSlice

#retira o limite de leitura das colunas
#from IPython.display import display
#pd.options.display.max_rows = None

#eliminta os avisos de depreciação e de falha na construção de redes neurais
#import warnings
#warnings.filterwarnings('ignore')


# ## Coleta de dados

# In[155]:

print('Programa iniciado com sucesso!')
print("Algoritmo Prévision")
print("Desenvolvido por: Henrique Mauler")

dados = pd.read_excel('COTA_TAPAJOS.xlsx',skiprows=1,index_col=0)
dados = dados.iloc[:,:7].copy()
dadosMensais = dados.resample('D').mean().resample('MS').mean() # Conversão dos dados para valores de média mensal
dadosMensais.index.Names = 'Dados médios mensais' # Definindo o nome da série temporal
dadosMensais = dadosMensais.interpolate() # Realiza interpolações dos dados não preenchimento (linearmente apenas, pois os intervalos não preenchidos são mínimos)


glb_defasagem = range(2)
glb_diff = [False,True]


# In[156]:


dados = pd.read_excel('VAZAO_TAPAJOS.xlsx', skiprows=1, index_col=0)
dados.head().set_index('DIAS')

dados = pd.read_excel('VAZAO_TAPAJOS.xlsx',skiprows=1,index_col=0).set_index('DIAS')
dados = dados.iloc[:,:4].copy()
dados = dados.resample('D').mean().resample('MS').mean()
#dadosMensais.interpolate(method='polynomial',order=2).plot(figsize=(18,6))


# In[157]:


temp = pd.read_excel('PRECIPITACAO_TAPAJOS.xlsx',skiprows=1,index_col=0)
temp = temp.resample('MS').mean()

#dados acoplados em uma tupla que contem, na sequencia, dados de : cota, cazão, e chuva 
dadosMensais = (dadosMensais,dados,temp)


# In[158]:


def read(caminho):
    return pd.read_csv(caminho,sep='\s+',engine='python',header=1,comment='#',
                    names=['Ano','Jan','Fev','Mar','Abr','Mai','Jun',
                           'Jul','Ago','Set','Out','Nov','Dez']).set_index('Ano').replace(-99.99,np.nan).replace('-99.99',np.nan).replace(-999.000,np.nan)


# In[159]:


#Coletando os dados referentes do NOAA
MEI = pd.read_csv('NOAA2/MEI.txt',sep='\s+',engine='python',header=1,comment='#')
MEI = MEI.set_index('Ano')



NINA3PLUS4 = read('NOAA2/nina34.txt')
NINA4 = read('NOAA2/NINO 4.txt')
ELNINO1PLUS2 = read('NOAA2/EL NINO 1+2.txt')
INDBIVENSO = read('NOAA2/INDICE BIVARIAVEL ENSO.txt')
NOA3SST = read('NOAA2/NOA 3 - SST.txt')
PDO = read('NOAA2/pdo.txt')
SOI = read('NOAA2/SOI.txt')
NOI = read('NOAA2/NOI')

NOAA = {
    'El nino 3+4' : NINA3PLUS4,
    'MEI' : MEI,
    'El nino 1+2' : ELNINO1PLUS2,
    'Indice bivariavel Eno' : INDBIVENSO,
    'Noa 3 sst' : NOA3SST,
    'Oscilação decadal do pacífico' : PDO,
    'Indice de oscilação sul' : SOI,
    'Indice Oceânico Nino' : NOI,
}


# ## Conversões e visualizações de dados

# In[160]:


# In[163]:


#Como ainda não sei onde está os dados preenchidos:
#dadosMensais.interpolate(method='polynomial',order=2).plot(figsize=(18,6))
#yy = [dadosMensais[0],dadosMensais[1],dadosMensais[2]]
yy = list(dadosMensais)
#yy[0].plot(figsize=(18,6))
#yy[1].plot(figsize=(18,6))
#yy[2].plot(figsize=(18,6))


# In[164]:


#Como ainda não sei onde está os dados preenchidos:
#dadosMensais.interpolate(method='polynomial',order=2).plot(figsize=(18,6))
y = [dadosMensais[0].interpolate(method='polynomial',order=2),
     dadosMensais[1].interpolate(method='polynomial',order=2),dadosMensais[2]]


# In[165]:


#y[0].plot(figsize=(18,6))
#y[1].plot(figsize=(18,6))
#y[2].plot(figsize=(18,6))


# In[166]:


def Log(mensagem, esp):
    espac = esp * "/t"
    print(espac + mensagem)


#função para converter o formato de dados presentes no site do NOAA
#depois da atualização com os dados presentes no hidroweb 
def NOAA2HIDROWEB(data : pd.DataFrame):
    toNumData  = {
        'Jan' : '01',
        'Fev' : '02',
        'Mar' : '03',
        'Abr' : '04',
        'Mai' : '05',
        'Jun' : '06',
        'Jul' : '07',
        'Ago' : '08',
        'Set' : '09',
        'Out' : '10',
        'Nov' : '11',
        'Dez' : '12',
    }
    dataList = list()
    for x in data.index:
        for y in data.columns:
            dataList.append('01/{}/{}'.format(toNumData[y],x))
    
    temp = pd.DataFrame({
        'Data' : np.array(dataList),
        'Value' : np.array(data).ravel()
    })
    temp['Data'] = pd.to_datetime(temp['Data'],format='%d/%m/%Y')
    temp = temp.set_index('Data').copy()
    return temp


# In[167]:


NOAA_Serie = pd.DataFrame()
debug = list()
for chave,valor in NOAA.items():
    debug.append(NOAA2HIDROWEB(valor))
    NOAA_Serie[chave] = NOAA2HIDROWEB(valor)['Value']
y.append(NOAA_Serie)
y[-1].columns = [x.upper() for x in y[-1].columns]






# In[169]:

#exportandos os dados para o excel porque eu desisti de trabalhar aqui
writer = ExcelWriter('Excel.xlsx')

# In[170]:

#deixando todas as variáveis com letra maiúscula para facilitar o filtro futuramente
y[0].columns = [x.upper() for x in y[0].columns]
y[1].columns = [x.upper() for x in y[1].columns]
y[2].columns = [x.upper() for x in y[2].columns]
y[-1].columns = [x.upper() for x in y[-1].columns]


# In[171]:

#Este bloco cria uma dataframe multindexado com todas as variáveis presentes coletadas, com suas chaves indicando 
#o tipo de variável que está sendo coletada
result = pd.concat([x.T for x in y],keys=['COTA','VAZÃO','CHUVA','ENO']).T
result.to_excel(writer,'Dados coletados')
result_multindex = result.copy()
result_multindex = result.copy().drop(columns = [['ENO','INDICE OCEÂNICO NINO'],['ENO','INDICE BIVARIAVEL ENO']], axis = 0) #Esses dados enos atrapalham o treinamento pois diminuiem o período da série temporal

#result_multindex = result_multindex.iloc[:,:5]


# In[172]:


#Armazena a variável que foi criada em uma tabela serializada
with open('resultMultindex.dframe','wb') as file:
    pickle.dump(result_multindex,file)


# In[173]:


result1 = pd.concat([x.T for x in y],keys=['Cota1','Vazão1','Eno1']).T.dropna()
result2 = pd.concat([x.T for x in y],keys=['Cota2','Vazão2','Eno2']).T.dropna()
result3 = pd.concat([x.T for x in y],keys=['Cota3','Vazão3','Eno3']).T.dropna()
result_extent = pd.concat([result1,result2,result3],keys=['Indice1','Indice2','Indice3']).T


# In[174]:


result = result.dropna()


# In[176]:


result = pd.concat([x.T for x in y],keys=['Cota','Vazão','Chuva','Eno']).T



#Esta linha considera que a atualização 2020 não foi realizada
result = result.copy().drop(columns = [['Eno','INDICE OCEÂNICO NINO'],['Eno','INDICE BIVARIAVEL ENO']], axis = 0) #Esses dados enos atrapalham o treinamento pois diminuiem o período da série temporal
# ## Inicio do teste de Mann Kendall

# In[177]:


#Realizando o teste de Man-kendall para cota e vazão

tabela_mks = {}


# In[178]:


def MannKendall(out):
    if isinstance(out,pd.Series):
        out = pd.DataFrame(out)
    temp = list()
    temp_name = list()
    for value in out.dropna().items():
        temp.append(mks.mk_test(value[1].values))
        temp_name.append(value[1].name)
    frame_dados = pd.DataFrame(temp,index=temp_name,columns=['Tendencia','h','p','z'])
    return frame_dados



# In[180]:


a1 = MannKendall(y[0].iloc[:-100])
a2 = MannKendall(y[1].iloc[:-100])
a3 = MannKendall(y[2].iloc[:-100])
a4 = MannKendall(y[3].iloc[:-100])
tabela_mks = pd.concat([a1,a2,a3,a4],keys=['Vazão','Cota','Chuva','NOAA'])


# In[181]:

# from itertools import combinations,permutations
from itertools import permutations,combinations
import scipy.stats as psy #para fazer correlação de pearson

log = list()
def relacionarDados(args,multiIndexKey = ' de ', relashionshipKey = '<>',
                    defasagem = 0,diferential = False,
                   withDiferentialIndex=True,keys = None,):
     #variável que eu criei por alguma razão que eu não me lembro.
    log = True
    convertion = True
    saida = {}
    logList = list()
    meuIndice = list()
    
    """
    O método em questão realizará um loop iterando entre uma permutação de dois em dois de items
    o métodos items de um dataframe nos retorna uma tupla contendo (nome do indice, série)
    logo, as variáveis 'chave' armazenam o nome da série (que contem o nome da estação)
    e as variáveis 'valor' contém a série pŕopriamente dita.
    """
    for conteudo in permutations(args.items(),2):  #items() retorna dois valores: o id da coluna e a serie correspondente (id da coluna será uma tupla caso seja multindexado)
        intern_logList = list()
        chave1 = conteudo[0][0]
        valor1 = conteudo[0][1]
        chave2 = conteudo[1][0]
        valor2 = conteudo[1][1]


        #Prestar atenção que, ao diferençar um valor, já teremos, obrigatóriamente, uma defasagem (para frente)!
        #Realizar as correções..
        if diferential:

            valor2 = valor2.diff()
            #Considerando que a variavel já causa um deslocamento.. ACREDITO QUE MEU RACIOCÍNIO ESTEJA CERTO..
            if defasagem > 1:
                defasagem = defasagem - 1
            else:
                print('Parâmetro de defasagem zero detectado, será considerao uma defasgem igual a um.')
        valor2 = valor2.shift(defasagem).dropna()


        #defasar
        #valor1 = pd.Series(valor1.values[defasagem:],index=valor1.index[:-defasagem],) #defaso em 'defasagem' mês(es) 

        intern_logList.append(chave1)
        intern_logList.append(chave2)
        if log == True:
            #logList.append((chave1,chave2,valor1,valor2))
            pass
        if isinstance(chave1,tuple): #o conteudo da chave é uma tupla? então temos uma série multindexada..
            chave1 = chave1[0] + multiIndexKey + chave1[1]
        if isinstance(chave2,tuple):
            chave2 = chave2[0] + multiIndexKey + chave2[1]
        relação = ' {} {} {}'.format(chave1,relashionshipKey,chave2)
        #saida[relação] = psy.pearsonr(valor1.dropna().values,valor2.dropna().values)
        valorMinimo = pd.DataFrame([valor1,valor2]).T.dropna()
        #Aplicação da correlação de Pearson nas duas variaveis atuais 'n' da permutação

        print(len(valorMinimo))

        if len(valorMinimo) == 0:
            print('valor minimo encontrado.')
            continue

        saida[relação] = psy.pearsonr(valorMinimo.iloc[:,0].values,valorMinimo.iloc[:,1].values)
        intern_logList.append(saida[relação])
        logList.append(tuple(intern_logList))

    saida['legenda'] = ['2-tailed','p-value'] 


    #Organiza os dados obtidos na correlação de Pearson para coloca-los em uma tabela multindexada
    #Caso a função entre neste método, o valor retornado é um pd.dataframe da uma relação entre estações.. (unitário)
    if convertion:
        extern_temp = list()
        result = list()
        noMultiIndex = True
        for v in logList:
            #preparando os índices
            temp = list()

            if not noMultiIndex:
                print('não deve entrar aqui')
                temp.append(v[0])
                temp.append(v[1])
            else:
                temp.append(v[0][0].lower() + ' de ' + v[0][1].lower())
                temp.append(v[1][0].lower() + ' de ' + v[1][1].lower())
            extern_temp.append(tuple(temp))
            #preparados os valores
            result.append(v[2])
        extern_temp = tuple(extern_temp)
        Pearson_Table = pd.DataFrame([x for x in result],index=pd.MultiIndex.from_tuples(extern_temp),columns=['Grau de correlação','valor p'])
        #pode ser escrito diretamente dessa forma?

        # É preferível que se entre nesta função
        if withDiferentialIndex:
            
            print('definindo multindexação')
            mensagem = 'com' if diferential == True else 'sem'
            mes = 'mês' if defasagem == 1 else 'meses'
            tt = [('{} {} {} diferenciação'.format(defasagem,mes,mensagem),x) for x in Pearson_Table.columns]
            Pearson_Table.columns = pd.MultiIndex.from_tuples(tt)
            return Pearson_Table

        return pd.DataFrame([x for x in result],index=extern_temp,columns=['Grau de correlação','valor p'])

    return (pd.DataFrame(saida).set_index('legenda'),logList)

def getValuesFromCorrelations(defasagens = [0], diferenciação = [False],savefile = 'ls_PearsonCorrelation.dframe',
                             forceCreateNewFile = False):
    #O objetivo deste bloco é apenas pegar todas as correlações e joga-las no Excel 


    #Variaveis importantíssimas!!!
    #O multiprocessamento, apesar de mais fácil, não funciona muito bem no jupyter-notebook por isso é descartado.
    Multiprocessamento = False
    #global minhasCorrelações
    minhasCorrelações = list()
    loadLastPearsonTable = False
    #global gambiarra
    #global diferen
    #diferen = diferenciação
    #gambiarra = len(defasagens) #variavel usada na função 'criaTabela' para obter os valores diferenciais 


    if os.path.isfile(savefile) and forceCreateNewFile:
    #    minhasCorrelações = loaded_model = pickle.load(open(filename, 'rb'))
        with open(savefile,'rb') as file:
            minhasCorrelações = pickle.load(file)
    else:
        print('não existe ')
        #Estranho eu não considerar a diferenciação neste caso.. mas analisar!!!!!!!!!!!!!!!
        for ind,dif in enumerate(diferenciação):
            minhasCorrelações.append([])
            for value in defasagens:
                print('defasando {} meses'.format(value))
                minhasCorrelações[ind].append(relacionarDados(result_multindex,keys=None,
                                                         defasagem=value,diferential = dif))
        import pickle
        with open(savefile,'wb') as fleName:
            pickle.dump(minhasCorrelações,fleName)
        #pickle.dump(minhasCorrelações,open('ls_PearsonCorrelation.dframe','wb'))



            #Código estranho? estou apenas defasando em dois sem sem diferenciar?! !!!!!!!!!!!!!!!!!
    #        writer = ExcelWriter('Pearson.xlsx')
    #        for i,value in enumerate(minhasCorrelações):
    #            value.to_excel(writer,'Permutação::{}'.format(i))
    #        writer.save()

    #writer = ExcelWriter('Pearson.xlsx')
    #for i,value in enumerate(minhasCorrelações):
    #    value.to_excel(writer,'Permutação {}'.format(i))
    #writer.save()

    
    return minhasCorrelações

'''
IMPORTANTE: 
A variação "minhasCorrelações" é uma lista de DataFrames contendo todas as defasagens de todas as estações por meio de uma permutação. Considerando o R2 e o rmse
'''

# In[185]:

#O objetivo desta função é basicamente criar Variação de importância elevada para o desenvolvimento do modelo MLP_TEMP
minhasCorrelações = getValuesFromCorrelations(glb_defasagem,glb_diff)


# In[186]:


#Toma cuidado com as funções. A variavel gambiarra está conectada pelo tamanho da defasagem


# In[187]:



# In[188]:


#Funções de erro
def NashSutcliffe(real,previsao):
    real = np.array(real)
    previsao = np.array(previsao)
    mediaReal  = real.mean()
    mediaPrevisao = previsao.mean()
    somatorioDenominador = 0
    somatorioNominador = 0
    
    for x,y in zip(real,previsao):

        somatorioNominador = somatorioNominador + (y-x)**2

        somatorioDenominador = somatorioDenominador + (x - mediaReal)**2
    return 1 - (somatorioNominador / somatorioDenominador)

def Rsme(real,previsao):
    real = np.array(real)
    previsao = np.array(previsao)
    
    v1 = [((Yreal-Yprev)/Yreal)**2 for Yreal,Yprev in zip(real,previsao)]
    v1 = sum(v1)
    v1 = v1 / len(real)
    v1 = v1**(1/2)
    
    #ou
    v1 = ((sum([((Yreal-Yprev)/Yreal)**2 for Yreal,Yprev in zip(real,previsao)])) / len(real))**(1/2)
    return v1


#Função arima criado totalmente mal optimizado, refazer quando a preguiça passar.
def gerarDadosArima(serie,polinomioArima = (1,0,1,1,1,1,12),ano_inicial = None, ano_final = None,prever = 36,objetivo = 'usar',porc = 0.7,fastMode = False):
    if gerarDadosArima.tempoExecução >= 20:
        print('O tempo de execução da função anterior foi de aproximadamente {} segundos. recomenda-se ligar o método poolThread = True'.format(gerarDadosArima.tempoExecução))
    import time
    series = serie
    temp1 = time.process_time()
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    #porc = 0.7
    X = series.values
    sumario = ''
    size = int(len(X) * porc)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    if not fastMode:
        for t in range(len(test)):
            #pode substituir por apenas ARIMA em vez de SARIMAX. o ARIMA apenas pede a serie e a ordem mensal
            model = SARIMAX(history, order=polinomioArima[:3],seasonal_order=polinomioArima[3:]) 
            model_fit = model.fit(disp=0)
            sumario = model_fit.summary()
            output = model_fit.forecast() #já posso prever varios meses seguidos em vez de perder tempo... Realizar a atualização posteriormente !!!!!!!!!!!!!!!!!!!
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t] 
            if objetivo == 'validar':
                history.append(yhat)
            elif objetivo == 'usar':
                history.append(obs) #caso seja passado o 'obs' o Arima treinará com os dados de teste (para sua aplicação), caso seja usado o 'yhat' ele será treinado com a previsão (para validação)
            #print('Previsão=%f, valor esperado=%f' % (yhat, obs))
    else:
        model = SARIMAX(history, order=polinomioArima[:3],seasonal_order=polinomioArima[3:]) 
        model_fit = model.fit(disp=0)
        output = model_fit.predict(len(test))
        predictions = output
        out = pd.DataFrame([train,predictions],index=['Valores testados','Previsões']).T   
        return out

    
    if ano_inicial == None:
        de = serie.index[0]
    elif isinstance(ano_inicial,int):
        de = serie.index[ano_inicial]
    elif isinstance(ano_inicial,str):
        de = pd.to_datetime(ano_inicial)
    else:
        raise Exception('Erro na entrada dos parâmetros: formato inválido de tipo ano_inicial')

    if ano_final == None:
        ate = serie.index[-1]
    elif isinstance(ano_final,int):
        ate = serie.index[ano_final]
    elif isinstance(ano_final,str):
        ate = pd.to_datetime(ano_final)
    else:
        raise Exception('Erro na entrada dos parâmetros: formato inválido de tipo ano_final')

    if objetivo == 'validar':
        gerarDadosArima.tempoExecução = abs(temp1 - time.process_time())
        print('tempo = {} segundos'.format(gerarDadosArima.tempoExecução))
        # plot
        #pyplot.figure(figsize=(16,8))
        #pyplot.plot(test)
        #pyplot.plot(predictions, color='red')
        #pyplot.show()
        out = pd.DataFrame([test,predictions],index=['Valores testados','Previsões']).T   
        return out
    elif objetivo == 'usar':
        #print('valor:: {}'.format(prever))
        data_range = pd.date_range(start=de,periods=len(serie.values) + prever,freq='MS')
        out = pd.DataFrame([serie.values,
                            model_fit.predict(start=0,end=len(serie.values) + prever),
                             data_range],
                       index=['Série original','Arima ({},{},{}) ({},{},{})'.format(*polinomioArima),'Periodo']).T.set_index('Periodo')
        out.set_index = data_range
        return (out,model_fit)
gerarDadosArima.tempoExecução = 10


# ## Previsao

# # Redes Neurais

# In[189]:


# In[191]:


teste = pd.DataFrame([[1,2,]],columns= pd.MultiIndex.from_tuples(['r1','r2']))
idx = pd.IndexSlice

#teste[index] = pd.Series(range(10))
#teste

# In[209]:


entradas = None
saida = None
#a defasagem não esta incluida.. o includeEnoDefasation deve estar falso



'''
Esta função er para obter uma lista de entradas e saidas para o MLP_TEMP. 
'''
def criarTabela(estaçãoEntrada,num_var = 3,relaçãoCom = None,defasagem = 0,
                includeEnoDefasation = True,ConsiderEno=False,minhasCorrelações = minhasCorrelações,limitCriteria=0.7,
                diff = False,allow_defasation_choose = True
               ):
    idx = pd.IndexSlice
    #capturando as variaveis e as correlacionando 
    get_station = estaçãoEntrada
    relashionWith = relaçãoCom
    

    if defasagem > len(glb_defasagem):
        print('não é possível realizar a maior defasagem. Não foi gerado uma correlação com a defasagem desejado')
        print('definindo a defasagem máxima para o valor de {}'.format(glb_defasagem))
        defasagem = len(glb_defasagem)
    
    if diff not in glb_diff:
        print('não foi encontrado a diferenciação desejada. finalizando a aplicação')
        return
    
    #existe uma diferenciação?
    if diff:
        minhasCorrelações = minhasCorrelações[1]
    else:
        minhasCorrelações = minhasCorrelações[0]
        #defasagem = defasagem + gambiarra #solução gambiarra, pois são 120 correlações: (60 sem diferenciação e 60 com)
    correlação = minhasCorrelações[defasagem].loc[get_station,idx[:,'Grau de correlação']]
    
    #Filtro na correlação para se obter apenas os valores que respeitam o R² definido.
    correlação = correlação[(correlação > limitCriteria) | (correlação < -limitCriteria)].copy()

    if isinstance(correlação.columns,pd.core.indexes.multi.MultiIndex):
        correlação.columns = [x[-1] for x in correlação.columns] #converte para uma tabela simples para análise
    #pegar os nomes das maiores num_var correlações
    cor = correlação.nlargest(num_var
                              ,['Grau de correlação']).index #Considerar as defasagens para o cálculo

 #   num_var = len(cor) if len(cor) < num_var else pass
    if len(cor) < num_var:
        num_var = len(cor)
    
    tipo,estação = get_station.split(' de ',1)
    
    #onde ocorre a captura dos dados, de fato.
    entrada = result_multindex.loc[:,idx[tipo.upper(),estação.upper()]]
    saida = list()
    
    
    #Agora que eu já sei os nomes das séries que apresentam as boas correlações, já tá hora de captura-las na tabela raiz
    for value in cor:

        tipo,estação = value.split(' de ',1)

        if relashionWith is not None:
            if tipo != relashionWith:
                continue
                
        try:
            tipo = tipo.upper()
            tt = result_multindex.loc[:,idx[tipo.upper(),estação.upper()]]
            
        except:
            try:
                tt = result_multindex.loc[:,idx[tipo.lower().capitalize(),estação.lower().capitalize()]]
            except:
                print('correlação não realizada.. pulando...')
                continue
            
        saida.append(tt)
    #juntando todos os dados em uma única tabela:
    #Esta primeira linha cria uma dataframe de linhas com todos os dados seguindo a série para as colunas
    tabela_operação = pd.DataFrame([entrada,*saida,])
    
    # a tranposta coloca-os no padrão, cujas linhas são o tempo.
    tabela_operação = tabela_operação.T
    
    
    if diff == True:
        #print('realizando diferenciação') Nesta situação a diferenciação não considera a defasagem, cuidado!
        tabela_operação.iloc[:,1:] = tabela_operação.iloc[:,1:].diff().copy()
        '''
        Informação importante com relação a diferenciação:
        A função abaixo permite realizar a diferenciação da coluna, o que significa que pode-se haver um deslocamento das variáveis. Logo, deve-se ter em mente que, quando ocorre uma diferenciação da variável 
        B,C,D hipotética em relação a uma variável hipotética A, significa que estamos deslocamento a variável A para frente, logo,
        a diferenciação realiza também o inverso da defasagem (não em sentido, mas em variáveis correlacionadas)


        Em outras palavras, é necessário apagar uma defasagem para "corrigir" isto, para então o programa interpretar de forma correta.
        '''
        if defasagem > 0:
            defasagem = defasagem - 1
        else:
            if allow_defasation_choose:
                print('a variável defasagem está com o valor zero, e a chamada da função exige um lista diferenciável. Atribuindo uma defasagem igual a um para o funcionamento do método.\
                    Para mudar isto, muda o parâmetro "allow_defasation_choose" para False')
            else:
                raise Exception("Não foi possível realizar a iteração. Valor de defasagem inválido para a diferenciação escolhida")
        
    if defasagem > 0:
        #print('relizando defasagem')
        tabela_operação.iloc[:,1:] = tabela_operação.iloc[:,1:].shift(defasagem).copy()
    
    #eu não entendi o que eu quis fazer aqui, mas funciona, então deixa quieto.
    xx = tabela_operação.dropna()
    xx.columns = pd.MultiIndex.from_tuples(xx.columns)
    return xx




#unused
#ERRO, as relações com os dados Enos sempre serão retornados na própria função
#criarTabela, visto que é o objetivo do PIBIC. Portantanto será ignorado
def criarCorrelaçãoListada(dadosEntrada,relacionadosCom):
    temp = [dadosEntrada.lower() + ' de ' + value.lower() for value in result_multindex.loc[:,dadosEntrada.upper()].columns]
    return [criarTabela(estEntrada.lower(),relaçãoCom=relacionadosCom.lower()) for estEntrada in temp]


#capturar todos os dados possíveis relacionados com os Enos, ou outro conforme determinado, em relação a uma variavel qualquer..
#montando um DataFrame com as informações obtidas..
"""
A função abaixo possui uma falha de segmentação. É bem provável que a função tenha erro caso a defasagem requerida
não esteja carregada na função 'relacionarDados' !
"""


import pickle

#para um melhor cálculo dos erros
from sklearn.metrics import mean_squared_error
from math import sqrt
getLastData = True

#preparando as entradas e saida de rede neural
defasagem = 2
porcentagem = 0.1
#variavel necessária = tabela_operação


#entradas = tabela_operação.iloc[:,1].values   
#saida = tabela_operação.iloc[:,0].values

#tbo = criarTabela('chuva de padronal',num_var=4,defasagem=4,limitCriteria=0.7)


# In[ ]:


neural_net = MLPRegressor(hidden_layer_sizes=[17,12],max_iter=2000)


# # O professor pediu para eu fazer manualmente os dados.. então bora lá...

# In[ ]:


tbo = pd.DataFrame()
tbo['COTA'] = yy[0].iloc[:,0].resample('MS').mean()
tbo['ENO'] = y[3].iloc[:,2:3].resample('MS').mean()


# In[ ]:


#O objetivo desta função é realizar o que o professor pediu para ser feito para ser colocado no relatório
#Uma combinação com todos os dados enos juntos..
#tipo: cota = 0, vazão = 1, chuva = 2

'''
é preciso obter as seguintes informações da saida desta função:
A Estação que está sendo treinada
A rede neural utilizada para os dados de treino
As entradas, assim como suas defasagens.
As entradas e as saidas em formato de séries temporais para a plotagem


'''
#Entradas : São as entradas da rede neural, no nosso caso, os dados Enos.
def getBMLP(entrada_calibra,entrada_valida,saida_calibra,saida_valida,logfilename='Log de redes neurais.txt',datafilename='redesTreinadas.sav',
       num_redes=5,num_max_camadas_intermediarias=3,num_max_neuronios=15,forceCreateNewNet=True, neural_net_structured = None,):
    
    nameData = datafilename
    logfile = logfilename
    
    if os.path.isfile(nameData) and not forceCreateNewNet:
        print('modelos de redes neurais encontrados. Tentando abrir..')
        print('espero que não')
        with open(nameData,'rb') as file:
            list_neural_nets = pickle.load(file)
    else:
        print('Criando novos modelos de redes neurals artificiais')
        minhasRedes = list()
        for x in range(num_redes):
            value = np.random.random_integers(1,num_max_camadas_intermediarias) #defina aleatoriamente o número de camadas intermediárias
            value = np.random.random_integers(1,num_max_neuronios, size=(value)) #define os números de neuronios em cada camada
            minhasRedes.append(MLPRegressor(hidden_layer_sizes=value,max_iter=1000))
        list_neural_nets = list()
#        entrada_calibra,entrada_valida,saida_calibra,saida_valida = model_selection.train_test_split(entradas,saida,test_size=porcentagem,random_state=7,shuffle=shuffle)
        with open(logfile,'w') as file:
            file.write('Índice da rede - Rsme - NashSutcliffe\r\n')          
            for ind,rede in enumerate(minhasRedes):
                rede.fit(entrada_calibra,saida_calibra)
             
                previsões = rede.predict(entrada_valida)
                reais = saida_valida

                Nash = r2_score(reais,previsões)
                rsme = sqrt(mean_squared_error(reais,previsões))
                resposta = '{} - {} - {}'.format(ind,rsme,Nash)

                file.write(resposta + '\r\n')
                list_neural_nets.append((rede,rsme,Nash))
            print('convergencia procurada em todas as redes geradas..')        
            #Atualização  2020 : Rede neural criada anteriormente por uma possível chamada anterior desta função. O Objetivo é verificar se o modelo criado é satisfatório para a nova estação.
            if neural_net_structured is not None:
                rede = neural_net_structured
                rede.fit(entrada_calibra, saida_calibra)
                previsões = rede.predict(entrada_valida)
                reais = saida_valida
                Nash = r2_score(reais,previsões)
                rsme = sqrt(mean_squared_error(reais, previsões))
                resposta = '{} - {} - {}'.format(ind,rsme,Nash)
                file.write(resposta + '\r\n')
                list_neural_nets.append((rede,rsme,Nash))           
            
            with open(nameData,'wb') as filename:   
                pickle.dump(list_neural_nets,filename)
    redesTreinadas  = np.array(list_neural_nets)
    #onde está a rede neural com melhor resultado na lista criada? procurando...
    for ind,value in enumerate(redesTreinadas):
        if value[2] == redesTreinadas[:,2].max():
            global indMax
            indMax = ind
            break
        
    #organizado melhor as variaveis
    #input's
    print('indice máximo',indMax)

    return minhasRedes[indMax]



#Função que será coringa para MLP_TEMP MLP_MULTI e MLP_NUEVO
def getModelByName(model_name = 'TEMP',output = None, inputs = None, name = "unknow_model", **option):
    if output is None:
        print("O Modelo não pode ser executado sem a presença de uma dado de saída objetivo, retornando nulo")
        return None
    
    out_calibra, out_valida, in_calibra, in_valida = model_selection.train_test_split([output,inputs],)
    
    rede = getBMLP(in_calibra, in_valida, out_calibra, out_valida,logfilename = model_name + "_" + name)
    
    pass


out_calibra, out_valida, in_calibra, in_valida = model_selection.train_test_split([output,inputs],)

rede = getBMLP(in_calibra, in_valida, out_calibra, out_valida,logfilename = model_name + "_" + name)

#Por enquanto irei desconsiderar o model_name, pois apenas quero facilitar a análise



#instruções de  alto nivel
def gen_system_neural_net(tipo = 0,optimizer=False,numMeses = 24,shuffle=False,withPlots = True,
              ignoreNegativeError = False,estaçãoID = 0,useStaticNeuralNet = True):
    global analyser
    analyser = list()
    '''
    Neste primeiro trecho, o objetivo é criar uma tabela que contenha a entrada de dados e as saida
    para melhor serem trabalhadas.
    '''
    
    tbo = pd.DataFrame()
    BMLP_forEachMonth = list() #Uma lista das melhores redes neurais obtidas em cada defasagem (não é a melhor de todas!)
    index_col = estaçãoID #Qual a estação especificamente?
    tbo[yy[tipo].columns[index_col]] = yy[tipo].iloc[:,index_col].loc['1980':'2016']

    r_2_test = 0.0
    BestDefasationIndex = None
    #tbo['ENO'] = y[3].iloc[:,0].loc['1980':'2016']
    for t in y[3]:
        tbo[t] = y[3][t]
    #Criando um modelo de redes neurais báse para termos uma relação comparativa mais precisa
    baseNeuralNetwork  = MLPRegressor(hidden_layer_sizes=[12, 17],max_iter=3000)
    analyser.append(['modelo inicial antes de começar as defasagens',deepcopy(tbo)])
    
    
    for intervalo in range(numMeses):
        tbo.iloc[:,1:] = tbo.iloc[:,1:].shift(1)
        tbo = tbo.dropna()
        saida = tbo.iloc[:,0].values
        entradas = tbo.iloc[:,1:].values
        analyser.append(deepcopy(tbo))
        entrada_calibra,entrada_valida,saida_calibra,saida_valida = model_selection.train_test_split(entradas,saida,shuffle=shuffle,test_size = 0.1)
        if optimizer:
            rede = getBMLP(entrada_calibra,entrada_valida,saida_calibra,saida_valida)
        else:
            if useStaticNeuralNet:
                rede = deepcopy(baseNeuralNetwork)
            else:
                rede = MLPRegressor(hidden_layer_sizes=[12, 17],max_iter=3000)
            if entradas.shape[1] > 1:        
                rede.fit(entrada_calibra,saida_calibra)
            else:
                rede.fit(entrada_calibra.reshape(-1,1),saida_calibra.reshape(-1,1))
                entrada_valida = entrada_valida.reshape(-1,1)
                saida_valida = saida_valida.reshape(-1,1)
        previsões = rede.predict(entrada_valida)
        reais = saida_valida
        erro = r2_score(reais,previsões)

        if ignoreNegativeError and error < -1:
            #analyser.append((erro,rede.predict(entrada_valida),saida_valida,rede))
            pass
            
        BMLP_forEachMonth.append((erro,rede.predict(entrada_valida),saida_valida,deepcopy(rede)))        
        
        if intervalo == numMeses - 1:
            pass
        
        #O r² encontrado é o melhor até então?
        if erro > r_2_test:
            print('o erro atual é ',r_2_test,' e o novo é ',erro)
            r_2_test = erro
            BestDefasationIndex = intervalo
        if withPlots and erro > 0.7:
            if shuffle == True:
                msn = 'considerando'
            else:
                msn = 'desconsiderando'
            print(f'na função interna, de estação ID {estaçãoID},{msn} cruzamento de dados, e de defasagem de {intervalo} mes(es), estamos plotando:')
            print('os valores parciais para validação')
            plt.figure(figsize=(18,6),)
            plt.plot(previsões,'b',label='Dados previstos')
            plt.plot(reais,'g')
            plt.show()
            print('os valores totais da série')
            plt.figure(figsize=(18,6),)
            plt.plot(rede.predict(entradas),'b',label='Dados previstos')
            plt.plot(saida,'g')
            plt.show()
            print("fim :)")
            
    print('fim da execução do script..')
    return entradas,saida,BMLP_forEachMonth,BestDefasationIndex

 #gen_system_neural_net(numMeses=30)



# In[ ]:


ID = 0
meses = 11
otimizador = False

#A variavel em questão seŕá uma lista de redes com as estações em questão
EstaçõesComRedesTreinadas = list()

#Tentando encontrar uma relação com todas as estações de cotas obtidas. (Por padrão, todas as estações são utilizadas, nomeadas por um ID numérico)
def generateNeuralNetSystem(IDstation = [0,1,2,3,4,5],Tentativas = 3, set_graphics_output = "context_map",tipo = 0):
    with open(set_graphics_output,'wb') as file:
        for ID in IDstation:
            founded = False
            
            #Como as redes são geradas de forma aleatória, há sempre uma chance de se obter uma rede neural não convergida, por isso, é interessante se ter algumas tentativas.
            for tentativas in range(Tentativas):
                print(f'tentativa {tentativas + 1}')
                print('consideração com a validação direta')
                entr,said,debug,indexAcess = gen_system_neural_net(optimizer=otimizador,numMeses=meses,withPlots=False,estaçãoID=ID,tipo = tipo)
                print('consideração com a validação cruzada')
                entr_shuf,said_shuf,debug_shuf,indexAcess_shuf = gen_system_neural_net(optimizer=otimizador,shuffle=False,numMeses=meses,
                                                                                withPlots=True,estaçãoID=ID,tipo = tipo)
                rresult = max([debug[indexAcess][0],debug_shuf[indexAcess_shuf][0]])
                if rresult > 0.7:
                    msg = f"TOTAL_EMB_{ID}"
                    print('considerando um valor de erro máximo de',rresult)
        #            '''
                    print('plotando o gráfico total da série sem embaralhamento')
                    plt.figure(figsize=(18,6))
                    temp = debug[indexAcess][3].predict(entr)
                    plt.plot(temp,label='previsão')
                    plt.plot(said,label='valor real')
                    plt.savefig(f"graphics/TOTAL_AEMB_{ID}", )
                    plt.show()
    

            
                    print('plotando o gráfico total da série com embaralhamento')
                    plt.figure(figsize=(18,6))
                    
                    plt.savefig(f"graphics/TOTAL_CEMB_{ID}", )
                    
                    plt.plot(debug_shuf[indexAcess_shuf][3].predict(entr_shuf))
                    plt.plot(said_shuf,)
                    plt.show()
    
                    ### Resposta aos dados com base nos dados de calibração
                    msg = f"PARC_EMB_{ID}"
                    print('plotando o gráfico de validação sem embaralhamento')
                    plt.figure(figsize=(18,6))

       
                    plt.savefig(f"graphics/VAL_AEMB_{ID}", )
                    plt.plot(debug[indexAcess][1])
                    plt.plot(debug[indexAcess][2],)
                    plt.show()
    
                    print('plotando o gráfico de validação com embaralhamento')
                    plt.figure(figsize=(18,6))
                    plt.plot(debug_shuf[indexAcess_shuf][1])
                    plt.plot(debug_shuf[indexAcess_shuf][2])
                    plt.savefig(f"graphics/VAL_CEMB_{ID}", )
                    plt.show()
        #            '''
                    print('para a rede treinada diretamente, de r² {} , obtemos para uma defasagem de {} '.format(debug[indexAcess][0],indexAcess+1))
                    print('para a rede treinada por cruzamento, de r² {} , obtemos para uma defasagem de de {} '.format(debug_shuf[indexAcess_shuf][0],indexAcess_shuf+1))
                    print('\r\n')
                    Founded = True
    
                    ### Plotando gráfico de erro quadŕativo
    
                    #vamos fazer um comparativo 
    
                    #entr,said,debug,indexAcess = gen_system_neural_net(optimizer=False,numMeses=60,withPlots=False)
                    #entr_shuf,said_shuf,debug_shuf,indexAcess_shuf = gen_system_neural_net(optimizer=False,shuffle=True,numMeses=60,
                    #                                                            withPlots=False)
    
                    for value in range(len(debug)):
                        l1 = np.array([debug[X][0] for X in range(len(debug))])
                        l2 = np.array([debug_shuf[X][0] for X in range(len(debug_shuf))])
                    plt.figure(figsize=(18,6))
                    plt.plot(l1,)
                    plt.plot(l2,)
                    plt.ylim((-1,1))
                    plt.savefig(f"graphics/ERROR_{ID}",)
                    plt.show()
                    break
                else:
                    print(f'infelizmente, não foi encontrado uma rede com boa convergência (o máximo foi de {rresult*100}%) para esta estação. pulando...')
                    continue
            if Founded == True:
                print('encontrada uma relação neural!')
    
                EstaçõesComRedesTreinadas.append({
                    'EstaçãoID' : ID,
                    'embug normal' : (entr,said,debug,indexAcess),
                    'embug cruzado' : (entr_shuf,said_shuf,debug_shuf,indexAcess_shuf),   
                })
                Founded = False
    
            else:
                print('não achei nada, vida que segue para a próxima estação')
    return EstaçõesComRedesTreinadas



# In[ ]:



from itertools import permutations



#Atualização 2020

ntable = result_multindex

print("Fim da execução da aplicação")



dados_cota = result.loc[:,"Cota"]
dados_enos = result.loc[:,"Eno"]


rr = list(map(lambda x : list(permutations(range(1,7),x)),range(1,3)))

# Esta função trará a combinção de entradas e saidas dos dados, ela não retorna nenhuma defasagem entre as variáveis, tendo então que ser feito por fora do iterador
def getIOCombination(set_data_output = dados_cota, set_data_input = dados_enos,n_iteration = [[1],[2],[3],[4],[5],[6],[1,2,3,]], defasation = [0], diferentiation = [False]):
    #print("Entrando")
    realloc_iteration = [np.array(n_iter) - 1 for n_iter in n_iteration]
    for name_serie, serie in set_data_output.items():
        for index_acess in realloc_iteration:
            #print(f"Explicando o package, de tipo {type(index_acess)} e de informação: {index_acess}. ")
            #index_acess = np.array(index_acess)
            #print("Loopando", type(index_acess))
            dataframe = pd.concat([serie,set_data_input.iloc[:,index_acess]], axis = 1, sort = False).dropna()
            #print("Loopando", index_acess)
            saida = dataframe.iloc[:,[0]]
            entrada = dataframe.iloc[:,1:]
            #print('len',len(entrada))
            yield entrada,saida
    yield None

for x in getIOCombination():
    print(f"entrada : {x[0].columns} '\t' :::::'\t' {x[1].columns}")





output = None
inputs = None
out_calibra, out_valida, in_calibra, in_valida = model_selection.train_test_split([output,inputs],)
rede = getBMLP(in_calibra, in_valida, out_calibra, out_valida,logfilename = "sem nome.log")


