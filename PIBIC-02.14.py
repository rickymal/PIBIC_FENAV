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
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pdb
from copy import deepcopy
import os
import os
import pickle
from itertools import product
import mk_test as mks
from pandas import ExcelWriter
idx = pd.IndexSlice

#retira o limite de leitura das colunas
#from IPython.display import display
#pd.options.display.max_rows = None

#eliminta os avisos de depreciação e de falha na construção de redes neurais
#import warnings
#warnings.filterwarnings('ignore')


# ## Coleta de dados

# In[155]:


dados = pd.read_excel('COTA_TAPAJOS.xlsx',skiprows=1,index_col=0)
dados = dados.iloc[:,:7].copy()
dadosMensais = dados.resample('D').mean().resample('MS').mean()
dadosMensais.index.Names = 'Dados médios mensais'
dadosMensais = dadosMensais.interpolate()


# In[156]:


dados = pd.read_excel('VAZAO_TAPAJOS.xlsx',skiprows=1,index_col=0)
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


#a procura de regiões com dados faltantes de Cota
null_columns=dadosMensais[0].columns[dadosMensais[0].isnull().any()]
dadosMensais[0][null_columns].isnull().sum()
dadosMensais[0][dadosMensais[0].isnull().any(axis=1)][null_columns]


# In[161]:


#a procura de regiões com dados faltantes de Vazão
null_columns=dadosMensais[1].columns[dadosMensais[1].isnull().any()]
dadosMensais[1][null_columns].isnull().sum()
dadosMensais[1][dadosMensais[1].isnull().any(axis=1)][null_columns].head()


# In[162]:


#a procura de regiões com dados faltantes de Vazão
null_columns=dadosMensais[2].columns[dadosMensais[2].isnull().any()]
dadosMensais[2][null_columns].isnull().sum()
dadosMensais[2][dadosMensais[1].isnull().any(axis=1)][null_columns].head()


# In[163]:


#Como ainda não sei onde está os dados preenchidos:
#dadosMensais.interpolate(method='polynomial',order=2).plot(figsize=(18,6))
yy = [dadosMensais[0],dadosMensais[1],dadosMensais[2]]
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


#função para converter o formato de dados presentes no site do NOAA
#depois da atualização com os dados presentes no hidroweb 
def NOAA2HIDROWEB(data : pd.DataFrame()):
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


# In[168]:


#A grande seca El Nino em 2005
y[-1]['Indice de oscilação sul'.upper()].loc['2000':'2013'].plot(figsize=(16,6),subplots=True)


# In[169]:


#exportandos os dados para o excel porque eu desisti de trabalhar aqui
writer = ExcelWriter('Excel.xlsx')
#y[0].to_excel(writer,'dados PIBIC')

# DF TO CSV
y[0].to_csv('PythonExport.csv', sep=',')


# In[170]:


#deixando todas as variáveis com letra maiúscula para facilitar o filtro futuramente
y[0].columns = [x.upper() for x in y[0].columns]
y[1].columns = [x.upper() for x in y[1].columns]
y[2].columns = [x.upper() for x in y[2].columns]
y[-1].columns = [x.upper() for x in y[-1].columns]


# In[171]:


#Este bloco cria uma dataframa multindexada com todas as variáveis presentes coletadas, com suas chaves indicando 
#o tipo de variável que está sendo coletada
result = pd.concat([x.T for x in y],keys=['COTA','VAZÃO','CHUVA','ENO']).T
result.to_excel(writer,'Dados coletados')
result_multindex = result.copy()







result_multindex = result_multindex.iloc[:,:5]




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


# In[175]:


#Este método apenas junta todos os dados
result = pd.concat(y, axis=1, join_axes=[y[0].index])


# In[176]:


result = pd.concat([x.T for x in y],keys=['Cota','Vazão','Chuva','Eno']).T


# ## Inicio do teste de Mann Kendall

# In[177]:


#Realizando o teste de Man-kendall para cota e vazão

tabela_mks = {}


# In[178]:


def MannKendall(out):
    import mk_test as mks
    if isinstance(out,pd.Series):
        out = pd.DataFrame(out)
    temp = list()
    temp_name = list()
    for value in out.dropna().items():
        temp.append(mks.mk_test(value[1].values))
        temp_name.append(value[1].name)
    frame_dados = pd.DataFrame(temp,index=temp_name,columns=['Tendencia','h','p','z'])
    return frame_dados


# In[179]:


a1 = MannKendall(y[0])
a2 = MannKendall(y[1])
a3 = MannKendall(y[2])
a4 = MannKendall(y[3])
tabela_mks = pd.concat([a1,a2,a3,a4],keys=['Vazão','Cota','Chuva','NOAA'])


# In[180]:


a1 = MannKendall(y[0].iloc[:-100])
a2 = MannKendall(y[1].iloc[:-100])
a3 = MannKendall(y[2].iloc[:-100])
a4 = MannKendall(y[3].iloc[:-100])
tabela_mks = pd.concat([a1,a2,a3,a4],keys=['Vazão','Cota','Chuva','NOAA'])


# In[181]:


tabela_mks


# ## teste

# In[182]:


#Empilhando funções para multindexação

a1 = pd.DataFrame({
    'a1' : [1,2,3],
    'a2' : [2,3,4],
})
a2 = pd.DataFrame({
    'a3' : [6,7,8],
    'a4' : [6,8,3],
})
dm1 = pd.concat([a1,a2],axis=1,keys=['datafram1','datafram2'])
a3 = pd.DataFrame({
    'a5' : [10,23,4],
    'a6' : [23,32,44],
})
a4 = pd.DataFrame({
    'a7' : [34,2,53],
    'a8' : [46,12,23],
})
dm2 = pd.concat([a3,a4],axis=1,keys=['datafram3','datafram4'])
pd.concat([dm1,dm2],axis=1,keys=['SuperFrame1','SuperFrame2'])


# In[183]:


#Apenas um teste para verificar uma alternativa de multindexação
tt = pd.DataFrame([
    [1,2,3,4,5],
    [3,2,3,4,5],
    [4,2,3,4,5],
])
tt.columns = pd.MultiIndex.from_tuples([('defasagem','oloko',x) for x in tt.columns])
tt


# ## fim dos testes

# # Utilizando as correlações para o inicio do trabalho com redes neurais

# In[184]:


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



    #O multiprocessamento, apesar de mais fácil, não funciona muito bem no jupyter-notebook por isso é descartado.
    Multiprocessamento = False
    global minhasCorrelações
    minhasCorrelações = list()
    loadLastPearsonTable = False
    global gambiarra
    global diferen
    diferen = diferenciação
    gambiarra = len(defasagens) #variavel usada na função 'criaTabela' para obter os valores diferenciais 


    if os.path.isfile(savefile) and forceCreateNewFile:
    #    minhasCorrelações = loaded_model = pickle.load(open(filename, 'rb'))
        with open(savefile,'rb') as file:
            minhasCorrelações = pickle.load(file)
    else:
        print('não existe ')
        if Multiprocessamento:
            print('multiprocessamento ligado ')
            from multiprocessing import Pool

            def f(x,y):
                ff = result_multindex
    #            print('defasagem de {} meses'.format(x))
                #Log deve ser TRUE para a convertion funcionar (retornar a tabela pronta)
                return relacionarDados(ff,keys = None,log = True,convertion=True,defasagem = y,diferential=x) 

            if __name__ == '__main__':
                print('iniciando o pool de Thread')
                with Pool(5) as p:
                    minhasCorrelações = p.starmap(f,product([False,True],range(60),))
                    #minhasCorrelações = p.map(f, range(0,25)) #sem defasagem até 2 anos de defasagem # colocar o join
                    #Juntar as correlações
                    alreadyMade = False
                    for x in minhasCorrelações[0].columns:
                        if isinstance(x,tuple):
                            try:
                                print('correlação com multipĺas indexações')
                                data = pd.concat(minhasCorrelações,axis=1)
                                writer = ExcelWriter('Ppearson.xlsx')
                                data.to_excel(writer,'Correlações')
                                writer.save()
                                alreadyMade = True
                            except:
                                print('um erro ocorreu durante o processo de gravação')
                                alreadyMade = False
                                pass
                            with open('ListaCorrelacoesPearson.sav','wb') as file:
                                pickle.dump(minhasCorrelações,file)
                            break
                        else:
                            break     
                if not alreadyMade:    
                    print('dados gerados com sucesso.')
                    writer = ExcelWriter('Pearson.xlsx')
                    print('salvando os dados no Excel...')
                    for i,value in enumerate(minhasCorrelações):
                        value.to_excel(writer,'Permutação->{}'.format(i))
                    writer.save()
                print('dados salvos com sucesso!')
        else:
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
glb_defasagem = range(6)
glb_diff = [False,True]


# In[185]:


getValuesFromCorrelations(glb_defasagem,glb_diff)


# In[186]:


#Toma cuidado com as funções. A variavel gambiarra está conectada pelo tamanho da defasagem


# In[187]:


#Métodos para filtros de dados, apenas a título de análise. As correlações de Pearson serão feitam utilizando a variavel log gerada no método 'relacionarDados'
def FilterBy(*args,):
    if len(args) == 1:
        return [x for x,y in enumerate(Pearson_Table.index) if y[0][0].lower() == args[0].lower()]
    else:
        return [y for x,y in enumerate(Pearson_Table.index) if y[0][0].lower() == args[0].lower() and y[0][1].lower() == args[1].lower()]

def getBy(*args):
    if isinstance(Pearson_Table.index[0][0],str): #o índice não esta no formato de tupla ,mas de string?
        resposta = args[0].lower() + ' de ' + args[1].lower()
        return [x for x,y in enumerate(Pearson_Table.index) if y[0].lower() == resposta or y[1].lower() == resposta]  
    elif isinstance(Pearson_Table.index[0],tuple): #o índice esta no formato de tupla ,então é tabela multindexada..
        return [x for x,y in enumerate(Pearson_Table.index) if [y1.lower() for y1 in y[0]] == [arg.lower() for arg in args] or [y1.lower() for y1 in y[1]] == [arg.lower() for arg in args]]
        #return [x for x,y in enumerate(Pearson_Table.index) if y[0] == args or y[1] == args]
120
#A variavel log List contem uma Lista de (Tuplas de (Tuplas de (objetos)))
def filterBy(param1,param2,correlação = 0.95, p = 1e-100,logValue = log,getLogList = False):
    indexList = list()
    for indice,value in enumerate(logValue):
        estações = value[0:2]
        relação = value[-1]
        changegableList = list()
        for indx1,estação in enumerate(estações):
            if estação[0].lower() == param1.lower():
                if estação[1].lower() == param2.lower():
                    if relação[0] >= correlação and relação[1] <= p:                    
                        indexList.append(indice)                                          
    if getLogList:
        try:
            import numpy as np
            return np.array(logValue)[indexList]
        except:
            print('Biblioteca NumPy não encontrada. retornando a lista dos índices filtrada...')
    return indexList

def getFilter(tabela,*args):
    

    tabela_filtrada = tabela.iloc[getBy(*args)]
    return tabela_filtrada.index
    #return tabela_filtrada
    resp = list()
    r2 = list()
    for i,value in enumerate(tabela_filtrada.index):
        resp = list()
        for j,val in enumerate(value):
            if val[0].lower() == args[0].lower() and val[1].lower() == args[1].lower() and j == 0:
                resp.append(value)
                break
            else:
                value = list(value)
                value[0],value[1] = value[1],value[0]
                #tuple(value)
                resp.append(tuple(value))
                break
            r2.append(resp)
    r2 = np.array(r2)
    print('resultados')
    return r2
    return pd.MultiIndex.from_arrays(tuple(r2))


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
    pyplot.figure(figsize=(16,8))
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


import threading


# In[190]:


#não faço ideia do que eu quis fazer aqui embaixo
temp = y[0].copy()

temp.shift(1)
temp3 = temp.copy()
temp3.iloc[:,1:] = temp3.iloc[:,1:].shift(1)

temp2 = temp.copy()
temp.iloc[:,1:].diff()
temp2.iloc[:,1:] = temp2.iloc[:,1:].diff()

minhasCorrelações[0][0].loc['cota de bsm'].head()[minhasCorrelações[0][0].loc['cota de bsm'].head() > 0.98]

minhasCorrelações[0][2].head()

tt = minhasCorrelações[0][0].head(10).copy()
tt.columns = ['a','b']
for val in tt.nlargest(3,'a').index:
    print(val)

minhasCorrelações[0][4].head()
# até aqui..


# In[191]:


teste = pd.DataFrame([[1,2,]],columns= pd.MultiIndex.from_tuples(['r1','r2']))
idx = pd.IndexSlice

#teste[index] = pd.Series(range(10))
#teste
teste


# In[200]:


gambiarra


# In[209]:


entradas = None
saida = None
import pdb
#a defasagem não esta incluida.. o includeEnoDefasation deve estar falso
def criarTabela(estaçãoEntrada,num_var = 3,relaçãoCom = None,defasagem = 0,
                includeEnoDefasation = True,ConsiderEno=False,minhasCorrelações = minhasCorrelações,limitCriteria=0.7,
                diff = False
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
            if tipo != relashion:
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
        
        ##ATENTAR!!!!!!!!!
        if defasagem > 0:
            defasagem = defasagem - 1
        else:
            print('a variável defasagem está com o valor zero. Definindo a defasagem como valor um para que a diferenciação ocorra normalmente.')
        
    if defasagem > 0:
        #print('relizando defasagem')
        tabela_operação.iloc[:,1:] = tabela_operação.iloc[:,1:].shift(defasagem).copy()
    
    
    #eu não entendi o que eu quis fazer aqui, mas funciona, então deixa quieto.
    xx = tabela_operação.dropna()
    xx.columns = pd.MultiIndex.from_tuples(xx.columns)
    return xx


#unused
def criarTabela_intervalo(intervalo,estação,restrição = None):
    ll = list()
    
    #obtendo a variavel..
    resposta = criarTabela(estação,num_var=0,defasagem=0,limitCriteria=0.0,)    
    resposta.columns = pd.MultiIndex.from_tuples([('Entrada',resp[0],resp[1]) for resp in resposta.columns])
    ll.append(resposta)
    for y in [False,True]:
        for x in intervalo:
            resposta = criarTabela(estação,num_var=5,defasagem=x,limitCriteria=0.7,diff=y)
            if restrição is not None:
                idx = pd.IndexSlice
                print('restrição:',restrição)


                try:
                    resposta.loc[:,[idx[restrição.upper(),:]]]
                    mens = 'com' if y == True else 'sem'
                    resposta.columns = pd.MultiIndex.from_tuples([('{} meses: {} diferenciação'.format(x,mens),resp[0],resp[1]) for resp in resposta.columns])
                except:
                    print('ERROR..')
                    continue
                finally:
                    pass
            else:
                mens = 'com' if y == True else 'sem'
                resposta.columns = pd.MultiIndex.from_tuples([('{} meses: {} diferenciação'.format(x,mens),resp[0],resp[1]) for resp in resposta.columns])
            ll.append(resposta)
    return pd.concat(ll,axis=1,names=['Defasagem','Tipo de variavel','estação'])



#unused
#ERRO, as relações com os dados Enos sempre serão retornados na própria função
#criarTabela, visto que é o objetivo do PIBIC. Portantanto será ignorado
def criarCorrelaçãoListada(dadosEntrada,relacionadosCom):
    temp = [dadosEntrada.lower() + ' de ' + value.lower() for value in result_multindex.loc[:,dadosEntrada.upper()].columns]
    return [criarTabela(estEntrada.lower(),relaçãoCom=relacionadosCom.lower()) for estEntrada in temp]

#unused
#PARA MOSTRAR AS RELAÇÕES ENTRE O ENO E AS CHUVAS
seriesEnos = list()
def _mostrarRelações_(table = result_multindex,relação='CHUVA',aProcuraDe='ENO'):
    for estac in result_multindex.head().loc[:,idx[relação,:]].columns:
        nn = []
        try:
            print('para:' + estac[1])
            for value in range(120):
                x = criarTabela('{} de {}'.format(estac[0].lower(),estac[1].lower()),num_var=50,defasagem=value,
                                limitCriteria=0.7)
                for ind,v in enumerate(x.columns):
                    if v[0] == aProcuraDe:
                        seriesEnos.append([value,x.iloc[:,ind]])            
            try:
                seriesEnos[0][1].name
            except:
                print('sem relações')
                continue
            for v in seriesEnos:
                if v[0] < 60:
                    mensagem = 'sem'
                    vip = v[0]
                else:
                    mensagem = 'com'
                    vip = v[0] - 60

                print('defasagem {}, {} diff, para {}'.format(vip,mensagem,v[1].name))
                nn.append(v[0])
            nn = np.array(nn)

            print(np.diff(nn))
        except:
            print('erro desconhecido ocorreu, pulando estação...')
            continue


#capturar todos os dados possíveis relacionados com os Enos, ou outro conforme determinado, em relação a uma variavel qualquer..
#montando um DataFrame com as informações obtidas..
"""
A função abaixo possui uma falha de segmentação. É bem provável que a função tenha erro caso a defasagem requerida
não esteja carregada na função 'relacionarDado' !
"""

#ERROR
def criaTabela_filtered(estacaoBase = 'cota de três marias', relashion = 'ENO',multindexKing = False):
    vv = list()
    tbo = criarTabela(estacaoBase,limitCriteria=1.1)
    for differentiation in diferen:
        for val in range(gambiarra):
            ll = criarTabela(estacaoBase,num_var=50,defasagem=val,limitCriteria=0.8,diff=differentiation)
            for noum in ll.columns:
                if noum[0] == relashion:
                    vv.append((differentiation,val,ll[noum]))
                    if not multindexKing:
                        tbo[noum[-1] + ' {}|{}'.format(differentiation,val)] = ll[noum]
                    else:
                        #Ainda com defeito, problema na função para gera a multindexação
                        msgDiff = 'com diferenciação' if differentiation == True else 'sem diferenciação'
                        msnMes = 'mês' if val == 1 else 'meses'
                        msnMes = f'{val} {msnMes}'
                        estName = noum[:]
                        tupleIndex = (msgDiff,msnMes,*estName)

                        global pdindex
                        pdindex = pd.MultiIndex.from_tuples([*tupleIndex],)
                        tbo[tupleIndex] = ll[noum]
    
    tt = tbo.iloc[:,1:].values
    if tt.shape[0] == 0:
        print('não foi encontrado nenhuma relação para servir como entrada para a rede MLP_TEMP')
        return None
    return tbo



from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
import pickle

#para um melhor cálculo dos erros
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
getLastData = True

#preparando as entradas e saida de rede neural
defasagem = 2
porcentagem = 0.1
#variavel necessária = tabela_operação


#entradas = tabela_operação.iloc[:,1].values   
#saida = tabela_operação.iloc[:,0].values

#tbo = criarTabela('chuva de padronal',num_var=4,defasagem=4,limitCriteria=0.7)



#print(entradas)
#print(saida)
def procurarMLP(entradas = entradas,saida = saida,withPlots=True,nameData = 'ListaRedesTreinadas.sav'):
  
    from sklearn import model_selection
    import pickle
    #para um melhor cálculo dos erros
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.metrics import r2_score 
    
    entrada_calibra,entrada_valida,saida_calibra,saida_valida = model_selection.train_test_split(entradas,saida,test_size=porcentagem,random_state=7)
    #criando as redes
    if not os.path.isfile(nameData) or True:
        num_redes = 20
        num_max_camadas_intermedias = 5
        num_max_neuronios = 30
        minhasRedes = list()
        for x in range(num_redes):
            value = np.random.random_integers(1,num_max_camadas_intermedias) #defina aleatoriamente o número de camadas intermediárias
            value = np.random.random_integers(1,num_max_neuronios, size=(value)) #define os números de neuronios em cada camada
            minhasRedes.append(MLPRegressor(hidden_layer_sizes=value,max_iter=2000))
    #Sem o pool de Thread's considerar um log de saida 
    if False: #os.path.isfile(nameData) or getLastData: #and False ou or True
        print('espero que não')
        with open(nameData,'rb') as file:
            meuResult = pickle.load(file)
    else:
        print('criando novos modelos')
        from sklearn import model_selection
        meuResult = list()
        #entrada_calibra,entrada_valida,saida_calibra,saida_valida = model_selection.train_test_split(entradas,saida,test_size=porcentagem,random_state=7)
        with open('logRedesNeurais.txt','w') as file:
            file.write('Índice da rede - Rsme - NashSutcliffe\r\n')
            for ind,rede in enumerate(minhasRedes):

                rede.fit(entrada_calibra,saida_calibra)
                #previsões = [rede.predict(x.reshape(1,-1)) for x in entrada_valida]
                previsões = rede.predict(entrada_valida)
                reais = saida_valida

                #por alguma razão, os valores Nash estão dando erro. Ainda não se sabe a razão...
                #Nash = NashSutcliffe(previsões,reais)
                #rsme = Rsme(previsões,reais)
                Nash = r2_score(reais,previsões)
                rsme = sqrt(mean_squared_error(reais,previsões))
                resposta = '{} - {} - {}'.format(ind,rsme,Nash)
                if Nash > 0.9:
                    print('excelente convergencia encontrada..')
                    with open(nameData,'wb') as filename:   
                        pickle.dump(meuResult,filename)
                    break
                file.write(resposta + '\r\n')
                meuResult.append((rede,rsme,Nash))
            print('convergencia procurada em todas as redes geradas..')
            with open(nameData,'wb') as filename:   
                pickle.dump(meuResult,filename)
    redesTreinadas  = np.array(meuResult)
    #onde está a rede neural com melhor resultado na lista criada? procurando...
    for ind,value in enumerate(redesTreinadas):
        if value[2] == redesTreinadas[:,2].max():
            global indMax
            indMax = ind
            break
    #organizado melhor as variaveis
    #input's
    import matplotlib.pyplot as plt
    #v1 = np.array([mnh.reshape(1,-1) for mnh in entrada_valida])
    #v1 = np.array([respostas[indMax][0].predict(v) for v in v1])

    print('indice máximo',indMax)
    v1 = minhasRedes[indMax].predict(entrada_valida)
    v2 = saida_valida
    if withPlots:
        #output's
        print('dados:',Nash)
        plt.figure(figsize=(18,6),)
        plt.plot(v1,'b',label='Dados previstos')
        plt.plot(v2,'g')
        plt.show()
    return v1,v2,minhasRedes[indMax]


def procurarBMLP(estação = 'cota de três marias', withPlots = False,nameData='None.sav'):
    
    extraFields = dict()

    tbo = criaTabela_filtered(estacaoBase =estação)
    
    if len(tbo.columns) <= 1:
        print('não foi encontrado uma entrada para a rede neural MLP_TEMP')
        return None

    ttbo = tbo.dropna()
    entradas = ttbo.iloc[:,1:].values
    saida = ttbo.iloc[:,0].values
    extraFields['tabelaBase'] = ttbo.copy()

    
    return (*procurarMLP(entradas,saida,withPlots=withPlots,nameData=nameData),extraFields)


# In[208]:


diferen


# In[206]:


criarTabela(estaçãoEntrada='cota de bsm',defasagem=5,diff=True)


# In[211]:


criaTabela_filtered(estacaoBase='cota de bsm')


# In[194]:


glb_defasagem = range(6)
glb_diff = [False,True]


# In[195]:


tbo = criaTabela_filtered(estacaoBase='cota de bsm')


# # FIM

# ## primeira rede neural única

# ## Piscina de redes neurais

# v1,v2,_ = procurarMLP(withPlots=False)
# 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(18,6),)
# plt.plot(v1,'b',label='Dados previstos')
# plt.plot(v2,'g')
# 
# 
# #agora dá série inteira
# vv1 = saida
# vv2 = [entrada.reshape(1,-1) for entrada in entradas]
# vv2 = [_.predict(v) for v in vv2]
# 
# #output's
# plt.figure(figsize=(18,6))
# plt.plot(vv1,'b')
# plt.plot(vv2,'g')
# 
# plt.savefig('EnoCotaTresMarias.png')
# plt.show()

# In[196]:


def createMLP_TEMP(estação = 'cota de três marias',withPLots = True,):

    field = dict()
    
    #Função procurarMLP Acoplada para parâmetro de estação -> procurarBMLP
    response = procurarBMLP(estação = 'cota de três marias')

    if response is None:
        return None
    v1,v2,rede_neural,fields = procurarBMLP(estação='cota de três marias')

    ttbo = fields['tabelaBase']
    entradas = ttbo.iloc[:,1:].values
    saida = ttbo.iloc[:,0].values

    
    #agora dá série inteira
    vv1 = saida
    vv2 = entradas
    #vv2 = [_.predict(v) for v in vv2]
    vv2 = rede_neural.predict(vv2)
    if withPLots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18,6),)
        plt.plot(v1,'b',label='Dados previstos')
        plt.plot(v2,'g')


        print(r2_score(vv2,vv1),)
        #output's
        plt.figure(figsize=(18,6))
        plt.plot(vv1,'b')
        plt.plot(vv2,'g')

        plt.savefig('EnoCotaTresMarias.png')
        plt.show()
    field['neural_net'] = rede_neural
    field['baseTable'] = ttbo
    field['IO'] = [entradas,saida,]
    
    return field


# In[197]:


r1 = createMLP_TEMP(estação='cota de três marias',withPLots = True)


# #Função procurarMLP Acoplada para parâmetro de estação -> procurarBMLP
# v3,v4 = v1,v2
# v1,v2,rede_neural,fields = procurarBMLP(estação='cota de bsm')
# 
# ttbo = fields['tabelaBase']
# entradas = ttbo.iloc[:,1:].values
# saida = ttbo.iloc[:,0].values
# 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(18,6),)
# plt.plot(v1,'b',label='Dados previstos')
# plt.plot(v2,'g')
# 
# #agora dá série inteira
# vv1 = saida
# vv2 = entradas
# #vv2 = [_.predict(v) for v in vv2]
# vv2 = rede_neural.predict(vv2)
# 
# #output's
# plt.figure(figsize=(18,6))
# plt.plot(vv1,'b')
# plt.plot(vv2,'g')
# 
# plt.savefig('EnoCotaTresMarias.png')
# plt.show()

# In[ ]:


#diferentes maneiras de detectar o índice do valor mínimo
'''
saida_valida = vv2
list(saida_valida).index(min(saida_valida))
###
import operator
min_index, min_value = min(enumerate(saida_valida), key=operator.itemgetter(1))
min_index
###
import numpy as np
np.argmin(saida_valida)

'''


###


# # Fazendo as tentativas individuais de coleta de dados

# In[ ]:


#Test
ttable = pd.DataFrame()
ttable['est'] = yy[0].iloc[:,0]
ttable['eno'] = y[3].iloc[:,0]
temp = ttable.copy()
temp.iloc[:,1] = temp.iloc[:,1].shift(1)
temp
#end Test


# In[ ]:


neural_net=  MLPRegressor(hidden_layer_sizes=[17,12],max_iter=2000)


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
def getBMLP(entrada_calibra,entrada_valida,saida_calibra,saida_valida,logfilename='Log de redes neurais.txt',datafilename='redeTreinada.sav',
       num_redes=5,num_max_camadas_intermediarias=5,num_max_neuronios=30,forceCreateNewNet=True,withPlots=True,
            shuffle=False,useStaticNeuralNet = True):
    
    '''
    Não é necessário, pois sera passado como parâmetro as entradas e a saida
    tbo = pd.DataFrame()
    logTable = list()
    tbo['COTA'] = yy[0].iloc[:,0].loc['1980':'2016']
    #tbo['ENO'] = y[3].iloc[:,0].loc['1980':'2016']
    for t in y[3]:
        tbo[t] = y[3][t]

    saida = tbo.dropna().iloc[:,0].values #necessita ser uma Série
    entradas = tbo.dropna().iloc[:,1:].values #necessita ser um DataFrame
    '''
    import pdb
    nameData = datafilename
    logfile = logfilename

    
    if os.path.isfile(nameData) and not forceCreateNewNet:
        print('modelos de redes neurais encontrados. Tentando abrir..')
        print('espero que não')
        with open(nameData,'rb') as file:
            meuResult = pickle.load(file)
    else:
        print('modelos de redes neurais não encontrados. Criando novos modelos..')
        minhasRedes = list()
        for x in range(num_redes):
            value = np.random.random_integers(1,num_max_camadas_intermediarias) #defina aleatoriamente o número de camadas intermediárias
            value = np.random.random_integers(1,num_max_neuronios, size=(value)) #define os números de neuronios em cada camada
            minhasRedes.append(MLPRegressor(hidden_layer_sizes=value,max_iter=2000))
        from sklearn import model_selection
        meuResult = list()
#        entrada_calibra,entrada_valida,saida_calibra,saida_valida = model_selection.train_test_split(entradas,saida,test_size=porcentagem,random_state=7,shuffle=shuffle)
        with open(logfile,'w') as file:
            file.write('Índice da rede - Rsme - NashSutcliffe\r\n')
            for ind,rede in enumerate(minhasRedes):
                rede.fit(entrada_calibra,saida_calibra)
                #previsões = [rede.predict(x.reshape(1,-1)) for x in entrada_valida]
                previsões = rede.predict(entrada_valida)
                reais = saida_valida

                #por alguma razão, os valores Nash estão dando erro. Ainda não se sabe a razão...
                #Nash = NashSutcliffe(previsões,reais)
                #rsme = Rsme(previsões,reais)
                Nash = r2_score(reais,previsões)
                rsme = sqrt(mean_squared_error(reais,previsões))
                resposta = '{} - {} - {}'.format(ind,rsme,Nash)
                if Nash > 0.9:
                    print('excelente convergencia encontrada..')
                    with open(nameData,'wb') as filename:   
                        pickle.dump(meuResult,filename)
                    break
                file.write(resposta + '\r\n')
                meuResult.append((rede,rsme,Nash))
            print('convergencia procurada em todas as redes geradas..')
            with open(nameData,'wb') as filename:   
                pickle.dump(meuResult,filename)
    redesTreinadas  = np.array(meuResult)
    #onde está a rede neural com melhor resultado na lista criada? procurando...
    for ind,value in enumerate(redesTreinadas):
        if value[2] == redesTreinadas[:,2].max():
            global indMax
            indMax = ind
            break
    #organizado melhor as variaveis
    #input's
    import matplotlib.pyplot as plt
    #v1 = np.array([mnh.reshape(1,-1) for mnh in entrada_valida])
    #v1 = np.array([respostas[indMax][0].predict(v) for v in v1])
    print('indice máximo',indMax)
#   v1 = minhasRedes[indMax].predict(entrada_valida)
#   v2 = saida_valida
    return minhasRedes[indMax]

def _testaEno_(tipo = 0,paraRelatorio=False,optimizer=False,numMeses = 24,shuffle=False,withPlots = True,
              ignoreNegativeError = False,estaçãoID = 0,useStaticNeuralNet = True):
    global analyser
    analyser = list()
    '''
    Neste primeiro trecho, o objetivo é criar uma tabela que contenha a entrada de dados e as saida
    para melhor serem trabalhadas.
    '''
    from sklearn.model_selection import train_test_split
    import pdb
    from copy import deepcopy
    
    tbo = pd.DataFrame()
    BMLP_forEachMonth = list() #Uma lista das melhores redes neurais obtidas em cada defasagem (não é a melhor de todas!)
    index_col = estaçãoID #Qual a estação especificamente?
    tbo[yy[tipo].columns[index_col]] = yy[tipo].iloc[:,index_col].loc['1980':'2016']

    r_2_test = 0.0
    BestDefasationIndex = None
    #tbo['ENO'] = y[3].iloc[:,0].loc['1980':'2016']
    for t in y[3]:
        tbo[t] = y[3][t]
    '''
    tbo = pd.DataFrame()
    BMLP_forEachMonth = list() #Uma lista das melhores redes neurais obtidas em cada defasagem (não é a melhor de todas!)
    index_col = 0 #Qual a estação especificamente?
    tbo[yy[0].columns[0]] = yy[0].iloc[:,0].loc['1980':'2016']
    tbo
    #tbo['ENO'] = y[3].iloc[:,0].loc['1980':'2016']
    for t in y[3]:
        tbo[t] = y[3][t]
    tbo
    '''
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
#                print('multidimensão')
#                print('shape',entradas.shape)
#                print('shape',saida.shape)
#                entrada_calibra,entrada_valida,saida_calibra,saida_valida = model_selection.train_test_split(entradas,saida,shuffle=False)
#                rede = MLPRegressor(hidden_layer_sizes=[19, 50,  4,],max_iter=2000)

                rede.fit(entrada_calibra,saida_calibra)
                entrada_valida = entrada_valida
                saida_valida = saida_valida
            else:
#                print('shape',entradas.shape)
#                print('shape',saida.shape)
#                entrada_calibra,entrada_valida,saida_calibra,saida_valida = model_selection.train_test_split(entradas,saida,shuffle=False,)
#                rede = MLPRegressor(hidden_layer_sizes=[19, 50,  4,],max_iter=2000)
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

#_testaEno_(numMeses=30)


# In[ ]:


#Apenas para testes
x = f'henrique mauler {3 + 8}',
x[0]

type(i*i for i in range(10))

def Func(x : int,limit = 10) -> int:
    y = x
    
    for _ in range(limit):
        yield y
        y = y + x
        x = x + 1
        
for presente in Func(10):
    print(presente)
#Fim dos testes


# In[ ]:


ID = 0
meses = 11
otimizador = False

#A variavel em questão seŕá uma lista de redes com as estações em questão
EstaçõesComRedesTreinadas = list()

#Tentando encontrar uma relação com todas as estações de cotas obtidas.
def generateNeuralNetSystem(IDstation = 6,Tentativas = 20):
    for ID in range(IDstation):
        founded = False
        '''
        for tentativas in range(5):
            print('tá entrando nesse loop?')
            entr,said,debug,indexAcess = _testaEno_(optimizer=otimizador,numMeses=meses,withPlots=False,estaçãoID=ID)
            entr_shuf,said_shuf,debug_shuf,indexAcess_shuf = _testaEno_(optimizer=otimizador,shuffle=True,numMeses=meses,
                                                                        withPlots=False,estaçãoID=ID)

            if indexAcess != indexAcess_shuf:
                print('igualdade entre os métodos de validação não encontrados')
            if abs(debug[indexAcess][0] - debug_shuf[indexAcess_shuf][0]) < 0.2 and debug[indexAcess][0] > 0.7:
                print('divergencia entre os erros quadráticos baixo, considerando rede criada com sucesso')
                break
            if ID == 5:
                print('infelizmente, não foi encontrado nenhum valor')


            break
        ### Resposta aos dados com base na série original
        '''
        for tentativas in range(Tentativas):
            print(f'tentativa {tentativas + 1}')
            print('consideração com a validação direta')
            entr,said,debug,indexAcess = _testaEno_(optimizer=otimizador,numMeses=meses,withPlots=False,estaçãoID=ID)
            print('consideração com a validação cruzada')
            entr_shuf,said_shuf,debug_shuf,indexAcess_shuf = _testaEno_(optimizer=otimizador,shuffle=False,numMeses=meses,
                                                                            withPlots=True,estaçãoID=ID)
            rresult = max([debug[indexAcess][0],debug_shuf[indexAcess_shuf][0]])
            if rresult > 0.7:

                print('considerando um valor de erro máximo de',rresult)
    #            '''
                print('plotando o gráfico total da série sem embaralhamento')
                plt.figure(figsize=(18,6))
                temp = debug[indexAcess][3].predict(entr)
                plt.plot(temp,label='previsão')
                plt.plot(said,label='valor real')
                plt.show()

                print('plotando o gráfico total da série com embaralhamento')
                plt.figure(figsize=(18,6))
                plt.plot(debug_shuf[indexAcess_shuf][3].predict(entr_shuf))
                plt.plot(said_shuf,)
                plt.show()

                ### Resposta aos dados com base nos dados de calibração

                print('plotando o gráfico de validação sem embaralhamento')
                plt.figure(figsize=(18,6))
                #temp = debug[indexAcess][3].predict(debug[indexAcess][1])
                plt.plot(debug[indexAcess][1])
                plt.plot(debug[indexAcess][2],)
                plt.show()

                print('plotando o gráfico de validação com embaralhamento')
                plt.figure(figsize=(18,6))
                plt.plot(debug_shuf[indexAcess_shuf][1])
                plt.plot(debug_shuf[indexAcess_shuf][2])
                plt.show()
    #            '''
                print('para a rede treinada diretamente, de r² {} , obtemos para uma defasagem de {} '.format(debug[indexAcess][0],indexAcess+1))
                print('para a rede treinada por cruzamento, de r² {} , obtemos para uma defasagem de de {} '.format(debug_shuf[indexAcess_shuf][0],indexAcess_shuf+1))
                print('\r\n')
                Founded = True

                ### Plotando gráfico de erro quadŕativo

                #vamos fazer um comparativo 

                #entr,said,debug,indexAcess = _testaEno_(optimizer=False,numMeses=60,withPlots=False)
                #entr_shuf,said_shuf,debug_shuf,indexAcess_shuf = _testaEno_(optimizer=False,shuffle=True,numMeses=60,
                #                                                            withPlots=False)

                for value in range(len(debug)):
                    l1 = np.array([debug[X][0] for X in range(len(debug))])
                    l2 = np.array([debug_shuf[X][0] for X in range(len(debug_shuf))])
                plt.figure(figsize=(18,6))
                plt.plot(l1,)
                plt.plot(l2,)
                plt.ylim((-1,1))
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


#Vamos testar para ver se as redes estão funcionando e se eu não fiz uma programação errada...
#Variavel -> Estação -> informação no dicionário -> Debug -> defasagem -> dadoRequerido
redeTeste = generateNeuralNetSystem()[0]['embug cruzado'][2][5][3]
EstaçõesComRedesTreinadas[0]['embug cruzado'][2][5][0]

yy[0].loc['1985':'1987'].head()

y[3].loc['1985':'1987'].head()

saida = yy[0].loc['1985':'1987'].iloc[:,0].values
saida

entradas = y[3].loc['1985':'1987'].values
entradas

retorno = redeTeste.predict(entradas)

retorno

ddata = pd.DataFrame([retorno,saida],['Previsão','Valor real'],).T
ddata.iloc[:,0] = ddata.iloc[:,0].shift(6)
ddata.plot(figsize=(18,6))
print(r2_score(ddata.dropna().iloc[:,1].values,ddata.dropna().iloc[:,0].values))

if __name__ == '__ main__':
    print('ola mundo!')