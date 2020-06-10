# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:04:54 2020

@author: jsvel
"""
# intento de codificar toda una base completa

import pandas as pd # libreria para trabajar con dataframes
import re
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer # funcion para convertir los datos a una matriz extensa
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


#model = GaussianProcessClassifier(1.0 * RBF(1.0))
#model = 

# es el directorio para trabajar con diferentes
#dir= 

from matplotlib import pyplot as plt # libreria para graficar
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree #, clasificacion usando cluster jerarquico


#pabierta es un dataframe con una sola columna con los datos que se quieres codificar
#ejemplos es un dataframe con 2 columnas "ejemplo" "codigo"
#tabla es un dataframe dónde tenemos el nombre de los ejemplos y lo unimos a los códigos



# la siguiente es una función para estructurar la base de datos para una codificación
# lo que se desea es agregar las columnas donde iran los nuevos datos
def base_cod(base,codificables):
    baseCod = pd.DataFrame() 
    nrow =  len(base.iloc[:,1])
    
    for j in list(base.columns):
        #agregamos columnas para los codigos
        if j in codificables:
           baseCod[j] = list(base[j])
           baseCod[j+'_COD1'] = [None]*nrow
           baseCod[j+'_COD2'] = [None]*nrow
           baseCod[j+'_COD3'] = [None]*nrow
        #si no hay que agregar columnas simplemente se la copiamos
        else:
            baseCod[j] = base[j]
            
    return baseCod    


# ejemplo del funcionamiento
#base2 = base_cod(base,codificables)

#base2.to_excel('COcasac.xlsx')


# funcion para entrenar el modelo y guardar un archivo del modelo
# los datos datos nuevos que no están en la lista
# list1 son los viejos y list2 los nuevos
def noDic (list1, list2):
   list_dif = [i for i in list2 if i not in list1]
   return list_dif
 

# p es una lista con las respuestas del código nuevo
# ejemplo es una lista con las respuestas del ejemplo
# cod son los codigos de las respuestas de la pregunta de ejemplo
# model es el tipo de modelo que se quiere usar
# cargar es el nombre del archivo donde hay un modelo guardado
#        
def codificar(p, ejemplo,cod, model  , tabla = [], cargar = "", pelim = []):
    #p hace referencia a las resopuestas de las preguntas que se quieren codificar
    # esta parte de cargar es para cargar un modelo en caso de que exista uno
    name = p.name
    ejemName = ejemplo.name
    nrow = len(p)
    p = [str(i) for i in p]
    
    ejemplo = [str(i) for i in ejemplo]
    def sas(i):
        try:
            i = int(i)
        except:    
            i = -999
        return i
    cod = [sas(i) for i in cod]
    def isvalue (i):
        j = not i == -999
        return j
    
    cond = [ isvalue(i) for i in cod]
    cod = np.array(cod)[cond]
    ejemplo = np.array(ejemplo)[cond]

    if cargar == "":
        
            #lo que viene a continuación es una solución vía countvectorizer
        
        # crea una matrix "sparse" (dispersa que tiene conteos de las palabras dentro de cada frase)
        vect = CountVectorizer( ngram_range = [1,4], min_df= 0 , stop_words = pelim, lowercase = True ).fit(ejemplo)
        vectAux = CountVectorizer( ngram_range = [1,1], min_df= 0 , stop_words = pelim, lowercase = True ).fit(ejemplo)
        #vect1 = CountVectorizer( ngram_range = [1,4], min_df= 0 , stop_words = pelim ).fit(p)
        vectAux2 = CountVectorizer( ngram_range = [1,1], min_df= 0 , stop_words = pelim, lowercase = True ).fit(p)
        #vect1 = CountVectorizer( min_df= 3 , stop_words = pelim ).fit(p)
        #vect.get_feature_names()
        
        # buscar cuales palabras no estan en el nuevo dic
        pEjemplo = vectAux.get_feature_names() 
        pNuevas = vectAux2.get_feature_names()
        nocodi =[i for i in pEjemplo if i not in pNuevas]
        if(len(nocodi) >= 1 ):
            print('Alerta, hay labras nuevas para la pregunta: '+ name + ' las palabras se guradaran en: ' +'PalabrasNuevas'+name+'.xslx' )
            nocodi = pd.DataFrame(nocodi)
            nocodi.to_excel('Temporal/Errores/PalabrasNuevas' + name + '.xlsx')
        
        #
        print(len(vect.get_feature_names()))
        print(len(p))
        
        #podemos trabajar con la matriz pura vect
        
        
        sparseEj = vect.transform(ejemplo)
        sparseP = vect.transform(p)
        
        #print(sparseX.A)
        #type(sparseX1.A)
        
        # usamos el modelo para ajustar
        # el modelo debe ser un objeto con el metodo fit
        fit1 = model.fit(sparseEj, cod)  
        # predecimos
        probs = fit1.predict_proba(sparseP)
        #categories = fit1.predict(sparseP)
        
        #print(probs)
        
        
        bestn = np.argsort(probs,axis =1)[:,-3:]
        #tomamos solo las predicciones sobre el 0.5 
        #print(fit1.classes_[bestn])
        def mayor(j,i):
            if j > i:
                k = j
            else:
                k = np.nan
            return k    
        for i in range(nrow):
            try:
                bestn[i,:] = [mayor(i,0.9) for i in bestn[i,:]]
            except:
                bestn[i,:] = 0
        
        prediction1 = fit1.classes_[bestn[:,2]]
        prediction2 = fit1.classes_[bestn[:,1]]
        prediction3 = fit1.classes_[bestn[:,0]]
        
        # guardamos los modelos
        import datetime
        fecha = str(datetime.datetime.now())
        from sklearn.externals import joblib
        
        try:
            filename = 'Temporal/' + ejemName +fecha+'.sav'
            joblib.dump(fit1, filename)
        except:
            print('\n no se ha podido guardar el mododelo')
            
            
    if cargar != "":
    # equivalentemente pudimos usar "a <> b" para expresar "a diferente de b"     
        print("aún estamos desarrollando la parte de cargar los modelos guardados")    
    
    return prediction1, prediction2, prediction3


# aqui va la base que se quiere codificar

#### funciones para arreglar los nombres de las bases
    
def fixnames(base):
    aux = [i.upper() for i in base.columns ]
    base.columns = [ re.sub(" ","",i) for i in aux ]
    
### algunas configuraciones previas

eliminar= pd.read_excel('Recursos/palabras_eliminar.xlsx')
elim = eliminar[eliminar['sugerencia'] == 0]
pelim = list(elim['lista'])

### 
####### nombres de las bases
Nuevo = '19125793_SATIS_HONDA_BDD 2'
Ejemplo = '19125793_SATIS_HONDA_BDD 1'


baseN = pd.read_excel(Nuevo + '.xlsx')
baseEj = pd.read_excel(Ejemplo + '.xlsx')
cdc = pd.read_excel('CC.xlsx')


# arreglamos los nombres
fixnames(baseN)
fixnames(baseEj)
#ejemplo = baseN['pregunta']
#cod = baseN['cod']
codificables = [str(i)  for i in cdc['Nomvar']]


model = MultinomialNB()


#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500,500,100),activation = 'relu', random_state=1)


#[i in codificables for i in list(baseN.columns)]
baseNcod = base_cod(baseN,codificables)


baseNcod.to_excel('cosa.xlsx')
#i= 0
#variable = cdc['Nomvar'][i]
#codificar(p = baseN[variable],ejemplo = baseEj[variable],cod = baseEj[variable],modelo = model)


def perdidoToNan(i):
    if i in ['-','']:
        j = np.nan
    else:
        j = i
    return j
    
def arreglarCod(respuesta,cod):
    cond = [ i in ['-',''] for i in  respuesta]
    aux = np.array(cod)
    
    aux[cond] = -999
    import re
    def inte(i):
        try:
            j = int(i)
        except:
            j = None
        return j
    
    aux = [str(i) for i in aux]
    aux = [re.sub("-999","-",i) for i in aux]
    aux = [inte(i) for i in aux]
    return aux


s= 0
#import re
faltantes = []    
for variable in codificables:
    try:
        # identificando lo que no son datos
        
        #baseNcod.loc[noempty][variable+'_COD1'],baseNcod.loc[noempty][variable+'_COD2'],baseNcod.loc[noempty][variable+'_COD3'] =  codificar(p = baseN[variable],ejemplo = baseEj[noempty2][variable],cod = baseEj[noempty2][variable+'_COD1'],model = model)
        baseNcod[variable+'_COD1'] , baseNcod[variable+'_COD2'] , baseNcod[variable+'_COD3'] =  codificar(p = baseN[variable],ejemplo = baseEj[variable],cod = baseEj[variable+'_COD1'],model = model)
        
        
        
        baseNcod[variable + '_COD1'] = arreglarCod(baseNcod[variable],baseNcod[variable + '_COD1'])
        baseNcod[variable + '_COD2'] = arreglarCod(baseNcod[variable],baseNcod[variable + '_COD2'])
        baseNcod[variable + '_COD3'] = arreglarCod(baseNcod[variable],baseNcod[variable + '_COD3'])
        #baseNcod[variable + '_COD2'] = [int(re.sub("-999","",str(i))) for i in baseNcod[variable+'_COD2'] ]
        #baseNcod[variable + '_COD3'] = [int(re.sub("-999","",str(i))) for i in baseNcod[variable+'_COD3'] ]
        print("se logro codificar " + variable)
        
        try:
            df1 = pd.DataFrame(baseNcod[variable])
            df2 = pd.DataFrame(baseNcod[variable + '_COD1'])
            df3 = pd.DataFrame(baseNcod[variable + '_COD2'])
            df4 = pd.DataFrame(baseNcod[variable + '_COD3'])
            df5 = pd.concat([df1,df2,df3,df4],axis = 1)
            df5.to_excel('Ejemplos/'+variable + '.xlsx')
            
            print('\n Logro guardar el ejemplo')
        except:    
            print('\n no se pudo guardar el ejemplo')
        
        
    except:    
        s = s+1
        faltantes.append(variable)
        print('No se codificó ' + variable  )


print('quedan pendientes por codificar ' + str(s))

baseNcod.to_excel('BaseCodificada.xlsx')

#baseNcod['P2'+'_COD1']
#[gsub("-1","",i) for i in baseNcod['P2'+'_COD1'] ]



