#Practica de laboratorio, para comprender y aprender la metodología de la
#ciencia de datos concentrandose en las etapa de comprensión y preparación de datos

import pandas as pd
import numpy as np 
import re

pd.set_option('display.max_columns', None)

recetas = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0103EN-SkillsNetwork/labs/Module%202/recipes.csv") 
print("Data read into dataframe!")

recetas.head()
recetas.shape

#Entonces nuestro conjunto de datos consta de 57.691 recetas.
#Cada fila representa una receta y para cada receta se documenta la cocina
#correspondiente y si existen o no 384 ingredientes en la receta, comenzando con
#almendras y terminando con calabacín


#Comprobando si los ingredientes para hacer un sushi existen en nuestro marco de datos:
ingredientes = list(recetas.columns.values)

print([match.group(0) for ingredientes in ingredientes for match in [re.compile(".*(rice).*").search(ingredientes)]if match])
print([match.group(0) for ingredientes in ingredientes for match in [re.compile(".*(wasabi).*").search(ingredientes)]if match])
print([match.group(0) for ingredientes in ingredientes for match in [re.compile(".*(soy).*").search(ingredientes)]if match])

# arroz es rice
# wasabi es wasabi
# la salsa de soya es soy

#Algoritmo de aprendizaje automático seleccionado son los árboles de decisión

recetas['country'].value_counts() #Tabla de frecuencia
recetas['country'].unique().tolist() # Lista de paises

#Observando la tabla anterior, podemos hacer las siguientes observaciones:
# 1. La columna de cocina está etiquetada como País, lo cual es inexacto
# 2. Los nombres de la cocinas no son consistentes ya que no todos comienzan con
#una primera letra mayúscula
# 3. Algunas cocinas se dublican como variación del nombre del país, como Vietnam y Vietnamita
# 4. Algunas cocinas tienen muy pocas recetas


column_names = recetas.columns.values
column_names[0] = "cocina"
recetas.columns = column_names

recetas

recetas["cocina"] = recetas["cocina"].str.lower()

recetas.loc[recetas["cocina"] == "austria", "cocina"] = "austrian"
recetas.loc[recetas["cocina"] == "belgium", "cocina"] = "belgian"
recetas.loc[recetas["cocina"] == "china", "cocina"] = "chinese"
recetas.loc[recetas["cocina"] == "canada", "cocina"] = "canadian"
recetas.loc[recetas["cocina"] == "netherlands", "cocina"] = "dutch"
recetas.loc[recetas["cocina"] == "france", "cocina"] = "french"
recetas.loc[recetas["cocina"] == "germany", "cocina"] = "german"
recetas.loc[recetas["cocina"] == "india", "cocina"] = "indian"
recetas.loc[recetas["cocina"] == "indonesia", "cocina"] = "indonesian"
recetas.loc[recetas["cocina"] == "iran", "cocina"] = "iranian"
recetas.loc[recetas["cocina"] == "italy", "cocina"] = "italian"
recetas.loc[recetas["cocina"] == "japan", "cocina"] = "japanese"
recetas.loc[recetas["cocina"] == "israel", "cocina"] = "israeli"
recetas.loc[recetas["cocina"] == "korea", "cocina"] = "korean"
recetas.loc[recetas["cocina"] == "lebanon", "cocina"] = "lebanese"
recetas.loc[recetas["cocina"] == "malaysia", "cocina"] = "malaysian"
recetas.loc[recetas["cocina"] == "mexico", "cocina"] = "mexican"
recetas.loc[recetas["cocina"] == "pakistan", "cocina"] = "pakistani"
recetas.loc[recetas["cocina"] == "philippines", "cocina"] = "philippine"
recetas.loc[recetas["cocina"] == "scandinavia", "cocina"] = "scandinavian"
recetas.loc[recetas["cocina"] == "spain", "cocina"] = "spanish_portuguese"
recetas.loc[recetas["cocina"] == "portugal", "cocina"] = "spanish_portuguese"
recetas.loc[recetas["cocina"] == "switzerland", "cocina"] = "swiss"
recetas.loc[recetas["cocina"] == "thailand", "cocina"] = "thai"
recetas.loc[recetas["cocina"] == "turkey", "cocina"] = "turkish"
recetas.loc[recetas["cocina"] == "vietnam", "cocina"] = "vietnamese"
recetas.loc[recetas["cocina"] == "uk-and-ireland", "cocina"] = "uk-and-irish"
recetas.loc[recetas["cocina"] == "irish", "cocina"] = "uk-and-irish"

recetas

#Eliminando las cocinas con <50 recetas
recetas_counts = recetas["cocina"].value_counts()
cocina_indices = recetas_counts > 50

cocina_to_keep = list(np.array(recetas_counts.index.values)[np.array(cocina_indices)])

rows_before = recetas.shape[0] # número de filas del marco de datos original
print("El número de filas del marco de datos original es {}.".format(rows_before))

recetas = recetas.loc[recetas['cocina'].isin(cocina_to_keep)]

rows_after = recetas.shape[0] # number of rows of processed dataframe
print("El número de filas del marco de datos procesados es {}.".format(rows_after))

print("{} filas eliminadas!".format(rows_before - rows_after))

recetas = recetas.replace(to_replace="Yes", value=1)
recetas = recetas.replace(to_replace="No", value=0)

# Analicemos los datos para obtener las recetas que contienen arroz, soja, wasabi y algas

recetas.head()

check_recetas = recetas.loc[ (recetas["rice"] == 1) &
                             (recetas["soy_sauce"] == 1) &
                             (recetas["wasabi"] == 1) &
                             (recetas["seaweed"] == 1)
                           ]

check_recetas

#Contemos los ingredientes en todas las recetas

ing = recetas.iloc[:, 1:].sum(axis=0)

ingredientes = pd.Series(ing.index.values, index = np.arange(len(ing)))
count = pd.Series(list(ing), index = np.arange(len(ing)))

# create the dataframe
ing_df = pd.DataFrame(dict(ingredientes = ingredientes, count = count))
ing_df = ing_df[["ingredientes", "count"]]
print(ing_df.to_string())

#Ahora tenemos un marco de datos de ingredientes y sus recuentos totales en todas las recetas.
#Ordenemos este marco de datos en orden descendente.

ing_df.sort_values(["count"], ascending=False, inplace=True)
ing_df.reset_index(inplace=True, drop=True)

print(ing_df)

#Sin embargo, hay un problema ya que hay aproximadamente 40.000 recetas estadounidenses
#lo que significa que los datos están sesgados hacia los ingredientes estadounidenses

#Por lo tanto, creemos un perfil para cada cocina, intentemos descubrir 
# qué ingredientes suelen utilizar los chinos y qué es la comida canadiense por ejemplo;

cocinas = recetas.groupby("cocina").mean()
cocinas.head()

#Imprimamos el perfil de cada cocina mostrando los cuatro ingredientes principales de cada cocina

num_ingredientes = 4

def print_top_ingredientes(row):
    print(row.name.upper())
    row_sorted = row.sort_values(ascending=False)*100
    top_ingredientes = list(row_sorted.index.values)[0:num_ingredientes]
    row_sorted = list(row_sorted)[0:num_ingredientes]

    for ind, ingredientes in enumerate(top_ingredientes):
        print("%s (%d%%)" % (ingredientes, row_sorted[ind]), end=' ')
    print("\n")

create_cocinas_profiles = cocinas.apply(print_top_ingredientes, axis=1)














