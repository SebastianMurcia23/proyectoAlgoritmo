from collections import defaultdict
import bibtexparser
import pandas as pd
import re
import matplotlib.pyplot as plt
import tqdm
from wordcloud import WordCloud
def menu():
    while True:
        print("\n--- Menú de Opciones ---")
        print("1. TimSort")
        print("2. Comb Sort")
        print("3. Selection Sort")
        print("4. Tree Sort")
        print("5. Pigeonhole Sort")
        print("6. BucketSort")
        print("7. QuickSort")
        print("8. HeapSort")
        print("9. Bitonic Sort")
        print("10. Gnome Sort")
        print("11. Binary Insertion Sort")
        print("12. RadixSort")
        
        opcion = input("Elige una opción: ")

        if opcion == "1":
           # Funciones de ordenamiento (Timsort)
            def ordenarInsercion(lista, inicio, fin, clave):
                for i in range(inicio + 1, fin + 1):
                    actual = lista[i]
                    j = i - 1
                    while j >= inicio and clave(lista[j]) > clave(actual):
                        lista[j + 1] = lista[j]
                        j -= 1
                    lista[j + 1] = actual

            def mezclarSegmentos(lista, inicio, mitad, fin, clave):
                izquierda = lista[inicio:mitad + 1]
                derecha = lista[mitad + 1:fin + 1]
                i = j = 0
                k = inicio

                while i < len(izquierda) and j < len(derecha):
                    if clave(izquierda[i]) <= clave(derecha[j]):
                        lista[k] = izquierda[i]
                        i += 1
                    else:
                        lista[k] = derecha[j]
                        j += 1
                    k += 1

                while i < len(izquierda):
                    lista[k] = izquierda[i]
                    i += 1
                    k += 1

                while j < len(derecha):
                    lista[k] = derecha[j]
                    j += 1
                    k += 1

            def ordenarTimsort(lista, tam_sublista, clave):
                total = len(lista)
                for inicio in range(0, total, tam_sublista):
                    fin = min(inicio + tam_sublista - 1, total - 1)
                    ordenarInsercion(lista, inicio, fin, clave)

                tam_actual = tam_sublista
                while tam_actual < total:
                    for inicio in range(0, total, 2 * tam_actual):
                        mitad = min(total - 1, inicio + tam_actual - 1)
                        fin = min(inicio + 2 * tam_actual - 1, total - 1)
                        if mitad < fin:
                            mezclarSegmentos(lista, inicio, mitad, fin, clave)
                    tam_actual *= 2
                
            # Cargar archivo bibtex
            ruta_archivo_bibtex = 'C:/Users/ASUS/Downloads/base de datos.bib'
            with open(ruta_archivo_bibtex, 'r', encoding='utf-8') as archivo:
                base_datos_bibtex = bibtexparser.load(archivo)

            lista_referencias = base_datos_bibtex.entries

            # Organizar datos, separando autores
            datos_referencias = []
            for referencia in lista_referencias:
                autores = referencia.get('author', 'Desconocido').split(' and ')
                año = referencia.get('year', 'Sin año')
                título = referencia.get('title', 'Sin título')
                journal = referencia.get('journal', 'Sin journal')
                publisher = referencia.get('publisher', 'Sin publisher')
                tipo = referencia.get('type', 'Otro')
                doi = referencia.get('doi', 'Sin DOI')
                source = referencia.get('source', 'Sin source')
                abstract = referencia.get('abstract', 'Sin abstract')

                for autor in autores:
                    datos_referencias.append({
                        'Autor': autor.strip(),
                        'Año': año,
                        'Título': título,
                        'Journal': journal,
                        'Publisher': publisher,
                        'Tipo': tipo,
                        'DOI': doi,
                        'Source': source,
                        'Publisher': publisher,
                        'Abstract': abstract
                    })

            # Limpiar años
            def limpiarAños(datos_referencias):
                for ref in datos_referencias:
                    año = ref['Año']
                    año_limpio = re.findall(r'\d+', año)
                    ref['Año'] = int(año_limpio[0]) if año_limpio else 0

            limpiarAños(datos_referencias)

            # Detectar base de datos
            def detectar_base_datos(referencia):
                url = referencia.get('url', '').lower()
                publisher = referencia.get('publisher', '').lower()
                journal = referencia.get('journal', '').lower()
                booktitle = referencia.get('booktitle', '').lower()

                if "sciencedirect.com" in url or "sciencedirect" in journal:
                    return "ScienceDirect"
                elif "routledge" in publisher or "francis" in journal:
                    return "Francis"
                elif "ieee" in url or "ieee" in booktitle or "ieee" in publisher:
                    return "IEEE"
                elif "sagepub.com" in url or "sage" in publisher:
                    return "Sage"
                elif "scopus.com" in url or "scopus" in journal or "conference" in referencia.get('type', '').lower():
                    return "Scopus"
                else:
                    return "Otra"

            # Aplicar la función de detección de base de datos
            for referencia in datos_referencias:
                referencia['Base de Datos'] = detectar_base_datos(referencia)

            # Ordenar por autor y año
            ordenarTimsort(datos_referencias, 32, clave=lambda ref: (ref['Autor'], ref['Año']))
            df_referencias = pd.DataFrame(datos_referencias)
            def graficar_referencias_por_año(df):
                df_filtrado = df[df['Año'] > 0]
                conteo = df_filtrado['Año'].value_counts().sort_index()
                plt.figure(figsize=(10, 6))
                conteo.plot(kind='bar', color='skyblue')
                plt.title('Cantidad de Referencias por Año')
                plt.xlabel('Año')
                plt.ylabel('Número de Referencias')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

            def graficar_autores_mas_citados(df):
                autores_mas_citados = df['Autor'].value_counts().head(15)
                plt.figure(figsize=(12, 8))
                autores_mas_citados.plot(kind='bar', color='orange')
                plt.title('15 Autores Más Citados')
                plt.xlabel('Autor')
                plt.ylabel('Número de Citaciones')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

            def graficar_tipo_producto_por_identificador(lista_referencias):
                identificadores = [ref.get('ENTRYTYPE', 'Otro') for ref in lista_referencias]
                conteo_identificadores = pd.Series(identificadores).value_counts()

                plt.figure(figsize=(8, 6))
                conteo_identificadores.plot(kind='bar', color='purple')
                plt.title('Cantidad de Productos por Tipo (basado en @)')
                plt.xlabel('Tipo de Producto (@)')
                plt.ylabel('Cantidad')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

            def graficar_referencias_por_journal(df):
                conteo_journals = df['Journal'].value_counts().head(15)
                plt.figure(figsize=(14, 10))
                conteo_journals.plot(kind='bar', color='teal')
                plt.title('15 Journals con Más Referencias')
                plt.xlabel('Journal')
                plt.ylabel('Número de Referencias')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

            def graficar_referencias_por_publisher(df):
                conteo_publishers = df['Publisher'].value_counts().head(15)
                plt.figure(figsize=(14, 10))
                conteo_publishers.plot(kind='bar', color='darkcyan')
                plt.title('15 Publishers con Más Referencias')
                plt.xlabel('Publisher')
                plt.ylabel('Número de Referencias')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

            def graficar_referencias_por_source(df):
                # Contamos la cantidad de cada valor en la columna 'Source'
                conteo_source = df['Source'].value_counts()
                
                # Generamos el gráfico
                plt.figure(figsize=(10, 6))
                conteo_source.plot(kind='bar', color='purple')
                plt.title('Cantidad de Referencias por Source')
                plt.xlabel('Source')
                plt.ylabel('Cantidad')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()


            def graficar_articulos_mas_citados(df):
                articulos_mas_citados = df['Título'].value_counts().head(15)
                
                # Aumentar aún más el tamaño de la figura
                plt.figure(figsize=(20, 12))
                
                # Crear el gráfico de barras
                articulos_mas_citados.plot(kind='bar', color='green')
                
                # Ajustar títulos y etiquetas con tamaños de fuente más grandes
                plt.title('15 Artículos Más Citados', fontsize=24)
                plt.xlabel('Artículo', fontsize=18)
                plt.ylabel('Número de Citaciones', fontsize=18)
                
                # Ajustar tamaño y rotación de etiquetas de los ejes
                plt.xticks(rotation=45, ha='right', fontsize=14)
                plt.yticks(fontsize=14)
                
                plt.tight_layout()  # Asegurar que el contenido se ajuste bien dentro de la figura
                plt.show()



            categorias = {
                "Habilidades": [
                    "Abstraction", "Algorithm", "Algorithmic thinking", "Coding", "Collaboration",
                    "Cooperation", "Creativity", "Critical thinking", "Debug", "Decomposition", 
                    "Evaluation", "Generalization", "Logic", "Logical thinking", "Modularity", 
                    "Patterns recognition", "Problem solving", "Programming", "Representation", 
                    "Reuse", "Simulation"
                ],
                "Conceptos Computacionales": [
                    "Conditionals", "Control structures", "Directions", "Events", "Functions", 
                    "Loops", "Modular structure", "Parallelism", "Sequences", "Software/hardware", 
                    "Variables"
                ],
                "Actitudes": [
                    "Emotional", "Engagement", "Motivation", "Perceptions", "Persistence", 
                    "Self-efficacy", "Self-perceived"
                ],
                "Propiedades psicométricas": [
                    "Classical Test Theory - CTT", "Confirmatory Factor Analysis - CFA", 
                    "Exploratory Factor Analysis - EFA", "Item Response Theory (IRT) - IRT", 
                    "Reliability", "Structural Equation Model - SEM", "Validity"
                ],
                "Herramienta de evaluación": [
                    "Beginners Computational Thinking test - BCTt", "Coding Attitudes Survey - ESCAS", 
                    "Collaborative Computing Observation Instrument", "Competent Computational Thinking test - cCTt", 
                    "Computational thinking skills test - CTST", "Computational concepts", 
                    "Computational Thinking Assessment for Chinese Elementary Students - CTA-CES", 
                    "Computational Thinking Challenge - CTC", "Computational Thinking Levels Scale - CTLS", 
                    "Computational Thinking Scale - CTS", "Computational Thinking Skill Levels Scale - CTS", 
                    "Computational Thinking Test - CTt", "Computational Thinking Test", 
                    "Computational Thinking Test for Elementary School Students - CTT-ES", 
                    "Computational Thinking Test for Lower Primary - CTtLP", 
                    "Computational thinking-skill tasks on numbers and arithmetic", 
                    "Computerized Adaptive Programming Concepts Test - CAPCT", "CT Scale - CTS", 
                    "Elementary Student Coding Attitudes Survey - ESCAS", "General self-efficacy scale", 
                    "ICT competency test", "Instrument of computational identity", "KBIT fluid intelligence subtest", 
                    "Mastery of computational concepts Test and an Algorithmic Test", 
                    "Multidimensional 21st Century Skills Scale", "Self-efficacy scale", 
                    "STEM learning attitude scale - STEM-LAS", "The computational thinking scale"
                ],
                "Diseño de investigación": [
                    "No experimental", "Experimental", "Longitudinal research", "Mixed methods", 
                    "Post-test", "Pre-test", "Quasi-experiments"
                ],
                "Nivel de escolaridad": [
                    "Upper elementary education - Upper elementary school", "Primary school - Primary education - Elementary school", 
                    "Early childhood education – Kindergarten - Preschool", "Secondary school - Secondary education", 
                    "High school - Higher education", "University – College"
                ],
                "Medio": [
                    "Block programming", "Mobile application", "Pair programming", "Plugged activities", 
                    "Programming", "Robotics", "Spreadsheet", "STEM", "Unplugged activities"
                ],
                "Estrategia": [
                    "Construct-by-self mind mapping - CBS-MM", "Construct-on-scaffold mind mapping - COS-MM", 
                    "Design-based learning - CTDBL", "Design-based learning - DBL", 
                    "Evidence-centred design approach", "Gamification", "Reverse engineering pedagogy - REP", 
                    "Technology-enhanced learning", "Collaborative learning", "Cooperative learning", 
                    "Flipped classroom", "Game-based learning", "Inquiry-based learning", 
                    "Personalized learning", "Problem-based learning", "Project-based learning", 
                    "Universal design for learning"
                ],
                "Herramienta": [
                    "Alice", "Arduino", "Scratch", "ScratchJr", "Blockly Games", "Code.org", 
                    "Codecombat", "CSUnplugged", "Robot Turtles", "Hello Ruby", "Kodable", 
                    "LightbotJr", "KIBO robots", "BEE BOT", "CUBETTO", "Minecraft", "Agent Sheets", 
                    "Mimo", "Py– Learn", "SpaceChem"
                ]
            }

            def obtener_diccionario_sinonimos(categorias):
                dic_sinonimos = {}
                for categoria, palabras in categorias.items():
                    for item in palabras:
                        terminos = item.split(" - ")  # Dividir los sinónimos usando el guion
                        for termino in terminos:
                            dic_sinonimos[termino.strip().lower()] = terminos[0].strip()
                return dic_sinonimos

            dic_sinonimos = obtener_diccionario_sinonimos(categorias)

            # Función para contar palabras unificadas en abstracts
            def contar_palabras_abstracts(lista_referencias, dic_sinonimos):
                conteo_palabras = defaultdict(int)
                
                for referencia in lista_referencias:
                    abstract = referencia.get('abstract', '').lower()
                    if abstract:
                        for palabra, unificada in dic_sinonimos.items():
                            conteo_palabras[unificada] += len(re.findall(r'\b' + re.escape(palabra) + r'\b', abstract))
                            
                return conteo_palabras

            # Realizar el conteo
            conteo_palabras = contar_palabras_abstracts(lista_referencias, dic_sinonimos)



            # Calcular frecuencias y mostrar los resultados
            df_conteo = pd.DataFrame(list(conteo_palabras.items()), columns=['Palabra Unificada', 'Frecuencia'])

            def graficar_frecuencia_palabras(df_conteo):
                df_conteo = df_conteo.sort_values(by='Frecuencia', ascending=False).head(20)
                plt.figure(figsize=(12, 8))
                plt.barh(df_conteo['Palabra Unificada'], df_conteo['Frecuencia'], color='skyblue')
                plt.xlabel("Frecuencia")
                plt.ylabel("Palabra Unificada")
                plt.title("Frecuencia de Palabras en Abstracts")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()


            def generar_nube_palabras(lista_referencias, categorias):
                # Combinar todos los abstracts
                texto_completo = ' '.join([ref.get('abstract', '') for ref in lista_referencias])
                
                # Crear un conjunto de todas las palabras clave de las categorías
                palabras_clave = set()
                for categoria, palabras in categorias.items():
                    for item in palabras:
                        # Separar los términos que están conectados con guiones
                        terminos = item.split(" - ")
                        for termino in terminos:
                            palabras_clave.add(termino.lower())
                
                # Crear un diccionario para contar la frecuencia de las palabras clave
                frecuencias = defaultdict(int)
                
                # Procesar el texto y contar frecuencias solo de las palabras clave
                for palabra in palabras_clave:
                    # Usar expresión regular para encontrar coincidencias exactas de palabras
                    coincidencias = len(re.findall(r'\b' + re.escape(palabra) + r'\b', texto_completo.lower()))
                    if coincidencias > 0:
                        frecuencias[palabra] = coincidencias
                
                # Configurar la nube de palabras
                wordcloud = WordCloud(
                    width=1600,
                    height=800,
                    background_color='white',
                    max_words=100,
                    min_font_size=10,
                    max_font_size=150,
                    colormap='viridis'
                )
                
                # Generar la nube de palabras
                wordcloud.generate_from_frequencies(frecuencias)
                
                # Mostrar la nube de palabras
                plt.figure(figsize=(20, 10))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Nube de Palabras Clave en Abstracts', fontsize=20, pad=20)
                plt.tight_layout(pad=0)
                plt.show()

            def crear_grafo_journals(df):
                # Obtener los 10 journals con mayor cantidad de artículos
                top_journals = df['Journal'].value_counts().head(10).index.tolist()

                # Crear el grafo
                G = nx.Graph()

                # Agregar los journals como nodos
                for journal in top_journals:
                    G.add_node(journal, tipo='journal')

                # Agregar los artículos más citados por cada journal
                for journal in top_journals:
                    articulos_journal = df[df['Journal'] == journal].sort_values('Citaciones', ascending=False).head(15)
                    for _, row in articulos_journal.iterrows():
                        articulo = row['Título']
                        país = row['País']
                        citaciones = row['Citaciones']
                        G.add_node(articulo, tipo='articulo', país=país, citaciones=citaciones)
                        G.add_edge(journal, articulo)

                return G

            def mostrar_estadisticas(G):
                # Número de nodos y aristas
                print(f"Número de nodos: {G.number_of_nodes()}")
                print(f"Número de aristas: {G.number_of_edges()}")

                # Distribución de grados
                grados = list(dict(G.degree()).values())
                plt.figure(figsize=(8, 6))
                plt.hist(grados, bins=range(max(grados)+1), edgecolor='black')
                plt.xlabel('Grado')
                plt.ylabel('Frecuencia')
                plt.title('Distribución de Grados')
                plt.show()

                # Centralidad de grado
                centrality = nx.degree_centrality(G)
                sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                print("\nTop 10 nodos por centralidad de grado:")
                for node, value in sorted_centrality[:10]:
                    print(f"{node}: {value:.3f}")

                # Centralidad de intermediación
                betweenness = nx.betweenness_centrality(G)
                sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
                print("\nTop 10 nodos por centralidad de intermediación:")
                for node, value in sorted_betweenness[:10]:
                    print(f"{node}: {value:.3f}")

            def graficar_grafo(G):
                # Crear un diccionario de colores para los nodos
                colores = {'journal': 'blue', 'articulo': 'orange'}

                # Posicionar los nodos
                pos = nx.spring_layout(G, k=0.5, seed=42)

                # Dibujar el grafo
                plt.figure(figsize=(12, 8))
                nx.draw_networkx_nodes(G, pos, node_color=[colores[G.nodes[n]['tipo']] for n in G.nodes()], node_size=300)
                nx.draw_networkx_edges(G, pos)
                nx.draw_networkx_labels(G, pos, font_size=8)

                # Agregar leyenda
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', label='Journal', markerfacecolor=colores['journal'], markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Artículo', markerfacecolor=colores['articulo'], markersize=10)
                ]
                plt.legend(handles=legend_elements, loc='upper left')

                plt.axis('off')
                plt.title('Relación entre Journals y Artículos Más Citados')
                plt.show()
                            

            # Añadir la opción en el menú para ver este análisis
            while True:
                print("\nSelecciona una opción para el gráfico:")
                print("1 - Gráfico de cantidad de referencias por año")
                print("2 - Gráfico de 15 autores más citados")
                print("3 - Gráfico de cantidad de productos por tipo (basado en @)")
                print("4 - Gráfico de 15 journals con más referencias")
                print("5 - Gráfico de 15 publishers con más referencias")
                print("6 - Gráfico de cantidad de referencias por base de datos (solo Sage, Francis, ScienceDirect, Scopus, IEEE)")
                print("7 - Gráfico de 15 artículos más citados")
                print("8 - Análisis de frecuencia de palabras clave en abstracts por categoría")
                print("9 - Nube de palabras de los abstracts")  
                print("0 - Salir")
                
                opcion = input("Ingresa el número de la opción deseada: ")

                if opcion == '1':
                    graficar_referencias_por_año(df_referencias)
                elif opcion == '2':
                    graficar_autores_mas_citados(df_referencias)
                elif opcion == '3':
                    graficar_tipo_producto_por_identificador(lista_referencias)
                elif opcion == '4':
                    graficar_referencias_por_journal(df_referencias)
                elif opcion == '5':
                    graficar_referencias_por_publisher(df_referencias)
                elif opcion == '6':
                    graficar_referencias_por_source(df_referencias)
                elif opcion == '7':
                    graficar_articulos_mas_citados(df_referencias)
                elif opcion == '8':
                   graficar_frecuencia_palabras(df_conteo)
                elif opcion == '9':
                   generar_nube_palabras(lista_referencias, categorias) 
                elif opcion == '10':
                    G = crear_grafo_journals(df_referencias)
                    mostrar_estadisticas(G)
                elif opcion == '0':
                    print("Programa terminado.")
                    break
                else:
                    print("Opción inválida. Intenta de nuevo.")


            # Guardar en un archivo Excel
            ruta_archivo_excel = 'C:/Users/ASUS/Downloads/timsort.xlsx'
            df_referencias.to_excel(ruta_archivo_excel, index=False)

            print(f"Archivo Excel generado en: {ruta_archivo_excel}")


        else:
            print("Opción no válida, intenta nuevamente.")

# Ejecutar el menú
menu()
