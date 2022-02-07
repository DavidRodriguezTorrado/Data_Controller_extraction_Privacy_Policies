import subprocess
import pandas as pd
import spacy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score

# Keywords con las que vamos a identificar si hay un controlador definido o no.
bag_of_words = ['“we”', '“our”', '“us”', '“ we”', '“ our”', '“ us”', '©', 'all rights reserved', '“company”', '(hereinafter',
                '( hereinafter' '\'we\'', '\'us\'', '\'our\'', '\"we\"', '\"us\"',
                '\"our\"', '‘we’', '‘us’', '‘our’', '"us"', '"our"', '"we"', '"services"', '“services”', '\'services\'',
                '"service"', '“service”', '\'service\'', '"App"', '“title”', '"title"', '\'title\'',
                '"apps"', '“apps”', '\'apps\'', '"app"', '“app”', '\'app\'', '"app/s"', '“app/s”', '\'app/s\'', '"site"', '“site”', '\'site\'',
                '"sites"', '“sites”', '\'sites\'', '"user"', '“user”', '\'user\'', '"users"', '“users”', '\'users\''
                '"privacy policy"', '“privacy policy”', '\'privacy policy\'', '"(collectively"', '“(collectively”', '\'(collectively\'',
                '(collectively'
                '("policy")', '(“policy”)', '(\'policy\')', 'data controller is', 'at no cost and is intended for use',
                'is the data controller', #'"agreement"', '“agreement”', '\'agreement\'',
                '« we »', '« our »', '« us »', '«we»', '«our»', '«us»', '"product"', '“product”', '\'product\''
                '« we »', '« our »', '« us »', '“ we”', '“ us”', '“ our”']

# Implementacion de re por si en el futuro queremos guardar lo contenido entre comillas de un texto.
# print(re.findall(r'“(.+?)”', frase))

# Evaluacion métricas funcionamiento del bag of words
def get_eval_metrics(real_Y, predicted_Y, positive_class=1, negative_class=0):
    metrics = {'precision': precision_score(real_Y, predicted_Y, pos_label=positive_class, average='binary'),
               'recall': recall_score(real_Y, predicted_Y, pos_label=positive_class, average='binary'),
               'f1-score': f1_score(real_Y, predicted_Y, pos_label=positive_class, average='binary'),
               'NVP': precision_score(real_Y, predicted_Y, pos_label=negative_class, average='binary'),
               'specificity': recall_score(real_Y, predicted_Y, pos_label=negative_class, average='binary'),
               'f1-score-negative': f1_score(real_Y, predicted_Y, pos_label=negative_class, average='binary'),
               'accuracy': balanced_accuracy_score(real_Y, predicted_Y)}
    tn, fp, fn, tp = confusion_matrix(real_Y, predicted_Y, labels=[positive_class, negative_class]).ravel()
    metrics['conf_matrix'] = [tn, fp, fn, tp]
    return metrics

def data_controller_classification_from_txt(name, policy_text):
    # Creacion de representan las columnas del dataframe que devuelve la funcion
    # Lista que contiene el nombre de la politica (puede ser un apk o una politica de un dominio) que se esta analizando
    final_name_list = []
    # Lista que contiene los parrafos finales quitando los que son demasiado cortos
    final_paragraph_list = []
    # Lista que contiene True y False acorde a cada parrafo de la lista anterior
    paragraph_controller_list = []
    # Lista que contiene una lista de tokens encontrados para cada parrafo
    tokens_found = []

    # Creacion del dataframe que contendra los resultados
    column_names = ['name', 'paragraph', 'contains_controller', 'token_list']
    df = pd.DataFrame(columns=column_names)

    # Lista que contiene los parrafos separados
    pre_paragraphs_list = policy_text.split('\n\n')

    # Preprocesado de los parrafos. Quitamos aquellos que tengan menos de 5 caracteres
    paragraphs = []
    for paragraph in pre_paragraphs_list:
        if len(paragraph) > 5:
            paragraphs.append(paragraph)

    # Recorremos los parrafos en busca de las keywords
    for paragraph in paragraphs:
        final_name_list.append(name)
        final_paragraph_list.append(paragraph)
        # print(element, '\n')

        paragraph_controller = False
        # Lista de tokens encontrados en cada parrafo
        token_list = []
        for word_to_check in bag_of_words:
            # Pasamos a minuscula el texto del parrafo para buscar las keywords
            if word_to_check in paragraph.lower():
                token_list.append(word_to_check)
                paragraph_controller = True
                # print('Se ha localizado un parrafo que puede contener controlador [{}]: \n'.format(word_to_check),
                #      paragraph, '\n')

        # Se guarda la lista de tokens y si ese parrafo se etiqueta como controlador
        if paragraph_controller:
            tokens_found.append(token_list)
        else:
            tokens_found.append([])
        paragraph_controller_list.append(paragraph_controller)

    # Rellenar las columnas del dataframe con los valores
    df['name'] = final_name_list
    df['paragraph'] = final_paragraph_list
    df['contains_controller'] = paragraph_controller_list
    df['token_list'] = tokens_found

    return df


# Las politicas estan en una carpeta cuyo path se pasa al siguiente metodo
def data_controller_classification_from_folder(path):

    # Necesitamos buscar todas las politicas
    output = subprocess.check_output(['ls', path])

    policies_list = output.decode('utf-8').split('\n')

    # Creacion del dataframe que contendra los resultados
    column_names = ['apk', 'paragraph', 'contains_controller', 'token_list']
    df = pd.DataFrame(columns=column_names)

    # Creacion de las listas que formaran cada columna del dataframe
    final_apk_list = []
    final_paragraph_list = []
    paragraph_controller_list = []
    tokens_found = []

    # Recorremos todas las politicas que queremos analizar
    for policy in policies_list:
        apk = policy[:len(policy) - 4]
        try:
            f = open(path+'/{}'.format(policy), 'r')
        except:
            continue

        pre_paragraphs_list = f.read().split('\n\n')

        # Preprocesado de los parrafos. Quitamos aquellos que tengan menos de 5 caracteres
        paragraphs = []
        for paragraph in pre_paragraphs_list:
            if len(paragraph) > 5:
                paragraphs.append(paragraph)

        # Recorremos los parrafos en busca de las keywords
        for paragraph in paragraphs:
            final_apk_list.append(apk)
            final_paragraph_list.append(paragraph)
            # print(element, '\n')

            paragraph_controller = False
            # Lista de tokens encontrados en cada parrafo
            token_list = []
            for word_to_check in bag_of_words:
                # Pasamos a minuscula el texto del parrafo para buscar las keywords
                if word_to_check in paragraph.lower():
                    token_list.append(word_to_check)
                    paragraph_controller = True

            # Se guarda la lista de tokens y si ese parrafo se etiqueta como controlador
            if paragraph_controller:
                tokens_found.append(token_list)
            else:
                tokens_found.append([])
            paragraph_controller_list.append(paragraph_controller)

    # Rellenar las columnas del dataframe con los valores
    df['apk'] = final_apk_list
    df['paragraph'] = final_paragraph_list
    df['contains_controller'] = paragraph_controller_list
    df['token_list'] = tokens_found

    return df


def data_controller_spacy_extraction(df):
    nlp = spacy.load("en_core_web_trf")

    spacy_list = []

    for i in range(len(df)):

        if df['contains_controller'].iloc[i].__eq__(False):
            spacy_list.append([])
            continue

        # Lista que guarda las organizaciones que encuentra spacy en cada parrafo
        entities = []

        # Vamos a analizar los parrafos que se refieren al copyright,
        # para analizar solo la frase que contiene este copyright
        sentence_list = df['paragraph'].iloc[i].splitlines()
        doc = nlp(df['paragraph'].iloc[i])
        found_copyright = False

        for sentence in sentence_list:
            start_sentence = df['paragraph'].iloc[i].find(sentence)
            end_sentence = start_sentence + len(sentence)
            if '©' in sentence:
                for ent in doc.ents:
                    if ent.start_char >= start_sentence and ent.end_char <= end_sentence:
                        if ent.label_ == 'ORG':
                            entities.append(ent.text)
                found_copyright = True
                spacy_list.append(entities)
                break
            if 'all rights reserved' in sentence:
                for ent in doc.ents:
                    if ent.start_char >= start_sentence and ent.end_char <= end_sentence:
                        if ent.label_ == 'ORG':
                            entities.append(ent.text)
                found_copyright = True
                spacy_list.append(entities)
                break
        if found_copyright:
            continue

        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities.append(ent.text)
        spacy_list.append(entities)

    df['spacy_results'] = spacy_list
    return df
