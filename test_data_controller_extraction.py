from data_controller_extraction import data_controller_classification_from_txt, data_controller_spacy_extraction

f = open('./validation_109_bog_policies/air.com.aceviral.mutantfightingcup2.txt', 'r')

name = 'air.com.aceviral.mutantfightingcup2'

df = data_controller_classification_from_txt(name, f.read())
df = data_controller_spacy_extraction(df)
print(df)
df.to_csv('./test_data_controller_from_text.csv', index=False)