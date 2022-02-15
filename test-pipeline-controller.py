from pipeline_controller_extraction import pipeline_get_controller

f = open('./test_top_100_policies/gov.irs.txt', 'r')
policy = f.read()

probable_controller, probable_list_controller, df = pipeline_get_controller('gov.irs', policy)

print('Probable controller: ')
print(probable_controller)

print('Probable list controller: ')
print(probable_list_controller)

#df.to_csv('./extraction_policies/final-test.csv', index=False)