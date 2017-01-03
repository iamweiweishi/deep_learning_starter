
steps: 
prepare data set → write the index of the data set into 'csv.txt' → training DNN models

'''
the first step: prepare data set
# baiduImageSpider
baiduSpider, developed on Python3

run in terminal as follows:
>python
>python spiderBaidu.py

You can substitute 'Your key words' at the end of the spiderBaidu.py
'''

'''
an alternative spider of the step: googleSpider
just like the baiduSpider,
run in terminal as follows:
>python
>python spiderGoogle.py

'''
###

'''
the second step: 
write the index of the data set into csv.txt, which is a standard data format in TF. 
You may refer to https://www.tensorflow.org/versions/r0.11/how_tos/reading_data/
(The csv.txt provided here is just an example. You might need to implement this step by yourself)
'''
###

'''
the final step: training DNN models:
run in terminal as follows:
>python
>python DNN.py
the prediction results is simply stored in thePredResults.txt
'''
