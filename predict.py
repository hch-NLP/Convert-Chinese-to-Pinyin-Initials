import bilsm_crf_model
import process_data
import numpy as np
import os
import os.path
import codecs
model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
def content_predict(text):
    #model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
    strk, length = process_data.process_data(text, vocab)
    model.load_weights('model/crf.h5')
    raw = model.predict(strk)[0][-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [chunk_tags[i] for i in result]
    print('转换后的拼音首字母序列为：',result_tags)    

if __name__ == '__main__':
    while True:
         text=input('请输入需要转换的中文序列：')
         content_predict(text=text)
