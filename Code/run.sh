

python process_data.py raw_data/train5500.txt raw_data/test.txt -type

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence.py -static -word2vec #> output/output.txt


