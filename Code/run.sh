

python process_data.py raw_data/train5500.txt raw_data/test.txt -type

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u conv_net_sentence.py -static -word2vec | tee output/static_6_output.txt
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u conv_net_sentence.py -nonstatic -word2vec | tee output/nonstatic_6_output.txt

python process_data.py raw_data/train5500.txt raw_data/test.txt -fine

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u conv_net_sentence.py -static -word2vec | tee output/static_50_output.txt
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u conv_net_sentence.py -nonstatic -word2vec | tee output/nonstatic_50_output.txt

