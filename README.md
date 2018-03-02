##  Improving Variational Encoder-Decoders in Dialogue Generation

### Description
This repository hosts the improved VEDs(Variational Encoder-Decoders) model for generative dialog modeling as described by Shen and Su et al. 2018

### Creating Datasets
1. Download the DailyDialog Corpus as released by Li, Su and Shen et al. (2017) which can be found : http://yanran.li/dailydialog.html
2. Create the dictionary from the corpus and Serialize the dicitonary and corpus.(we give a demo convert_text2dict.py for creating pkl file)
3. Download Word2Vec trained by GoogleNes: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM.
4. Changing dataproducer.py to generate tfrecord from the serialized corpus text(We use TFRecord for fast and stable training process)

### Model Training
We use Tensorflow1.0 and Python2.7 for convenient.
1. Create a new "Checkpoints"  directories inside it.
2. Change the parameters in main.py according to your GPU memeory size.
3. Read the core code file WS_vhred.py 
4. Change and Run main.py 
### References

    Improving Variational Encoder-Decoders in Dialogue Generation. Xiaoyu Shen, Hui Su, Shuzi Niu, Vera Demberg. 2018. AAAI https://arxiv.org/abs/1802.02032
    The DailyDialog Corpus: DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, Shuzi Niu. 2017. IJCNLP. https://arxiv.org/abs/1710.03957.
