# nlp-data-loading-framework
We are trying to define a framework for NLP tasks that easily maps any kind of word embedding data set with any kind of text data set. The framework should decrease the amount of additional code needed to work on different NLP tasks. <br/>
We have found that for many NLP

Currently the framework has the following capabilities:

# DataLoader
data_loader class maps embeddings to text data sets. This code needs to edited to be able to accept different kinds embedding and text data sets. <br/>
In order to combine Fast-Text embeddings with the SNLI data set we can call the data_loader by:
```
dl = DataLoader(data_set='SNLI', embedding_loading='in_dict', embeddings_initial='FastText-Wiki', 
                embedding_params={}, K_embeddings=float('inf'))
gen = dl.get_generator(data_set='train', batch_size=64, drop_last=True)
data , batch = gen.next()
```
The generator loops through the defined `data_set` once. So for each epoch, a new generator has to be called. `drop_last=True` defines that if the last batch is not full, it is not passed. <br/>

This class loads embeddings based on a defined strategy. Currently two versions are implemented:
 - `embedding_loading='top_k'`
 - `embedding_loading='in_dict'` <br/>
`top_k` loads the first `k` embeddings from file, assuming that they are sorted by most frequent on the top. If all embeddings should be loaded set `K_embeddings=float('inf')` <br/>
`in_dict` preloads all embeddings and the selects only those embeddings that occure in the text data set. <br/>

The class also gives the possibility to store all the loaded data on disc in a pickle file and load it again into the object. this can be done by <br/>

`dl.load()` <br/> 
`dl.dump()` <br/> 

However, these functions only dump what has currently been loaded into the object. To load everything at the start and then dump it to file call <br/>

`dl.get_all_and_dump()` <br/>
This function also automatically bucketizes all the sentences based on the defined bucketizing strategy. <br/>

Currently a set of out-of-the-box embedding and text data sets have been implemented. These are:

## Embeddings
  - Any kind of Embeddings in text documents in the structure 
      ```
      <word>\t<float>\t<float>\t...\t<float>\n
      ```
    can be processed using the data_loading parameter embeddings_initial='Path'. The parameters, e.g. where the embedding data can be found is passed in a dictionary: <br/>
    `embedding_params = {'path':'../data/embeddings/bow2.words', 'name':'bow2'}`
    
  - pre-implemented word embeddings are:
      - Fast-Text: <br/>
          https://fasttext.cc/docs/en/english-vectors.html
          - `embeddings_initial='FastText-Crawl'`
          - `embeddings_initial='FastText-Wiki'`
      - Glove-Embeddings:<br/>
          https://nlp.stanford.edu/projects/glove/
          - `embeddings_initial='Glove-Twitter-25'`
          - `embeddings_initial='Glove-Twitter-50'`
          - `embeddings_initial='Glove-Twitter-100'`
          - `embeddings_initial='Glove-Twitter-200'`
          - `embeddings_initial='Glove-Common-42B-300'`
          - `embeddings_initial='Glove-Common-840B-300'`
          - `embeddings_initial='Glove-Wiki-50'`
          - `embeddings_initial='Glove-Wiki-100'`
          - `embeddings_initial='Glove-Wiki-200'`
          - `embeddings_initial='Glove-Wiki-300'`
       - Lear-Embeddings: <br/>
          - `embeddings_initial='Lear'`
       - Polyglot-Embeddings: <br/>
          http://bit.ly/19bSoAS
          - `embeddings_initial='Polyglot'`
 
## Text Data Sets
The text data set implements a bucketized loading structure. That means, that sentences are bucketized based on their length (conditioned on words in the dictionary) and stored in memory. <br/>
A generator is callable that loops through each of the data points randomly by first sampling a bucket, and then sampling from each bucket. <br/>
Out-of-the-box text data sets are:
 - SNLI data set <br/>
   `data_set='SNLI'` <br/>
 https://nlp.stanford.edu/projects/snli/snli_1.0.zip <br/>
 - Billion Word Benchmark data set <br/>
   `data_set='BillionWords'` <br/>
 
 
 
 
 
 
 
 
 
 
 
 
 

