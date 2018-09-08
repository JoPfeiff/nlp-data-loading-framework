# nlp-data-loading-framework
We are trying to define a framework for NLP tasks that easily maps any kind of word embedding data set with any kind of text data set. The framework should decrease the amount of additional code needed to work on different NLP tasks so that little amount of work is needed for preprocessing and defining the architecture. 

Currently the framework has the following capabilities:

# DataLoader
data_loader class maps embeddings to text data sets. This code needs to edited to be able to accept different kinds embedding and text data sets. <br/>

This class loads embeddings based on a defined strategy. Currently two versions are implemented:
 - `embedding_loading='top_k'`
 - `embedding_loading='in_dict'` <br/>
`top_k` loads the first `k` embeddings from file, assuming that they are sorted by most frequent on the top<br/>
`in_dict` preloads all embeddings and the selects only those embeddings that occure in the text data set. <br/>

Currently a set of out-of-the-box embedding and text data sets have been implemented. These are:

## Embeddings
  - Any kind of Embeddings in text documents in the structure 
      ```
      <word>\t<float>\t<float>\t...\t<float>\n
      ```
    can be processed using the data_loading parameter embeddings_initial='Path'
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
  

