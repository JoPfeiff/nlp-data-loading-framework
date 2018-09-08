# nlp-data-loading-framework
We are trying to define a framework for NLP tasks that easily maps any kind of word embedding data set with any kind of text data set. The framework should decrease the amount of additional code needed to work on different NLP tasks so that little amount of work is needed for preprocessing and defining the architecture. 

Currently the framework has the following capabilities:

# Embeddings
  - Any kind of Embeddings in text documents in the structure 
      ```
      <word>\t<float>\t<float>\t...\t<float>\n
      ```
    can be processed using the data_loading parameter embeddings_initial='Path'
  - pre-implemented word embeddings are:
      - Fast-Text: \\
          https://fasttext.cc/docs/en/english-vectors.html
          - embeddings_initial='FastText-Crawl'
          - embeddings_initial='FastText-Wiki'
      - Glove-Embeddings:
          https://nlp.stanford.edu/projects/glove/
          - embeddings_initial='Glove-Twitter-25'
          - embeddings_initial='Glove-Twitter-50'
          - embeddings_initial='Glove-Twitter-100'
          - embeddings_initial='Glove-Twitter-200'
          - embeddings_initial='Glove-Common-42B-300'
          - embeddings_initial='Glove-Common-840B-300'
          - embeddings_initial='Glove-Wiki-50'
          - embeddings_initial='Glove-Wiki-100'
          - embeddings_initial='Glove-Wiki-200'
          - embeddings_initial='Glove-Wiki-300'
       - Lear-Embeddings:
          - embeddings_initial='Lear'
       - Polyglot-Embeddings:
          http://bit.ly/19bSoAS
          - embeddings_initial='Polyglot'
        

