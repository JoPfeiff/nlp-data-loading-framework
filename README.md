# Framework for NLP Text Data
We are trying to define a framework for NLP tasks that easily maps any kind of word embedding data set with any kind of text data set. The framework should decrease the amount of additional code needed to work on different NLP tasks. <br/>
We have found that for many NLP tasks similar preprocessing steps are needed. <br/>
This entails 
  - tokenizing the text 
  - replacing words with embeddings (pretrained or newly learnt)
  - bucketizing sentences based on their length
  - padding sentences to a specific length
  - replacing unseen words with `<UNK>`
  - creating a generator that loops through the sentences <br/>
 
We therefore want to create a framework that provides these common functionalities out-of-the-box to be able to focus on the core task of the project faster. <br/>

Currently the framework has the following capabilities:

# DataLoader
This is the main class to-be-called and can be found in `data_loading/data_loader.py` <br/>
The `DataLoader` class maps embeddings to text data sets. This code needs to edited to be able to accept different kinds embedding and text data sets. <br/>
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

The current tokenizer is based on spaCy.io and can easily be replaced in `data_loading/data_utils.py` in the function `tokenize()`

Currently a set of out-of-the-box embedding and text data sets have been implemented. These are:

## Embeddings
The core class is `Embeddings` which can be found in `embeddings/embeddings.py`. However, this should only be used as the super class for the specialized embeddings. New embedding inherit this class (e.g. `class FastTextEmbeddings(Embeddings)` in  `embeddings/fasttext_embeddings.py`). Only if the embeddings are to be initialized randomly, the core `Embeddings` class is to be called. <br/>

A generic path-based embedding class is implemented that can process any kind of Embeddings stored as text documents in the structure 
      ```
      <word>\t<float>\t<float>\t...\t<float>\n
      ```
This object is called if the parameter `embeddings_initial='Path'` is called when creating the `data_loading` object. The parameters, e.g. where the embedding data is stored, is passed as a dictionary: <br/>
    `embedding_params = {'path':'../data/embeddings/bow2.words', 'name':'bow2'}` <br/>
    
A set of pre-implemented word embeddings are:
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
      
### Implementing new Embedding classes
To implement a new Embedding class, this should inherit the class `Embeddings` which can be found in `embeddings/embeddings.py`. This has all the basic functionality implemented thats needed for most embedding data. The new Embedding class only needs two functions which are data set dependet. These are: <br/>
 - `load_top_k(self, K, preload=False)` <br/>
      This loads the top_k embeddings from file with the assumption that the embeddings are ordered based on their frequency. The following functionalities for implementing this function are important:
      - Adding the term should be done using <br/>
        `self.add_term(term, preload=preload)`
      - If special embeddings (`<UNK>`, `<PAD>`, `<START>`, `<END>`) need to be added, this is to be done using <br/>
        `special_embeddings = self.add_special_embeddings(len(embeddings[0]), preload=preload)`
      - The function should return the embeddings as a `np.array()` <br/>
        `return np.array(embeddings)`     
      
 - `get_name(self)` <br/>
    This should return the name of the embeddings e.g. `'FastText-Wiki'`

The new functionality needs to be added to `DataLoader` in  `data_loading/data_loader.py`. The defined object needs to be callable in this object using a name. Two new lines need to be added to:
```
            if self.embeddings_initial in FASTTEXT_NAMES:
                self.embedding = FastTextEmbeddings(self.embeddings_initial)
            elif self.embeddings_initial in POLYGLOT_NAMES:
                self.embedding = PolyglotEmbeddings()
            elif self.embeddings_initial in LEAR_NAMES:
                self.embedding = LearEmbeddings()
            elif self.embeddings_initial in GLOVE_NAMES:
                self.embedding = GloveEmbeddings(self.embeddings_initial)
            elif self.embeddings_initial == "Path":
                self.embedding = PathEmbeddings(self.embedding_params)
            else:
                raise Exception("No valid embedding was set")
```

For reference please look at `embeddings/fasttext_embeddings.py`
 
## Text Data Sets
The text data set implements a bucketized loading structure. That means, that sentences are bucketized based on their length (conditioned on words in the dictionary) and stored in memory. <br/>
A generator is callable that loops through each of the data points randomly by first sampling a bucket, and then sampling from each bucket. <br/>
Out-of-the-box text data sets are:
 - SNLI data set <br/>
   `data_set='SNLI'` <br/>
 https://nlp.stanford.edu/projects/snli/snli_1.0.zip <br/>
 - Billion Word Benchmark data set <br/>
   `data_set='BillionWords'` <br/>
 
### Implementing New Text Data Sets

New text data sets are to inherit the class `TextData` that can be found in `text/text_data.py`. This class has out-of-the-box functionalities like loading and storing data, but also a generator function is defined here, which samples from bucketized sentences. <br/>
The new class needs two functions:
 - loading() <br/>
    The data is loaded to memory and extracted into sentences. <br/>
    Important factors of this function:
     - data is to be stored in `self.data_set` <br/>
        This variable has been initialized in `TextData` and is a dictionary.
     - raw data is to be stored in `self.data_set['train']`, `self.data_set['dev']` and `self.data_set['test']` 
     - each data point is to be stored as a dictionary element ({}) and stored in the list (e.g. `self.data_set['train'] = []`) 
     - parsing a sentence is to be done using 
     ```
     elem['sentence1_positions'] = self.embeddings.encode_sentence(elem['sentence1'], 
                                                                   initialize=initialize_term, 
                                                                   count_up=True) 
     ```
        
 
 
 
 
 
 
 
 
 
 
 

