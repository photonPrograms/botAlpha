import pandas as pd
import gensim
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import pickle

# data frame creation for movies released in and after 2000
df = pd.read_csv("wiki_movie_plots_deduped.csv", sep = ",")
df = df[df["Release Year"] >= 2000]

print(df.head())

# text corpus creation and preprocessing
textCorpus = df["Plot"].values
processedCorpus = preprocess_documents(textCorpus)
taggedCorpus = [TaggedDocument(d, [i]) for i, d in enumerate(processedCorpus)]

# Doc2Vec model creation
model = Doc2Vec(taggedCorpus, dm = 0, vector_size = 200,
        window = 2, min_count = 1, epochs = 10, hs = 1)

# saving the model to a file
fname = "modelDoc2Vec"
pickle.dump(model, open(fname, "wb"))

# testing the model with a sample input
testString = "Sith lord fights Jedi with light saber on star destroyer"
testString = gensim.parsing.preprocessing.preprocess_string(testString)
testDocVector = model.infer_vector(testString)
sims = model.dv.most_similar(positive = [testDocVector])
for s in sims:
    print("{}: {}".format(s[1], df["Title"].iloc[s[0]]))
