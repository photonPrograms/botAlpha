# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

"""
class ActionHelloWorld(Action):

     def name(self) -> Text:
         return "action_hello_world"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         dispatcher.utter_message(text="Hello World!")

         return []
"""

import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.parsing.preprocessing import preprocess_string
import pickle


model = pickle.load(open("../modelDoc2Vec", 'rb'))

df = pd.read_csv("../wiki_movie_plots_deduped.csv", sep = ",",
        usecols = ["Release Year", "Title", "Plot"])
df = df[df["Release Year"] >= 2000]


class ActionMovieSearch(Action):

    def name(self) -> Text:
        return "action_movie_search"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ramble = tracker.latest_message["text"]
        newDoc = preprocess_string(ramble)
        testDocVector = model.infer_vector(newDoc)
        similar = model.dv.most_similar(positive = [testDocVector])
        movies = [df["Title"].iloc[s[0]] for s in similar[:5]]

        output = "Movies matching the plot: {}".format(movies)
        output = output.replace("[", "").replace("]", "")

        dispatcher.utter_message(text = output)
        return []

class ActionAddition(Action):

     def name(self) -> Text:
         return "action_addition"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         num1 = float(tracker.get_slot("num1"))
         num2 = float(tracker.get_slot("num2"))
         ans = num1 + num2
         output = "BEEP...BEEP...{}".format(ans)

         dispatcher.utter_message(text = output)

         return []

class ActionMultiplication(Action):

     def name(self) -> Text:
         return "action_multiplication"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         num1 = float(tracker.get_slot("num1"))
         num2 = float(tracker.get_slot("num2"))
         ans = num1 * num2
         output = "BEEP...BEEP...{}".format(ans)

         dispatcher.utter_message(text = output)
         return []
