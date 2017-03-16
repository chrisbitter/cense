import Decider.decider
import json


#
# Implementation of a Decider making decisions based upon a lookup table
#
class LookupDecider(Decider):

    __lookup_file_path = ""
    __lookup = {}

    #
    # Initiates a lookup table from a given file
    # If no path is given, a standard table will be used
    #
    def __init__(self, lookup_file='NNs/lookup9.json'):
        self.__lookup_file_path = lookup_file
        with open(lookup_file, 'r') as fp:
            self.__lookup = json.load(fp)

    #
    # Persists the lookup table into a json file.
    # If no path is specified the table will be written into the file it was read from
    #
    def persist_lookup_table(self, lookup_file=None):
        file_path = self.__lookup_file_path
        if lookup_file is not None:
            file_path = lookup_file
        with open(file_path, 'w') as fp:
            json.dump(self.__lookup, fp)

    def decide(self, state):
        pass
