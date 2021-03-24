"""
    Reads in JSON file and converts it to a Dataframe for futher use.

    Written By: Nathan Nesbitt, Kathryn Lecha
    Edit Date: 2021-02-20
"""

from json import load
from pandas import read_json, read_csv

from .errors import NoDataStored

import os


class Data:
    def __init__(self, filename):
        """
        Simply initializes an object that is an open file that can be
        written to.

        @param filename: String path to a file on the system that the
            user can access
        """
        self.filename = filename
        self.load_data()

    def load_data(self):
        """
        load the data by filetype
        """
        # Tries to open the file
        if not os.path.exists(self.filename):
            raise FileNotFoundError("Data file doesn't exist")

        if ".jsonl" in self.filename:
            self.load_json_lines(self.filename)
        elif ".json" in self.filename:
            self.load_json(self.filename)
        elif ".csv" in self.filename:
            self.load_csv(self.filename)
        else:
            message = (
                "" + self.filename + " is not a supported dataformat. " +
                "The following datatypes are supported: [.csv, .json, .jsonl]"
            )
            raise NoDataStored(message)

    def load_json(self, filename):
        self.df = read_json(filename)

    def load_csv(self, filename):
        self.df = read_csv(filename)

    def load_json_lines(self, filename):
        """
        load the data as json lines
        """
        # load the data
        data = ""
        with open(self.filename) as f:
            data = f.read()

        # convert to dataframe
        try:
            self.df = read_json('[%s]' % ','.join(data.splitlines()))
        except KeyError:
            message = "" + filename + " is empty. There is no data to read."
            raise NoDataStored(message)

    def delete_file(self):
        """ Deletes the file """
        try:
            os.remove(self.filename)
        except PermissionError:
            raise PermissionError(
                "Error Deleting File. No Permission to delete."
            )
        except OSError:
            raise OSError("Error Deleting File.")

    def get_data(self):
        """
        Gets the data from the object and returns it.

        @returns Pandas Dataframe object
        """
        return self.df

    def print_data(self):
        """ Prints the head of the data (default 5). """
        self.df.head()

    def print_types(self):
        """ Prints the types of the dataframe stored. """
        self.df.dtypes
