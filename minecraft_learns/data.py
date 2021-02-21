"""
    Reads in JSON file and converts it to a Dataframe for futher use.

    Written By: Nathan Nesbitt
    Date: 2020-11-08
"""

from json import load
from pandas import read_json, read_csv

from .errors import NoDataStored

import os


class Data:
    def __init__(self, location):
        """
        Simply initializes an object that is an open file that can be
        written to.

        @param location: String path to a directory on the system that the
            user can access
        @param filename: String filename that the user can access.
        """
        self.location = location

        # Tries to open the file
        if not os.path.exists(self.location):
            raise FileNotFoundError("Data file doesn't exist")

        if ".jsonl" in location:
            self.load_json_lines(location)
        elif ".json" in location:
            self.load_json(location)
        elif ".csv" in location:
            self.load_csv(location)
        else:
            message = "" + location + " is not a valid dataformat"
            raise NoDataStored(message)

    def load_json(self, location):
        self.df = read_json(location)

    def load_csv(self, location):
        self.df = read_csv(location)

    def load_json_lines(self, location):
        """
        load the data as json lines
        """
        with open(self.location) as f:
            data = f.read()
        try:
            self.df = read_json('[%s]' % ','.join(data.splitlines()))
        except KeyError:
            message = "" + location + " is empty"
            raise NoDataStored(message)

    def delete_file(self):
        """ Deletes the file """
        try:
            os.remove(self.location)
        except PermissionError:
            raise PermissionError("Error Deleting File. No Permissions.")
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
