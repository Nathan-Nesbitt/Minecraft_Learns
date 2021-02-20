"""
    Reads in JSON file and converts it to a Dataframe for futher use.

    Written By: Nathan Nesbitt
    Date: 2020-11-08
"""

from json import load
from pandas import read_json

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

        json_lines = self._load_data_json_lines().splitlines()
        self.df = read_json('[%s]' % ','.join(json_lines))

    def _load_data_json_lines(self):
        """
        load the data as json lines
        """
        with open(self.location) as f:
            data = f.read()
        return data

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
