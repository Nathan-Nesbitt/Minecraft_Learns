"""
    Reads in JSON file and converts it to a Dataframe for futher use.

    Written By: Nathan Nesbitt
    Date: 2020-11-08
"""


from pandas import read_csv

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
        self.df = read_csv(self.location, sep=",", header=0)

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
