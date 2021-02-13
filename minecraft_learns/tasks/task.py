"""
    Defines a task to be done
    
    Written By: Kathryn Lecha
    Date: 2021-01-26
"""



class Task:
    """
    Generic Abstract Task Class all tasks inherit from
    """
    def __init__(self):
        self.X_columns = None
        self.y_column = None

    def read_data(self, location):
        """
        read in the data selected as a dataframe
        """
        # self.data = Data()
        pass
