"""
    This is a file containing all of the required errors for Minecraft_Learns

    Written By: Kathryn Lecha
    Date: 2021-02-12
"""


class IncorrectFlow(Exception):
    """
    Error class that defines all flow errors during user tasks
    For example if the user tries to predict without fitting, this error
    can be thrown
    """

    def __init__(self, message="The method cannot be executed yet"):
        self.message = message
        super().__init__(self.message)


class UnProcessedData(IncorrectFlow):
    """
    Exception which indicates that the data has not been processed
    """

    def __init__(self):
        self.message = "The data must be processed before training"
        super().__init__(self.message)


class ModelNotFit(IncorrectFlow):
    """
    Exception which indicates that the model has not been fit yet
    """

    def __init__(self):
        self.message = "The model must be fit before prediction"
        super().__init__("The model must be fit before prediction")


class NoDataStored(Exception):
    """
    Error class that defines all flow errors during user tasks
    For example if the user tries to predict without fitting, this error
    can be thrown
    """

    def __init__(self, message="No Data has been stored"):
        self.message = message
        super().__init__(self.message)
