"""
    This is a file containing all of the required errors for the Minecraft
    Education API.

    Written By: Kathryn Lecha
    Date: 2021-02-12
"""


class IncorrectFlow(Exception):
    """
    Error class that defines all flow errors during user tasks
    For example if the user tries to predict without fitting, this error
    can be thrown
    """

    def __init__(self, method, message="The method cannot be executed yet"):
        self.method = method
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.method} -> {self.message}"


class UnProcessedData(IncorrectFlow):
    """
    Exception which indicates that the data has not been processed
    """

    def __init__(self, entered_type):
        super().__init__(entered_type, "The data has not been processed yet")


class ModelNotFit(IncorrectFlow):
    """
    Exception which indicates that the model has not been fit yet
    """

    def __init__(self, entered_type):
        super().__init__(entered_type, "The model has not been fit yet")
