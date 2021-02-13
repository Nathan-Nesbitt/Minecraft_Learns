"""
    This is a file containing all of the required errors for the Minecraft
    Education API.

    Written By: Kathryn Lecha
    Date: 2021-02-12
"""


class TypeNotFound(Exception):
    """
        Base Exception Class for all issues regarding user inputted types
        that can be handled in the game. For example if you try to create
        an event handler or "hook" for a non-existing event, for example
        "on-bounce" this can be thrown.
    """
    def __init__(
        self, entered_type,
        message="The type of Event or Command was not valid Minecraft Command"
    ):
        self.entered_type = entered_type
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.entered_type} -> {self.message}'


class TaskNotFound(TypeNotFound):
    """
        Exception which indicates that the task has not been implemented
    """
    def __init__(self, entered_type):
        super().__init__(
            entered_type,
            "The Entered Task does not exist"
        )


class ModelNotFound(TypeNotFound):
    """
        Exception which indicates that the model has not been implemented
    """
    def __init__(self, entered_type):
        super().__init__(
            entered_type,
            "The entered Model was not found"
        )


class HasNoFunction(Exception):
    """
        Exception which indicates that there is no specified event handler
        for the user's Event. This means that the user has forgotten to
        indicate what they would like done when some event happens.
    """

    def __init__(
        self,
        message="No Function has been chosen to be run when the event occurs."
    ):
        self.message = message
        super().__init__(self.message)


class MessageAlreadyCalled(Exception):
    """
        Exception for when you have tried to append the current message
        call to the user twice, this simply warns the user that they shouldn't
        be repeating the same function multiple times as it is inefficient and
        the function should simply be changed.
    """

    def __init__(
        self,
        message="You have tried to add 2 instances of the same Event or "
        + "Command to the queue. Make sure you aren't duplicating code"
    ):
        self.message = message
        super().__init__(self.message)
