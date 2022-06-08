from collections import OrderedDict

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params
        self.visdom = None


    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError


    def track(self, image, info: dict = None) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError


    def visdom_draw_tracking(self, image, box):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        elif isinstance (box, list):
            box = box
        else:
            box = (box,)
        # self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        self.visdom.register((image, *box), 'Tracking', 1, '跟踪过程')
