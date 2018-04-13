class BoundingBox(object):
    def __init__(self, bounding_box_vals):
        self.x, self.y, self.w, self.h = bounding_box_vals
        self.size = self.w * self.h

    def is_inside(self, other):
        return self.x >= other.x and\
               self.y >= other.y and\
               self.x + self.w <= other.x + other.w and\
               self.y + self.h <= other.y + other.h

    @staticmethod
    def combine(b1, b2):
        x = min(b1.x, b2.x)
        y = min(b1.y, b2.y)
        w = max(b1.x + b1.w, b2.x + b2.w) - x
        h = max(b1.y + b1.h, b2.y + b2.h) - y
        return BoundingBox((x, y, w, h))
