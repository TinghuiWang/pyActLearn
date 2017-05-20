import math
import matplotlib.lines as mlines
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms


class LabeledLine(mlines.Line2D):
    """Draw a line with label in the middle on top of the line
    """
    def __init__(self, *args, **kwargs):
        self.text = mtext.Text(0, 0, '', verticalalignment='center', horizontalalignment='center',
                               color='b')
        super().__init__(*args, **kwargs)
        self.text.set_text(self.get_label())
        self.text.set_zorder(self.get_zorder() + 1)
        self.text.set_alpha(self.get_alpha())

    def set_figure(self, figure):
        self.text.set_figure(figure)
        super().set_figure(figure)

    def set_axes(self, axes):
        self.text.set_axes(axes)
        super().set_axes(axes)

    def set_transform(self, transform):
        texttrans = transform
        self.text.set_transform(texttrans)
        super().set_transform(transform)

    def set_data(self, x, y):
        if len(x) > 1:
            text_x = .5 * (x[0] + x[1])
            text_y = .5 * (y[0] + y[1])
            text_rot = - math.degrees(math.atan((y[1] - y[0]) / (x[1] - x[0])))
            self.text.set_position((text_x, text_y))
            self.text.set_rotation(text_rot)
        super().set_data(x, y)

    def draw(self, renderer):
        # draw my label at the end of the line with 2 pixel offset
        super().draw(renderer)
        self.text.draw(renderer)
