""" Simple example on how to log scalars and images to tensorboard without tensor ops. """
""" https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514 """
__author__ = "Michael Gygli"

import tensorflow as tf
import numpy as np
from StringIO import StringIO

class NonTensorTensorboardLogger(object):
    """Logging in tensorboard without tensorflow ops.
    Example:
    >>> import numpy as np
    >>> log_dir = '/tmp/test'
    >>> logger = Logger(log_dir)  # or if u already have a tf.summary.FileWriter(log_dir)
    >>> logger = Logger(tf.summary.FileWriter(log_dir))
    >>> for i in range(1000):
    >>>    logger.log_histogram('test_hist',np.random.rand(50)*(i+1),i)
    """


    def __init__(self, log_dir=None, summary_writer=None, log_frequency=5):
        """Creates a summary writer logging to log_dir."""
        if summary_writer == None:
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = summary_writer
        self.log_freq = log_frequency
        self.log_count = -1

    def log_now(self, force_log=False):
        """ control logging """
        self.log_count += 1
        if self.log_count % self.log_freq == 0 or force_log:
            return True
        return False

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_single_image(self, tag, img_as_string, height, width, step, force_log=False):
        if not self.log_now(force_log) and not force_log:
            return  # do nothing if its not time to log
        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=img_as_string.getvalue(),
                                   height=height,
                                   width=width)
        # Create a Summary value
        image_summary = tf.Summary.Value(tag=tag, image=img_sum)
        # Create and write Summary
        summary = tf.Summary(value=[image_summary])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_images(self, tag, images, step, force_log=False):
        if not self.log_now(force_log) and not force_log:
            return  # do nothing if its not time to log
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000, force_log=False):
        if not self.log_now(force_log) and not force_log:
            return  # do nothing if its not time to log
        """Logs the histogram of a list/vector of values."""

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        # self.writer.flush()

if __name__ == '__main__':
    log_dir = 'log_dir'
    logger = NonTensorTensorboardLogger(log_dir)  # or if u already have a tf.summary.FileWriter(log_dir)
    logger = NonTensorTensorboardLogger(tf.summary.FileWriter(log_dir))
    for i in range(1000):
       logger.log_histogram('ROC_CURVE', np.random.rand(50)*(i+1), i)