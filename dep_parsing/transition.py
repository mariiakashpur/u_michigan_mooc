class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def node_has_head(conf, node):
        for triple in conf.arcs:
            if triple[-1] == node:
                return True

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer[0]

        if not conf.buffer or not conf.stack or idx_wi == 0 or Transition.node_has_head(conf, idx_wi):
            return -1

        conf.arcs.append((idx_wj, relation, idx_wi))
        conf.stack.pop(-1)

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        idx_wi = conf.stack[-1]
        if not conf.buffer or not conf.stack or not Transition.node_has_head(conf, idx_wi):
            return -1
        conf.stack.pop(-1)

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1
        idx_wj = conf.buffer.pop(0)
        conf.stack.append(idx_wj)