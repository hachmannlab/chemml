import itertools


FORK = u'\u251c'
LAST = u'\u2514'
VERTICAL = u'\u2502'
HORIZONTAL = u'\u2500'
NEWLINE = u'\u23ce'


def _format_newlines(prefix, formatted_node):
    """
    Convert newlines into U+23EC characters, followed by an actual newline and
    then a tree prefix so as to position the remaining text under the previous
    line.
    """
    replacement = u''.join([
        NEWLINE,
        u'\n',
        prefix])
    return formatted_node.replace(u'\n', replacement)


def _format_tree(node, format_node, get_children, prefix=u''):
    children = list(get_children(node))
    next_prefix = u''.join([prefix, VERTICAL, u'   '])
    for child in children[:-1]:
        yield u''.join([prefix,
                        FORK,
                        HORIZONTAL,
                        HORIZONTAL,
                        u' ',
                        _format_newlines(next_prefix,
                                         format_node(child))])
        for result in _format_tree(child,
                                   format_node,
                                   get_children,
                                   next_prefix):
            yield result
    if children:
        last_prefix = u''.join([prefix, u'    '])
        yield u''.join([prefix,
                        LAST,
                        HORIZONTAL,
                        HORIZONTAL,
                        u' ',
                        _format_newlines(last_prefix,
                                         format_node(children[-1]))])
        for result in _format_tree(children[-1],
                                   format_node,
                                   get_children,
                                   last_prefix):
            yield result


def format_tree(node, format_node, get_children):
    lines = itertools.chain(
        [format_node(node)],
        _format_tree(node, format_node, get_children),
        [u''],
    )
    return u'\n'.join(lines)


def print_tree(*args, **kwargs):
    print(format_tree(*args, **kwargs))


def test():
    from operator import itemgetter
    tree = (
        'foo', [
            ('bar', [
                ('a', []),
                ('b', []),
            ]),
            ('baz', []),
            ('qux', [
                ('c\nd', []),
            ]),
        ],
    )
    print format_tree(
        tree, format_node=itemgetter(0), get_children=itemgetter(1))