import copy
from itertools import groupby, combinations, chain

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from string import ascii_uppercase
import matplotlib as mpl

DEBUG = False

shortcut_symbols = {'V': '\\vee', '^': '\\wedge', '_': '\\neg', '(': '(', ')': ')', ',': ','}
literal_symbols = dict(zip(ascii_uppercase[:], list(map(lambda x: '\\mathcal{' + x + '}', ascii_uppercase[:]))))
operator_symbols = {'implication': '\\rightarrow', '(': '(', ')': ')', 'and': '\\wedge', 'or': '\\vee', 'not': '\\neg',
                    ',': ','}
special_symbols = {'turnstile': '\\models'}
table_symbols = {**literal_symbols, **operator_symbols}


def split(_list, delim):
    return [list(group) for k, group in groupby(_list, lambda s: s == delim) if not k]


def find_dict_key(_dict, value):
    return list(_dict.keys())[list(_dict.values()).index(value)]


def power_set(iterable):
    s = list(iterable)
    return set(map(frozenset, chain.from_iterable(combinations(s, r) for r in range(len(s)+1))))


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]


class AST:
    parent = None
    child_names = []

    def __setattr__(self, key, value):
        if value is not None and key in self.child_names:
            value.parent = self

        self.__dict__[key] = value

    @property
    def children(self):
        return [self.__dict__.get(key) for key in self.child_names]

    def get_child_name(self, child):
        for name in self.child_names:
            if self.__dict__.get(name) == child:
                return name

    def set_child(self, child, value):
        if value is not None:
            value.parent = self
        self.__dict__[self.get_child_name(child)] = value

    def get_all_from_inside(self, _filter):
        res = []
        while True:
            op = self.get_deepest(lambda o: _filter(o) and o not in res)
            if op is None:
                break
            res.append(op)
            yield op

    def get_deepest(self, _filter):
        for child in self.children:
            res = child.get_deepest(_filter)
            if res is not None:
                return res
        if _filter(self):
            return self
        else:
            return None

    def get_highest(self, _filter):
        if _filter(self):
            return self

        for child in self.children:
            res = child.get_deepest(_filter)
            if res is not None:
                return res

        return None

    def _replace(self, old, new):
        if old in self.children:
            self.set_child(old, new)

        for child in self.children:
            assert isinstance(child, AST)
            child._replace(old, new)

    def _delete(self, target):
        if target in self.children:
            if type(self) == Head:
                self.child = None
            elif type(self) == BinOp:
                self.parent.set_child(self, self.left if target == self.right else self.right)
            elif type(self) == UnaryOp:
                assert isinstance(self.parent, AST)
                self.parent._delete(self)

            return

        for child in self.children:
            assert isinstance(child, AST)
            child._delete(target)

    @staticmethod
    def are_equal(tree1, tree2):
        if type(tree1) != type(tree2):
            return False

        if type(tree1) == Literal:
            return tree1.token == tree2.token
        else:
            res = False

            if not res:
                for child1, child2 in zip(tree1.children, tree2.children):
                    res = AST.are_equal(child1, child2)
            if not res:
                for child1, child2 in zip(reversed(tree1.children), tree2.children):
                    res = AST.are_equal(child1, child2)

            return res


class Head(AST):
    child_names = ['child']

    def __init__(self, child):
        self.child = child
        self.child.parent = self

    def __str__(self):
        return str(self.child)

    def delete(self, target):
        assert isinstance(self.child, AST)
        self.child._delete(target)

    def replace(self, old, new):
        assert isinstance(self, AST)
        self._replace(old, new)


class BinOp(AST):
    child_names = ['left', 'right']

    def __init__(self, left, op, right):
        self.left = left
        self.left.parent = self
        self.op = op
        self.right = right
        self.right.parent = self

    if DEBUG:
        def __repr__(self):
            return f'({repr(self.left)} {self.op} {repr(self.right)})'


class UnaryOp(AST):
    child_names = ['expr']

    def __init__(self, op, expr):
        self.op = op
        self.expr = expr
        self.expr.parent = self

    if DEBUG:
        def __repr__(self):
            return f'{self.op}({repr(self.expr)})'


class Literal(AST):
    def __init__(self, token):
        self.token = token

    if DEBUG:
        def __repr__(self):
            return self.token


class NodeVisitor:
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor, metaclass=Singleton):
    def visit_BinOp(self, node):
        if node.op in ['and', 'or', 'implication']:
            if type(node.parent) == Head or (type(node.parent) == BinOp and node.op == node.parent.op):
                return self.visit(node.left) + [table_symbols[node.op]] + self.visit(node.right)
            else:
                return [table_symbols['(']] + \
                       self.visit(node.left) + [table_symbols[node.op]] + self.visit(node.right) + \
                       [table_symbols[')']]

    @staticmethod
    def visit_Literal(node):
        return [node.token]

    def visit_UnaryOp(self, node):
        if node.op == 'not':
            return [table_symbols['not']] + self.visit(node.expr)

    def simplify(self, head):
        # A -> B = -A v B
        for bin_op in head.get_all_from_inside(lambda bo: type(bo) == BinOp and bo.op == 'implication'):
            bin_op.op = 'or'
            bin_op.left = UnaryOp('not', bin_op.left)

        if DEBUG:
            Text().add_line(['1)'] + self.interpret(head))

        # -(A v -B) = -A ^ B; -(A ^ -B) = -A v B
        for unary_op in head.get_all_from_inside(lambda uo: type(uo) == UnaryOp and uo.op == 'not' and
                                                 type(uo.expr) == BinOp):
            if unary_op.expr.op == 'or':
                unary_op.expr.op = 'and'
            elif unary_op.expr.op == 'and':
                unary_op.expr.op = 'or'

            unary_op.expr.left = UnaryOp('not', unary_op.expr.left)
            unary_op.expr.right = UnaryOp('not', unary_op.expr.right)

            head.replace(unary_op, unary_op.expr)

        if DEBUG:
            Text().add_line(['2)'] + self.interpret(head))

        # --A = A
        for unary_op in head.get_all_from_inside(lambda uo: type(uo) == UnaryOp and uo.op == 'not' and
                                                 type(uo.expr) == UnaryOp and uo.expr.op == 'not'):
            head.replace(unary_op, unary_op.expr.expr)

        if DEBUG:
            Text().add_line(['3)'] + self.interpret(head))

        # A v (B ^ C) = (A v B) ^ (A v C)
        for bin_op in head.get_all_from_inside(lambda bo: type(bo) == BinOp and bo.op == 'or' and
                                               type(bo.left) == BinOp and bo.left.op == 'and'):
            head.replace(
                bin_op,
                BinOp(
                    BinOp(bin_op.left.left, 'or', bin_op.right),
                    'and',
                    BinOp(bin_op.left.right, 'or', bin_op.right)
                )
            )
        for bin_op in head.get_all_from_inside(lambda bo: type(bo) == BinOp and bo.op == 'or' and
                                               type(bo.right) == BinOp and bo.right.op == 'and'):
            head.replace(
                bin_op,
                BinOp(
                    BinOp(bin_op.left, 'or', bin_op.right.left),
                    'and',
                    BinOp(bin_op.left, 'or', bin_op.right.right)
                )
            )

        if DEBUG:
            Text().add_line(['4)'] + self.interpret(head))

        # (A v A) ^ (B v C) = A ^ (B v C)
        for disjunct in head.get_all_from_inside(
            lambda o: type(o) == BinOp and o.op == 'or'
            and
            ((type(o.parent) == BinOp and o.parent.op == 'and') or type(o.parent) == Head)
        ):
            sub_head = Head(disjunct)
            for literal1 in sub_head.get_all_from_inside(lambda o: type(o) == Literal):
                for literal2 in sub_head.get_all_from_inside(
                    lambda o: o != literal1 and type(o) == Literal and o.token == literal1.token and
                    type(o.parent) == type(literal1.parent)
                ):
                    sub_head.delete(literal2)

            head.replace(disjunct, sub_head.child)

        if DEBUG:
            Text().add_line(['5)'] + self.interpret(head))

        # (A v B) ^ (A v B) = A v B
        for op1 in head.get_all_from_inside(lambda o: type(o.parent) == BinOp and o.parent.op == 'and'):
            for op2 in head.get_all_from_inside(lambda o: o != op1 and type(o.parent) == BinOp and
                                                o.parent.op == 'and'):
                if AST.are_equal(op1, op2):
                    head.delete(op2)

        if DEBUG:
            Text().add_line(['6)'] + self.interpret(head))

        # (A v B v -A) ^ (C v D) = B ^ (C v D)
        for disjunct in head.get_all_from_inside(
            lambda o:
            (type(o.parent) == Head and type(o) != BinOp) or
            (type(o.parent) == Head and type(o) == BinOp and o.op != 'and') or
            (type(o.parent) == BinOp and o.parent.op == 'and')
        ):
            sub_head = Head(disjunct)
            for literal1 in sub_head.get_all_from_inside(lambda o: type(o) == Literal):
                for literal2 in sub_head.get_all_from_inside(
                    lambda o: o != literal1 and type(o) == Literal and o.token == literal1.token and
                    type(o.parent) != type(literal1.parent)
                ):
                    if literal2 is not None:
                        sub_head.delete(literal1)
                        sub_head.delete(literal2)

                    if sub_head.child is None:
                        break
                if sub_head.child is None:
                    break
            if sub_head.child is None:
                break
            head.replace(disjunct, sub_head.child)

        if DEBUG:
            Text().add_line(['7)'] + self.interpret(head))

    def interpret(self, tree):
        if type(tree) == Head:
            return self.visit(tree.child)
        return self.visit(tree)

    def calc(self):
        Text().clear_lines()
        left_part = split(Text().lines[0], special_symbols['turnstile'])[0]

        conjuncts = ConjunctList()
        line = []
        for expr in split(left_part, ','):
            head = Head(Parser(expr).parse())
            if DEBUG:
                Text().add_line(self.interpret(head))
            self.simplify(head)

            if head.child:
                for op in head.get_all_from_inside(
                    lambda o:
                    (type(o.parent) == Head and type(o) != BinOp) or
                    (type(o.parent) == Head and type(o) == BinOp and o.op != 'and') or
                    (type(o.parent) == BinOp and o.parent.op == 'and')
                ):
                    line += ([','] if line else []) + self.interpret(op)

                    conjuncts.append(DisjunctList())
                    for literal in op.get_all_from_inside(
                        lambda o: (type(o) == Literal and type(o.parent) != UnaryOp) or type(o) == UnaryOp
                    ):
                        conjuncts[-1].append(literal)

        right_part = split(Text().lines[0], special_symbols['turnstile'])[1]
        head = Head(UnaryOp('not', Parser(right_part).parse()))
        if DEBUG:
            Text().add_line(self.interpret(head))
        self.simplify(head)

        if head.child:
            for op in head.get_all_from_inside(
                    lambda o:
                    (type(o.parent) == Head and type(o) != BinOp) or
                    (type(o.parent) == Head and type(o) == BinOp and o.op != 'and') or
                    (type(o.parent) == BinOp and o.parent.op == 'and')
            ):
                line += ([','] if line else []) + self.interpret(op)

                conjuncts.append(DisjunctList())
                for literal in op.get_all_from_inside(
                    lambda o: (type(o) == Literal and type(o.parent) != UnaryOp) or type(o) == UnaryOp
                ):
                    conjuncts[-1].append(literal)

        Text().add_line(Text().add_brackets(line))

        for q in range(len(conjuncts) - 1):
            for w in range(q + 1, len(conjuncts)):
                for k in conjuncts[q]:
                    if type(k) == UnaryOp and k.op == 'not' and k.expr in conjuncts[w]:
                        conjuncts[q].remove(k)
                        conjuncts[w].remove(k.expr)
                        Text().add_line(Text().add_brackets(conjuncts.get_symbols()))
                    if type(k) == Literal and UnaryOp('not', copy.deepcopy(k)) in conjuncts[w]:
                        conjuncts[q].remove(k)
                        conjuncts[w].remove(UnaryOp('not', copy.deepcopy(k)))
                        Text().add_line(Text().add_brackets(conjuncts.get_symbols()))


class ConjunctList(list):
    def get_symbols(self):
        res = []
        not_empty = [conjunct for conjunct in self if conjunct != []]
        for conjunct in not_empty:
            if conjunct != not_empty[0]:
                res += [operator_symbols[',']]
            for disjunct in conjunct:
                if disjunct != conjunct[0]:
                    res += [operator_symbols['or']]
                res += Interpreter().interpret(disjunct)
        if res:
            return res
        else:
            return ['True']


class DisjunctList(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def __contains__(self, item):
        for binar in self:
            if AST.are_equal(item, binar):
                return True
        return False

    def remove(self, item):
        for binar in self:
            if AST.are_equal(item, binar):
                super().remove(binar)


class Parser(metaclass=Singleton):
    def error(self):
        raise Exception('Invalid syntax')

    def __init__(self, symbols):
        self.tokens = iter([find_dict_key(table_symbols, symbol) for symbol in symbols])
        self.current_token = next(self.tokens)

    def eat(self, token):
        if self.current_token == token:
            self.current_token = next(self.tokens, None)
        else:
            self.error()

    def negation(self):
        token = self.current_token
        if token == 'not':
            self.eat('not')
            node = UnaryOp(token, self.negation())
            return node
        elif token in literal_symbols:
            self.eat(token)
            return Literal(token)
        elif token == '(':
            self.eat('(')
            node = self.implicant()
            self.eat(')')
            return node

    def conjunct(self):
        node = self.negation()

        while self.current_token == 'and':
            self.eat('and')
            node = BinOp(left=node, op='and', right=self.negation())

        return node

    def disjunct(self):
        node = self.conjunct()

        while self.current_token == 'or':
            self.eat('or')
            node = BinOp(left=node, op='or', right=self.conjunct())

        return node

    def implicant(self):
        node = self.disjunct()

        while self.current_token == 'implication':
            self.eat('implication')
            node = BinOp(left=node, op='implication', right=self.implicant())

        return node

    def parse(self):
        node = self.implicant()

        if self.current_token is not None:
            self.error()

        return node


mpl.rcParams['toolbar'] = 'None'
for action in ['fullscreen', 'home', 'back', 'forward', 'pan', 'zoom', 'save', 'quit', 'grid', 'yscale', 'xscale',
               'all_axes']:
    plt.rcParams[f'keymap.{action}'] = ''
def_ax = plt.gca()
plt.axis('off')
plt.gcf().canvas.set_window_title('Resolution technique')


# mng = plt.get_current_fig_manager()
# mng.window.state('zoomed')


def get_window_size():
    fig = plt.gcf()
    return fig.get_size_inches() * fig.dpi


def onclick(event):
    if event.key:
        if event.key == 'enter':
            Interpreter().calc()
        elif event.key == 'right':
            Text().move_cursor_right()
        elif event.key == 'left':
            Text().move_cursor_left()
        elif event.key == 'backspace':
            Text().delete_symbol()
        elif event.key in shortcut_symbols:
            Text().add_symbol(shortcut_symbols[event.key])
        elif event.key.islower() and event.key.upper() in literal_symbols:
            Text().add_symbol('\mathcal{' + event.key.upper() + '}')


# noinspection PyUnusedLocal
def onresize(event):
    ButtonTable().redraw()


plt.gcf().canvas.mpl_connect('key_press_event', onclick)
plt.gcf().canvas.mpl_connect('resize_event', onresize)


def partial(f, *args1):
    # noinspection PyUnusedLocal
    def wrapped(*args2):
        return f(*args1)

    return wrapped


def add_dollar(text):
    return f'${text}$'


class Text(metaclass=Singleton):
    cur_pos = 0
    text_plt = None
    symbols = [special_symbols['turnstile']]
    lines = [symbols]

    def __init__(self):
        self.redraw()

    @property
    def lines_with_decor(self):
        res = copy.deepcopy(self.lines)
        res[0].insert(self.cur_pos, '\\mid')
        self.add_brackets(res[0], end_pos=res[0].index(special_symbols['turnstile']))
        return res

    @staticmethod
    def add_brackets(symbols, start_pos=0, end_pos=None):
        res = symbols[:]
        res.insert(start_pos, '\{')
        res.insert(len(res) if not end_pos else end_pos + 1, '\}')
        return res

    def move_cursor_right(self):
        self.cur_pos = (self.cur_pos + 1) % (len(self.lines[0]) + 1)
        self.redraw()

    def move_cursor_left(self):
        self.cur_pos -= 1
        if self.cur_pos == -1:
            self.cur_pos = len(self.lines[0]) + 1
        self.redraw()

    def redraw(self):
        plt.sca(def_ax)
        if self.text_plt:
            self.text_plt.remove()

        lines_with_dollar = list(map(lambda line: add_dollar(' '.join(line)), self.lines_with_decor))

        lines_text = '\n'.join(lines_with_dollar)
        self.text_plt = plt.text(0, 1, lines_text, verticalalignment='top', fontsize=20)
        plt.draw()

    def update_lines(self):
        self.lines = split(self.symbols, '\n')

    def clear_lines(self):
        self.lines = [self.lines[0]]
        self.symbols = self.lines[0]

    def add_symbol(self, symbol, append=False):
        if append:
            self.symbols.append(symbol)
        else:
            self.symbols.insert(self.cur_pos, symbol)
            self.cur_pos += 1

        self.update_lines()
        self.redraw()

    def delete_symbol(self):
        if self.cur_pos > 0 and self.symbols[self.cur_pos - 1] != '\models':
            self.symbols.pop(self.cur_pos - 1)
            self.cur_pos -= 1

            self.update_lines()
            self.redraw()

    def add_line(self, symbols):
        for symbol in ['\n'] + symbols:
            self.add_symbol(symbol, True)
        self.redraw()


# ['\\mathcal{A}', '\\vee',
#  '(', '\\mathcal{B}', '\\wedge', '(', '\\mathcal{C}', '\\wedge', '\\mathcal{D}', ')', ')']:
# ['\\mathcal{A}', '\\rightarrow', '\\mathcal{B}', '\\rightarrow', '(', '\\mathcal{C}', '\\vee', '(',
#  '\\mathcal{F}', '\\wedge', '\\mathcal{G}', ')', '\\rightarrow', '\\mathcal{D}', ')']:
# ['(', '\\mathcal{A}', '\\vee', '\\mathcal{B}', ')', '\\wedge', '(', '\\mathcal{A}', '\\vee', '\\mathcal{B}', ')']:
for test_symbol in \
    ['\\neg', '\\mathcal{A}', '\\vee', '\\mathcal{C}', ',', '\\neg', '\\mathcal{D}', '\\rightarrow',
     '\\mathcal{B}', ',', '\\mathcal{A}', '\\vee', '\\neg', '\\mathcal{B}']:
    Text().add_symbol(test_symbol)

Text().move_cursor_right()

for test_symbol in ['\\mathcal{D}', '\\vee', '\\mathcal{C}']:
    Text().add_symbol(test_symbol)


# Text().add_symbol(symbols_dict['A'])
# Text().add_symbol(symbols_dict['implication'])
# Text().add_symbol(symbols_dict['B'])
# Text().add_symbol(symbols_dict['implication'])
# Text().add_symbol(symbols_dict['C'])
# Text().add_symbol(symbols_dict['implication'])
# Text().add_symbol(symbols_dict['D'])
# Text().cur_pos += 1
# Text().add_symbol(symbols_dict['E'])


class ButtonTable(metaclass=Singleton):
    cell_height = 0.075
    cell_width = 0
    buttons = dict()
    button_axes = dict()

    def redraw(self):
        if self.button_axes:
            for ax in self.button_axes.values():
                plt.delaxes(ax)

        size = get_window_size()
        ratio = size[0] / size[1]
        cell_quad_width = self.cell_height / ratio
        cells_per_line = 1 // cell_quad_width

        self.cell_width = 1 / cells_per_line

        for num, (name, symbol) in enumerate(table_symbols.items()):
            self.button_axes[name] = plt.axes([(num % cells_per_line) * self.cell_width,
                                               (num // cells_per_line) * self.cell_height,
                                               self.cell_width, self.cell_height])
            self.buttons[name] = Button(self.button_axes[name], '')
            self.buttons[name].on_clicked(partial(Text().add_symbol, symbol))
            plt.sca(self.button_axes[name])
            plt.text(0.5, 0.5, add_dollar(symbol), horizontalalignment='center', verticalalignment='center',
                     fontsize=20 * (size[1] / 480))

        last_button_index = len(table_symbols.items())

        self.button_axes['calc'] = plt.axes([(last_button_index % cells_per_line) * self.cell_width,
                                             (last_button_index // cells_per_line) * self.cell_height,
                                             self.cell_width, self.cell_height])
        self.buttons['calc'] = Button(self.button_axes['calc'], 'Calc')
        self.buttons['calc'].on_clicked(
            lambda ignore_arg: Interpreter().calc()
        )


plt.show()
