import math
from PySide6 import QtCore, QtWidgets
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class NetworkGraphWidget(QtWidgets.QWidget):
    nodeClicked = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.node_size = 600

        # 创建一个图形对象
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # 创建一个垂直布局
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 初始化图
        self.G = None
        self.pos = {}

        # 创建并绘制初始图
        self.draw_graph()

        # 记录上一次点击的节点
        self.last_node = None

        # 设置画布接受鼠标事件
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def set_graph(self, g=None):
        self.G = g
        if self.G is not None:
            self.pos = nx.shell_layout(self.G)
        else:
            self.pos = {}

        self.draw_graph()

    def draw_graph(self):
        # 清除之前的绘图
        self.ax.clear()

        if self.G is None:
            g = nx.DiGraph()
            nx.draw(g)
            self.canvas.draw()
            return

        # 绘制图
        color_list = ["skyblue"] * len(self.G.nodes)
        nx.draw(
            self.G,
            self.pos,
            ax=self.ax,
            with_labels=True,
            node_color=color_list,
            edge_color="gray",
            node_size=self.node_size,
            font_size=6,
        )

        # 刷新画布
        self.canvas.draw()

    def on_click(self, event):
        if self.G is None:
            return

        if event.inaxes is None:
            return

        # 获取点击位置
        x, y = event.xdata, event.ydata
        xx, yy = self.ax.transData.transform((x, y))
        node_radius = math.sqrt(self.node_size) / 2

        # 找到最近的节点
        closest_node = None
        min_distance = float("inf")
        for node, (n_x, n_y) in self.pos.items():
            n_xx, n_yy = self.ax.transData.transform((n_x, n_y))
            distance = math.sqrt((n_xx - xx) ** 2 + (n_yy - yy) ** 2)
            if distance <= node_radius and distance < min_distance:
                min_distance = distance
                closest_node = node

        if closest_node is not None:
            # emit signal
            self.nodeClicked.emit(str(closest_node))

        # 重新计算布局并绘制图
        self.pos = nx.shell_layout(self.G)
        self.draw_graph()
