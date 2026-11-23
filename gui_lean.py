"""PyQt6 GUI components for visualizing the Metro reinforcement learning environment."""

from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QMainWindow,
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QToolTip,
    QStatusBar,
    QToolBar,
    QListWidget,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QPointF, QTimer
from PyQt6.QtGui import QPen, QColor, QIcon, QAction, QBrush, QPainter, QPainterPath
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
try:
    from PyQt6.QtOpenGL import QOpenGLFunctions_2_0
except ImportError:  # pragma: no cover - optional dependency on QtOpenGL bindings
    QOpenGLFunctions_2_0 = None
from topologyutils import node_in_segment_percentage
from topology.node_type import NodeType
# from linedata import platforms, line_segments, calc_platform_positions
from linedata import (
    platforms,
    routes,
    calc_platform_positions,
    get_platform_by_id,
    stations,
)
from logger import logger, add_logger_to_gui

class HoverableItem:
    """Mixin class for hoverable graphics items"""
    def __init__(self, tooltip_prefix=""):
        self.tooltip_prefix = tooltip_prefix
        self.setAcceptHoverEvents(True)

    def _show_tooltip(self, event, item):
        """Show tooltip for the item"""
        disrupted_text = ""
        if hasattr(item, 'start_platform') and hasattr(item, 'end_platform'):
            # This is a line segment, check if it's disrupted
            # 使用从父窗口传递过来的 disruption_manager
            if self.parent_window.disruption_manager.is_segment_disrupted(item.id):
                disrupted_text = " [Disrupted]"
        
        QToolTip.showText(
            event.screenPos(), 
            f"{self.tooltip_prefix} Id: {item.id} Weight: {item.weight}{disrupted_text}"
        )

    def _hide_tooltip(self):
        """Hide the tooltip"""
        QToolTip.hideText()

class HoverableGraphicsLineItem(QGraphicsLineItem, HoverableItem):
    """Graphics line item that highlights edges and displays edge metadata on hover."""

    def __init__(self, edge, parent_window, *args, **kwargs): # 1. 添加 parent_window
        super().__init__(*args, **kwargs)
        HoverableItem.__init__(self, "Edge")
        self.edge = edge
        self.parent_window = parent_window # 2. 保存 parent_window 的引用

    def hoverEnterEvent(self, event):
        disruption_info = self.parent_window.disruption_manager.get_segment_disruption_info(self.edge.id) # 3. 使用父窗口的 manager
        color = 'blue'
        width = 2
        if disruption_info:
            if disruption_info['type'] == 'BLOCK':
                color = 'darkred'
                width = 3
            elif disruption_info['type'] == 'SPEED_LIMIT':
                color = 'darkorange'
                width = 3
        
        self.setPen(QPen(QColor(color), width))
        self._show_tooltip(event, self.edge)
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        disruption_info = self.parent_window.disruption_manager.get_segment_disruption_info(self.edge.id) # 4. 使用父窗口的 manager
        color = 'grey'
        width = 2
        if disruption_info:
            if disruption_info['type'] == 'BLOCK':
                color = 'red'
                width = 4
            elif disruption_info['type'] == 'SPEED_LIMIT':
                color = 'orange'
                width = 4

        self.setPen(QPen(QColor(color), width))
        self._hide_tooltip()
        super().hoverLeaveEvent(event)

class HoverableGraphicsEllipseItem(QGraphicsEllipseItem, HoverableItem):
    """Interactive node item that can display a platform context menu when right-clicked."""

    def __init__(self, node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        HoverableItem.__init__(self, "Node")
        self.node = node

    def contextMenuEvent(self, event):
        """Show a context menu that allows switching platform operating modes."""
        if self.scene() is None:
            super().contextMenuEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        for item in self.scene().items(scene_pos):
            if not isinstance(item, HoverableGraphicsEllipseItem):
                continue
            platform = get_platform_by_id(item.node.id)
            if platform is None:
                continue
            self._show_platform_context_menu(event, platform)
            return

        super().contextMenuEvent(event)

    def _show_platform_context_menu(self, event, platform):
        menu = QMenu()
        menu_actions = {
            "default": QAction("Default", menu),
            "stop": QAction("Stop", menu),
            "jump": QAction("Skip", menu),
        }

        for mode, action in menu_actions.items():
            action.setCheckable(True)
            action.setChecked(platform.mode == mode)
            action.triggered.connect(lambda checked, m=mode: self.set_station_mode(platform, m))
            menu.addAction(action)

        menu.exec(event.screenPos().toPoint())

    def set_station_mode(self, platform, mode):
        """Update the platform mode and refresh the brush color for the node."""
        if mode == "default":
            platform.mode = "default"
            self.setBrush(QBrush(QColor('lightblue')))
        elif mode == "stop":
            platform.mode = "stop"
            self.setBrush(QBrush(QColor('yellow')))
        elif mode == "jump":
            platform.mode = "jump"
            self.setBrush(QBrush(QColor('green')))

class OpenGLViewport(QOpenGLWidget):
    """OpenGL-backed viewport that keeps the background color in sync without flicker."""

    def __init__(self, background_color=QColor("white"), parent=None):
        super().__init__(parent)
        self._background_color = QColor(background_color)
        # Avoid Qt trying to fill the background twice which causes flicker.
        self.setAutoFillBackground(False)
        self._gl_functions = None

    def set_background_color(self, color):
        qcolor = QColor(color)
        if qcolor == self._background_color:
            return
        self._background_color = qcolor
        # Force a refresh so the new clear color is applied immediately.
        self.update()

    def initializeGL(self):
        funcs = None
        if QOpenGLFunctions_2_0 is not None:
            try:
                funcs = QOpenGLFunctions_2_0()
                funcs.initializeOpenGLFunctions()
            except Exception:
                funcs = None

        self._gl_functions = funcs
        self._clear_color_buffer()

    def paintGL(self):
        self._clear_color_buffer()

    def _clear_color_buffer(self):
        funcs = self._gl_functions
        if funcs is None:
            return

        try:
            funcs.glClearColor(
                self._background_color.redF(),
                self._background_color.greenF(),
                self._background_color.blueF(),
                1.0,
            )
            funcs.glClear(0x00004000)
        except AttributeError:
            pass


class CustomGraphicsView(QGraphicsView):
    """QGraphicsView configured for smooth panning, zooming, and OpenGL acceleration."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.parent = parent
        background = getattr(parent, "view_bg_color", "#ffffff") if parent else "#ffffff"
        gl_viewport = OpenGLViewport(background, self)
        self.setViewport(gl_viewport)  # Enable hardware acceleration
        gl_viewport.set_background_color(background)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheNone)
        brush = QBrush(QColor(background))
        self.setBackgroundBrush(brush)
        if scene is not None:
            scene.setBackgroundBrush(brush)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(zoom_factor, zoom_factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

@dataclass
class TrainArrivalState:
    """Metadata describing a train arrival during a refresh cycle."""

    just_arrived: bool
    is_at_platform: bool
    alighting_count: int


class MetroWindow(QMainWindow):
    """Main application window that renders topology, timeline, and train metadata."""

    def __init__(self, env, env_index=0):
        super().__init__()
        self.env = env
        self.env_index = env_index
        # 5. 从当前环境实例获取中断管理器
        self.disruption_manager = self.env.world.disruption_manager
        self.train_items = {}
        self.train_label_items = {}
        self.train_passenger_items = {}
        self.station_waiting_items = {}
        self.train_alighting_items = {}
        self.train_previous_positions = {}
        self.alighting_display_timers = {}
        self.segment_items = {}
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        
        # Track all rendered graphics items keyed by domain object
        self.node_items = {}
        self.edge_items = {}
        self.train_items = {}
        self.train_label_items = {}
        self.train_passenger_items = {}
        self.station_waiting_items = {}
        # Track platform graphics items so they can be refreshed safely
        self.platform_items = []
        # Map each train to its alighting label in the topology view
        self.train_alighting_items = {}
        # Timeline traces keyed by train for lightweight updates
        self.timeline_traces = {}
        self.timeline_point_min_dx = 0.01
        # Remember previous positions to detect arrivals at platforms
        self.train_previous_positions = {}
        # Timers used to remove alighting info after a short delay
        self.alighting_display_timers = {}
        self.refresh_timer = None

        # Shared styling
        self.panel_bg_color = "#E5E7EB"
        self.view_bg_color = "#f5f7ff"

        # Continue with UI component initialisation once attributes are ready
        self._init_window()
        self._init_layouts()
        self._init_statusbar()
        self._init_graph()
        self._setup_timers()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self.refresh_trains)
        self.refresh_timer.start()
    def _init_window(self):
        """Initialize window properties"""
        logger.info("Initializing window")
        self.setWindowTitle(f"Metro RL Environment - Instance {self.env_index}")
        self.setGeometry(100 + self.env_index * 50, 100 + self.env_index * 30, 1500, 800)
        self.setWindowIcon(QIcon('logo.ico'))

    def _init_layouts(self):
        """Initialize layout structure"""
        # Create main components
        self.scene = QGraphicsScene()

        self.train_list_widget = self._create_train_list()
        self.edgeList = QListWidget()
        self.edgeList.setStyleSheet(
            f"background-color: {self.panel_bg_color}; border: none;"
        )
        self.nodeList = QListWidget()
        self.nodeList.setStyleSheet(
            f"background-color: {self.panel_bg_color}; border: none;"
        )

        self.timeline_scene = QGraphicsScene()
        self.timeline_view = CustomGraphicsView(self.timeline_scene, self)
        self.timeline_view.setStyleSheet("border: 1px solid grey;")

        self.topology_scene = QGraphicsScene()
        self.topology_view = CustomGraphicsView(self.topology_scene, self)
        self.topology_view.setStyleSheet("border: 1px solid grey;")
        
        # Create passenger flow info panel
        self.flow_info_widget = self._create_flow_info_widget()

        toprow_layout = QHBoxLayout()
        toprow_layout.setSpacing(2)  # Reduce spacing between widgets
        toprow_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        # toprow_layout.addWidget(self.view, stretch=1)
        toprow_layout.addWidget(self.topology_view, stretch=3)
        toprow_layout.addWidget(self.flow_info_widget, stretch=2)

        toprow_widget = QWidget()
        toprow_widget.setLayout(toprow_layout)

        bottomrow_layout = QHBoxLayout()
        bottomrow_layout.setSpacing(2)  # Reduce spacing between widgets
        bottomrow_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        bottomrow_layout.addWidget(self.timeline_view, stretch=3)
        bottomrow_layout.addWidget(self.train_list_widget, stretch=1)

        bottomrow_widget = QWidget()
        bottomrow_widget.setLayout(bottomrow_layout)

        graph_layout = QVBoxLayout()
        graph_layout.setSpacing(0)  # Remove spacing between top and bottom rows
        graph_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        graph_layout.addWidget(toprow_widget, stretch=1)
        graph_layout.addWidget(bottomrow_widget, stretch=1)

        graph_widget = QWidget()
        graph_widget.setLayout(graph_layout)

        layout = QHBoxLayout()
        layout.addWidget(graph_widget)

        layout.setContentsMargins(1, 1, 1, 1)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Populate flow panel with initial data
        self._update_flow_info_panel()

    def _init_statusbar(self):
        """Initialize status bar and labels"""
        logger.info("Initializing status bar")
        statusBar = self.statusBar()

        # Create status labels
        self.status_labels = {
            'time': QLabel("Time: 0"),
            'status': QLabel("Status: Stopped")
        }

        # Add time label to the left
        statusBar.setContentsMargins(5, 0, 5, 2)
        statusBar.addWidget(self.status_labels['time'])
        statusBar.addPermanentWidget(self.status_labels['status'])

    def _init_graph(self):
        """Initialize graph visualization"""
        logger.info("Initializing graph")
        nodes, edges = self.env.nodes, self.env.edges

        # Initialize storage for graph items
        self.node_items = {}
        self.edge_items = {}

        # Initialize storage for graph items
        self.train_items = {}

        # Get the viewport size
        self.timeline_width = self.timeline_view.viewport().width()
        self.timeline_height = self.timeline_view.viewport().height()
        
        self.maxtime = 6 * 3600
        self.timeline_margin = 40  # Margin from edges
        self.timeline_tickwidth = (self.timeline_width - 2 * self.timeline_margin) / self.maxtime
        self.timeline_point_min_dx = max(self.timeline_tickwidth * 0.25, 0.01)
        
        self.platform_horiz_pos = {}

        # Draw edges first (so they appear under nodes)
        self._draw_edges(edges)

        # Then draw nodes
        self._draw_nodes(nodes)

        # draw platform nodes
        self._draw_platform_nodes(platforms)

    def _setup_timers(self):
        """Initializes and starts the timers for UI updates."""

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_trains)
        self.refresh_timer.start(100)
    
    def _draw_edges(self, edges):
        """Draw edges on the graph"""
        pen = QPen(QColor('grey'))
        pen.setWidth(2)
        
        for edge in edges:
            # Calculate edge positions
            start_pos = self._get_node_position(edge.start_node)
            end_pos = self._get_node_position(edge.end_node)
            
            # Create and configure edge item
            line = HoverableGraphicsLineItem(
                edge,
                self,
                start_pos.x(),
                start_pos.y(),
                end_pos.x(),
                end_pos.y(),
            )
            line.setPen(pen)
            
            # Add to scene and store reference
            self.scene.addItem(line)
            self.edge_items[edge.id] = line
            
            # Add to edge list
            self.edgeList.addItem(
                f"Edge {edge.id}: {edge.start_node.weight} -> {edge.end_node.weight}"
            )

    def _draw_nodes(self, nodes):
        """Draw nodes on the graph"""
        logger.info("Drawing nodes")
        node_pen = QPen(QColor('grey'))
        node_brush = QColor('lightblue')

        for node_id, node in nodes.items():
            # Calculate node size based on weight
            node_radius = 5 if node.weight > 1 else 2
            pos = self._get_node_position(node)

            # Create and configure node item
            circle = HoverableGraphicsEllipseItem(
                node,
                pos.x() - node_radius,
                pos.y() - node_radius,
                node_radius * 2,
                node_radius * 2
            )
            circle.setPen(node_pen)
            circle.setBrush(node_brush)

            # Add to scene and store reference
            self.scene.addItem(circle)
            self.node_items[node.id] = circle

            # Add to node list
            self.nodeList.addItem(f"Node {node.id}: {node.weight}")

    def _draw_platform_nodes(self, platforms):
        """Draw platform nodes on the topology canvas"""
        for item in self.platform_items:
            self.topology_scene.removeItem(item)
        self.platform_items.clear()
        # Remove previously drawn waiting passenger labels
        for label in self.station_waiting_items.values():
            self.topology_scene.removeItem(label)
        self.station_waiting_items.clear()

        logger.info("Drawing platform nodes")
        # 原函数返回 {Platform(): (x,y)}，转换为 {platform.id: (x,y)}
        _raw_positions = calc_platform_positions(platforms, 70, 70)
        self.plat_positions = {p.id: pos for p, pos in _raw_positions.items()}

        platform_pen = QPen(QColor('grey'))
        platform_brush = QColor('lightblue')
        line_pen = QPen(QColor('grey'), 2)

        # 简单分类用于标签摆放
        import re
        aux_keywords = re.compile(r"(连接|渡线|出段线|入段线|备用线|存车线|交叉渡线|折返线)")
        def cls(p):
            if "上行" in p.name: return "up"
            if "下行" in p.name: return "down"
            return "aux"

        # 记录已放置标签矩形，做碰撞避让
        label_rects = []

        def place_label(text_item, x, y):
            # 尝试避让已有标签
            rect = text_item.boundingRect()
            px, py = x, y
            tries, step = 0, 12.0
            def overlaps(a, b):
                return (a.x() < b.x() + b.width() and a.x() + a.width() > b.x() and
                        a.y() < b.y() + b.height() and a.y() + a.height() > b.y())
            while tries < 30:
                r = rect.translated(px, py)
                if not any(overlaps(r, o) for o in label_rects):
                    text_item.setPos(px, py)
                    label_rects.append(r)
                    return
                # 向下微移（up类先向上试一次，再向下）
                if tries == 0 and text_item.data(0) == "up":
                    py -= step
                else:
                    py += step
                tries += 1
            text_item.setPos(x, y)
            label_rects.append(rect.translated(x, y))

        # 先画平台与名字
        for platform in platforms:
            node_type = getattr(platform, "node_type", None)
            is_real_platform = node_type == NodeType.PLATFORM

            platform_radius = 6
            pos = self.plat_positions[platform.id]
            circle = HoverableGraphicsEllipseItem(
                platform, pos[0] - platform_radius, pos[1] - platform_radius,
                platform_radius * 2, platform_radius * 2
            )
            circle.setPen(platform_pen)
            circle.setBrush(platform_brush)
            self.topology_scene.addItem(circle)

            # 名字标签：依据类别放置在不同方位，减少重叠
            kind = cls(platform)
            name_item = self.topology_scene.addText(f"{platform.name}")
            name_item.setData(0, kind)  # 标记类别供避让方向使用
            if kind == "up":
                lx, ly = pos[0] - 10, pos[1] - (platform_radius + 16)
            elif kind == "down":
                lx, ly = pos[0] - 10, pos[1] + (platform_radius + 4)
            else:
                # 辅助点：放右侧
                lx, ly = pos[0] + (platform_radius + 8), pos[1] - (platform_radius + 6)
            place_label(name_item, lx, ly)

            # 等待人数标签：上行放下方， 下行放上方，辅助放下方
            if is_real_platform:
                waiting_label = self.topology_scene.addText("Waiting: 0")
                waiting_label.setDefaultTextColor(QColor('blue'))
                if kind == "down":
                    wx, wy = pos[0] - 15, pos[1] - (platform_radius + 18)
                else:
                    wx, wy = pos[0] - 15, pos[1] + (platform_radius + 18)
                waiting_label.setPos(wx, wy)
                self.station_waiting_items[platform.id] = waiting_label

        # 再画区段
        for seg in self.env.world.segments:
            # seg 是 LineSegment；端点统一用 id 查询
            start_pos = self.plat_positions.get(getattr(seg.start_platform, "id", None))
            end_pos = self.plat_positions.get(getattr(seg.end_platform, "id", None))
            if not start_pos or not end_pos:
                continue
            x1, y1 = start_pos
            x2, y2 = end_pos
            line_item = self.topology_scene.addLine(x1, y1, x2, y2, line_pen)
            self.segment_items[seg] = line_item
                # 初始着色
        self.refresh_segment_colors()

    def refresh_segment_colors(self):
        """根据当前中断状态刷新区段颜色：
           正常=灰色；BLOCK=红色；SPEED_LIMIT=橙色"""
        dm = getattr(self, "disruption_manager", None)
        if not dm or not hasattr(self, "segment_items"):
            return
        for seg_id, item in self.segment_items.items():
            info = dm.get_segment_disruption_info(seg_id)
            if info:
                if info.get("type") == "BLOCK":
                    color = QColor("#ff3b30")   # 红
                elif info.get("type") == "SPEED_LIMIT":
                    color = QColor("#ff9500")   # 橙
                else:
                    color = QColor("grey")
                width = 4
            else:
                color = QColor("grey")
                width = 2
            pen = item.pen()
            if pen.color() != color or pen.width() != width:
                pen.setColor(color)
                pen.setWidth(width)
                item.setPen(pen)

    def _get_node_position(self, node) -> QPointF:
        """Convert node coordinates to scene coordinates"""
        return QPointF(node.x * 100, node.y * 100)

    def _get_node_position_in_topology(self, node, segment) -> QPointF:
        """Convert node coordinates to topology scene coordinates for a segment."""
        start_position = self.plat_positions.get(segment.start_platform.id)
        end_position = self.plat_positions.get(segment.end_platform.id)
        if start_position is None or end_position is None:
            # 兜底：返回(0,0)避免崩溃
            return QPointF(0, 0)
        offset = node_in_segment_percentage(node, segment, self.env.segment2nodes)
        x = start_position[0] + (end_position[0] - start_position[0]) * offset
        y = start_position[1] + (end_position[1] - start_position[1]) * offset
        return QPointF(x, y)

    @staticmethod
    def _extract_position_value(position, index, default=-1):
        try:
            candidate = position[index]
            candidate = candidate.item() if hasattr(candidate, "item") else candidate
            return int(candidate)
        except (IndexError, TypeError, AttributeError, ValueError):
            pass
        try:
            return int(position.item())
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            return int(position)
        except (TypeError, ValueError):
            return default

    def _create_flow_info_widget(self) -> QWidget:
        """Create passenger flow panel similar to the separate dialog."""
        container = QWidget()
        container.setFixedWidth(int(self.width() * 0.2) + 2)
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        container.setStyleSheet(
            f"background-color: {self.panel_bg_color}; border: 1px solid #d9deff;"
        )

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        title_label = QLabel("Passenger Flow Info")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(title_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none; background: transparent;")
        main_layout.addWidget(scroll_area)

        content_widget = QWidget()
        content_widget.setStyleSheet(f"background-color: {self.panel_bg_color};")
        scroll_area.setWidget(content_widget)

        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(4, 0, 4, 0)
        content_layout.setSpacing(4)

        train_title = QLabel("Train Passenger Flow")
        train_title.setStyleSheet("font-weight: bold;")
        content_layout.addWidget(train_title)

        self.flow_train_layout = QVBoxLayout()
        self.flow_train_layout.setSpacing(2)
        content_layout.addLayout(self.flow_train_layout)

        station_title = QLabel("Station Waiting Passengers")
        station_title.setStyleSheet("font-weight: bold; margin-top: 6px;")
        content_layout.addWidget(station_title)

        self.flow_station_layout = QVBoxLayout()
        self.flow_station_layout.setSpacing(2)
        content_layout.addLayout(self.flow_station_layout)

        content_layout.addStretch(1)

        self.flow_train_labels = []
        self.flow_station_labels = {}

        self._ensure_flow_info_entries()

        return container

    def _ensure_flow_info_entries(self):
        """Ensure passenger flow panel has labels for current trains and stations."""
        if not hasattr(self, 'flow_train_layout'):
            return

        trains = list(self.env.get_all_trains())

        if not trains and not self.flow_train_labels:
            placeholder = QLabel("No train data")
            placeholder.setWordWrap(True)
            self.flow_train_layout.addWidget(placeholder)
            self.flow_train_labels.append(placeholder)

        while len(self.flow_train_labels) < len(trains):
            idx = len(self.flow_train_labels)
            label = QLabel(f"Train {idx + 1}: Initializing")
            label.setWordWrap(True)
            self.flow_train_layout.addWidget(label)
            self.flow_train_labels.append(label)

        world = getattr(self.env, 'scenario', None)
        if world and hasattr(world, 'world') and hasattr(world.world, 'platform_nodes'):
            nodes_items = list(world.world.platform_nodes.items())
            try:
                nodes_items.sort(key=lambda item: item[0])
            except TypeError:
                pass
            for node_id, node in nodes_items:
                if node_id not in self.flow_station_labels:
                    name = getattr(node, 'name', node_id)
                    label = QLabel(f"Station {name}: Initializing")
                    label.setWordWrap(True)
                    self.flow_station_layout.addWidget(label)
                    self.flow_station_labels[node_id] = label

    def _create_train_list(self) -> QWidget:
        """Create and configure train list widget"""
        logger.info("Creating train list")
        container = QWidget()
        container.setFixedWidth(int(self.width() * 0.2) + 2)
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        title_label = QLabel("Train Positions")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(title_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none; background: transparent;")
        main_layout.addWidget(scroll_area)

        self.trainList = QListWidget()
        self.trainList.setStyleSheet(
            f"background-color: {self.panel_bg_color}; border: none;"
        )
        scroll_area.setWidget(self.trainList)

        return container

    def _update_status(self):
        """Update status bar information"""
        disrupted_segments = self.disruption_manager.get_disrupted_segments() # 8. 使用 self.disruption_manager
        disruption_info = f", Disrupted Segments: {len(disrupted_segments)}" if disrupted_segments else ""
        
        self.status_labels['time'].setText(f"Time: {self.env.state.time}")
        self.status_labels['status'].setText(
            f"Status: {'Running' if self.env.state.is_running else 'Stopped'}"
        )

    def _update_flow_info_panel(self):
        """Refresh embedded passenger flow information."""
        if not hasattr(self, 'flow_train_labels'):
            return

        self._ensure_flow_info_entries()

        trains = list(self.env.get_all_trains())
        for idx, train in enumerate(trains):
            label = self.flow_train_labels[idx]
            train_identifier = getattr(train, 'name', getattr(train, 'id', idx + 1))

            passenger_count = 0
            if hasattr(train, 'passenger_count') and train.passenger_count is not None:
                try:
                    passenger_count = int(train.passenger_count[self.env_index].item())
                except (IndexError, AttributeError, ValueError):
                    try:
                        passenger_count = int(train.passenger_count.item())
                    except (AttributeError, ValueError):
                        passenger_count = 0

            position_text = "Unknown"
            if hasattr(train, 'state') and hasattr(train.state, 'position'):
                try:
                    position_text = int(train.state.position[self.env_index].item())
                except (IndexError, AttributeError, ValueError):
                    try:
                        position_text = int(train.state.position.item())
                    except (AttributeError, ValueError):
                        position_text = train.state.position

            dest_info_items = []
            if hasattr(train, 'passengers_by_destination') and train.passengers_by_destination:
                for dest, counts in train.passengers_by_destination.items():
                    try:
                        count_value = int(counts[self.env_index].item())
                    except (IndexError, AttributeError, ValueError):
                        continue
                    if count_value > 0:
                        dest_info_items.append((dest, count_value))

            dest_suffix = ""
            if dest_info_items:
                dest_info_items.sort(key=lambda item: item[1], reverse=True)
                formatted = ", ".join([f"Stn{dest}({count}p)" for dest, count in dest_info_items[:3]])
                dest_suffix = f" | Main destinations: {formatted}"

            label.setText(
                f"Train {train_identifier}: Position {position_text} | Passengers {passenger_count}{dest_suffix}"
            )

        for idx in range(len(trains), len(self.flow_train_labels)):
            self.flow_train_labels[idx].setText("No train data")

        world = getattr(self.env, 'scenario', None)
        if world and hasattr(world, 'world') and hasattr(world.world, 'platform_nodes'):
            for node_id, node in world.world.platform_nodes.items():
                label = self.flow_station_labels.get(node_id)
                if label is None:
                    continue

                waiting_count = 0
                if hasattr(node, 'waiting_passengers') and node.waiting_passengers is not None:
                    try:
                        waiting_count = int(node.waiting_passengers[self.env_index].item())
                    except (IndexError, AttributeError, ValueError):
                        try:
                            waiting_count = int(node.waiting_passengers.item())
                        except (AttributeError, ValueError):
                            waiting_count = 0

                node_name = getattr(node, 'name', node_id)
                label.setText(f"Station {node_name}: Waiting {waiting_count}")

    def zoomIn(self):
        logger.info("Zooming in")
        self.topology_view.scale(1.25, 1.25)

    def zoomOut(self):
        logger.info("Zooming out")
        self.topology_view.scale(0.8, 0.8)

    def _add_timeline_point(self, x, y):
        """Add a point to the timeline visualization"""
        point = QGraphicsEllipseItem(x, y, 2, 2)
        point.setBrush(QBrush(QColor('green')))
        point.setPen(QPen(Qt.PenStyle.NoPen))
        self.timeline_scene.addItem(point)

    def _ensure_timeline_trace(self, train):
        """Ensure a reusable timeline trace exists for the given train."""
        trace = self.timeline_traces.get(train)
        if trace is None:
            path_item = QGraphicsPathItem()
            path_item.setPen(QPen(QColor('green'), 1.5))
            self.timeline_scene.addItem(path_item)
            trace = {
                "item": path_item,
                "path": QPainterPath(),
                "last_step": None,
                "last_pos": None,
            }
            self.timeline_traces[train] = trace
        elif trace["item"].scene() is None:
            self.timeline_scene.addItem(trace["item"])
        return trace

    def _append_timeline_point(self, train, x, y, current_step):
        """Append a point to the train's timeline path with basic decimation."""
        if self.timeline_tickwidth <= 0:
            return

        trace = self._ensure_timeline_trace(train)
        last_step = trace["last_step"]
        if last_step is not None and current_step == last_step:
            return

        path = trace["path"]
        point = QPointF(float(x), float(y))

        # Reset the trace when the simulation restarts or rewinds
        if last_step is None or current_step < last_step or trace["last_pos"] is None:
            path = QPainterPath(point)
        else:
            last_pos = trace["last_pos"]
            if last_pos is not None and abs(point.x() - last_pos.x()) < self.timeline_point_min_dx:
                trace["last_step"] = current_step
                return
            path.lineTo(point)

        trace["path"] = path
        trace["item"].setPath(path)
        trace["last_step"] = current_step
        trace["last_pos"] = point

    def resizeEvent(self, event):
        """Handle window resize event to adjust timeline graph"""
        super().resizeEvent(event)
        self._update_timeline()

    def showEvent(self, event):
        """Handle window show event to adjust timeline graph"""
        super().showEvent(event)
        self._update_timeline()

    def _update_timeline(self):
        """Update timeline visualization based on current viewport size"""
        self.timeline_scene.clear()
        self.timeline_traces.clear()
        pen = QPen(QColor('grey'))
        pen.setWidth(2)

        pen2 = QPen(QColor('lightgrey'))
        pen2.setWidth(1)

        # Get the viewport size
        self.timeline_width = self.timeline_view.viewport().width()
        self.timeline_height = self.timeline_view.viewport().height()
        self.timeline_margin = 40
        self.timeline_tickwidth = (self.timeline_width - 2 * self.timeline_margin) / self.maxtime
        self.timeline_point_min_dx = max(self.timeline_tickwidth * 0.25, 0.01)

        self.platform_horiz_pos = {}
        station_lookup = {}
        for group in stations:
            for platform in group:
                station_lookup[platform] = group

        station_key_to_label = {
            tuple(sorted(p.id for p in group)): f"s{idx}"
            for idx, group in enumerate(stations, start=1)
        }

        station_groups = []
        seen_group_keys = set()

        for platform in platforms:
            group = station_lookup.get(platform, (platform,))
            group_key = tuple(sorted(p.id for p in group))
            if group_key in seen_group_keys:
                continue
            seen_group_keys.add(group_key)
            station_groups.append(tuple(group))

        for group in stations:
            group_key = tuple(sorted(p.id for p in group))
            if group_key in seen_group_keys:
                continue
            seen_group_keys.add(group_key)
            station_groups.append(tuple(group))

        if not station_groups:
            return

        interval = (self.timeline_height - (2 * self.timeline_margin)) / max(len(station_groups), 1)

        for idx, group in enumerate(station_groups):
            y = self.timeline_height - self.timeline_margin - (idx * interval)
            for platform in group:
                self.platform_horiz_pos[platform] = y

            self.timeline_scene.addLine(
                self.timeline_margin,
                y,
                self.timeline_width - self.timeline_margin,
                y,
                pen2,
            )

            group_key = tuple(sorted(p.id for p in group))
            station_label = station_key_to_label.get(group_key)

            if station_label is None:
                ordered_names = dict.fromkeys(p.name for p in group)
                station_label = " / ".join(ordered_names.keys())
            station_text = self.timeline_scene.addText(station_label)
            station_text_height = station_text.boundingRect().height()
            station_text_width = station_text.boundingRect().width()
            station_text.setPos(
                self.timeline_margin - station_text_width / 2 - 15,
                y - station_text_height / 2,
            )

        # for i in range(len(switchnodes)):
        #     # Draw switchnodes lines
        #     self.platform_horiz_pos[switchnodes[i]] = self.timeline_height-self.timeline_margin-(i*interval)

        # Draw axes using full width/height while respecting margins
        self.timeline_scene.addLine(self.timeline_margin, self.timeline_height-self.timeline_margin, self.timeline_width-self.timeline_margin, self.timeline_height-self.timeline_margin, pen)  # X axis
        self.timeline_scene.addLine(self.timeline_margin, self.timeline_margin, self.timeline_margin, self.timeline_height-self.timeline_margin, pen)    # Y axis
        # Add time labels with proper spacing
        # time_width = width - (2 * margin)
        # for i in range(5):
        #     time_text = self.timeline_scene.addText(f"{i*6}:00")
        #     time_text.setPos(margin + (i * time_width/4), height-margin+10)

        self._draw_planned_timetable()


    def _draw_planned_timetable(self):
        """Render the planned timetable along the timeline axes."""
        scenario = getattr(self.env, "scenario", None)
        world = getattr(scenario, "world", None) or getattr(self.env, "world", None)
        if not world:
            return

        timetable = getattr(world, "active_timetable", None)
        if timetable is None or not timetable.runs:
            return

        palette = [
            QColor(59, 130, 246),  # blue-500
            QColor(16, 185, 129),  # emerald-500
            QColor(236, 72, 153),  # pink-500
            QColor(245, 158, 11),  # amber-500
            QColor(168, 85, 247),  # violet-500
        ]

        for idx, run in enumerate(timetable.runs):
            if not run.stops:
                continue

            color = palette[idx % len(palette)]
            pen = QPen(color)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidthF(1.5)
            pen.setCosmetic(True)

            path = QPainterPath()
            last_point = None

            for stop in run.stops:
                platform = get_platform_by_id(stop.platform_id)
                if platform not in self.platform_horiz_pos:
                    continue

                y = self.platform_horiz_pos[platform]
                arrival_x = self.timeline_margin + stop.arrival_time * self.timeline_tickwidth
                departure_x = self.timeline_margin + stop.departure_time * self.timeline_tickwidth

                arrival_pt = QPointF(arrival_x, y)
                departure_pt = QPointF(departure_x, y)

                if last_point is None:
                    path.moveTo(arrival_pt)
                else:
                    path.lineTo(arrival_pt)

                if departure_x != arrival_x:
                    path.lineTo(departure_pt)
                    last_point = departure_pt
                else:
                    last_point = arrival_pt

            if path.elementCount() > 1:
                path_item = self.timeline_scene.addPath(path, pen)
                path_item.setZValue(-1)


    def refresh_trains(self):
        """Refresh train list, topology symbols, and timeline traces."""
        self._prepare_train_refresh()
        self.refresh_segment_colors()

        trains = list(self.env.get_all_trains())
        train_states = self._collect_train_states(trains)
        self._populate_train_list(train_states)

        current_step = self._extract_position_value(getattr(self.env, "steps", 0), self.env_index, default=0)
        base_x = self.timeline_margin + current_step * self.timeline_tickwidth

        active_trains = set()
        for train, node_id, run_id, segment_idx in train_states:
            active_trains.add(train)
            if node_id == -1:
                continue
            segment = self._select_segment(node_id)
            if segment is None:
                continue
            self._update_train_visuals(train, node_id, segment, current_step, base_x)

        self._cleanup_inactive_traces(active_trains)
        self._update_flow_info_panel()

    def _prepare_train_refresh(self):
        """Reset transient UI state and refresh platform metadata."""
        self.trainList.clear()
        self.refresh_platform_colors()
        # self.update_switchnode_occupancy(self.env)
        self._refresh_station_waiting_labels()

    def _refresh_station_waiting_labels(self):
        """Update the waiting passenger indicators for each platform."""
        scenario = getattr(self.env, "scenario", None)
        world = getattr(scenario, "world", None)
        if not world:
            return
        for node_id, node in world.platform_nodes.items():
            waiting_label = self.station_waiting_items.get(node_id)
            if waiting_label is None:
                continue
            waiting_value = getattr(node, "waiting_passengers", None)
            if waiting_value is None:
                continue
            count = self._extract_position_value(waiting_value, self.env_index, default=0)
            waiting_label.setPlainText(f"w: {count}")

    def _collect_train_states(self, trains):
        """Return a list of `(train, node_id, run_id, segment_idx)` tuples."""
        states = []
        for train in trains:
            node_id = self._extract_position_value(train.state.position, self.env_index, default=-1)
            run_id = None
            segment_idx = None

            assigned = getattr(train.state, "assigned_run", None)
            if assigned is not None:
                value = self._extract_position_value(assigned, self.env_index, default=-1)
                if value >= 0:
                    run_id = value

            segment_tensor = getattr(train.state, "run_segment_index", None)
            if segment_tensor is not None:
                value = self._extract_position_value(segment_tensor, self.env_index, default=-1)
                if value >= 0:
                    segment_idx = value

            states.append((train, node_id, run_id, segment_idx))
        return states

    def _populate_train_list(self, train_states):
        """Populate the sidebar with readable train position entries."""
        for train, node_id, run_id, segment_idx in train_states:
            if node_id == -1:
                continue
            train_name = getattr(train, "name", getattr(train, "id", "Unknown"))
            details = [f"Node {node_id}"]

            # Append current action in a readable form (Stop/Move)
            action_val = self._extract_position_value(getattr(train, "action", None), self.env_index, default=-1)
            if action_val != -1:
                action_text = "Stop" if action_val == 0 else ("Move" if action_val == 1 else str(action_val))
                details.append(f"Act {action_text}")
            if run_id is not None:
                details.append(f"Run {run_id}")
            if segment_idx is not None:
                details.append(f"Seg {segment_idx}")
            detail_text = " | ".join(details)
            self.trainList.addItem(f"Train {train_name}: {detail_text}")

    def _select_segment(self, node_id):
        """Return the first segment associated with `node_id`, or None."""
        node_segments = getattr(self.env, "node2segments", {})
        try:
            segments = node_segments[node_id]
        except (KeyError, TypeError):
            return None
        if isinstance(segments, (list, tuple)):
            return segments[0] if segments else None
        return segments

    def _update_train_visuals(self, train, node_id, segment, current_step, base_x):
        """Draw timeline and topology items for a specific train."""
        arrival_state = self._compute_arrival_state(train, node_id)
        node = self.env.nodes.get(node_id)
        if node is None:
            return

        self._update_timeline_path(train, node, segment, base_x, current_step)
        self._remove_previous_train_items(train, arrival_state)

        node_position = self._get_node_position_in_topology(node, segment)
        train_item = HoverableGraphicsEllipseItem(
            node,
            node_position.x() - 5,
            node_position.y() - 5,
            10,
            10,
        )
        train_item.setBrush(QBrush(QColor("red")))
        self.topology_scene.addItem(train_item)
        self.train_items[train] = train_item

        train_id = getattr(train, "name", getattr(train, "id", "?"))
        train_label = self.topology_scene.addText(str(train_id))
        train_label.setDefaultTextColor(QColor("black"))
        train_label.setPos(node_position.x() - 15, node_position.y() - 25)
        self.train_label_items[train] = train_label

        passenger_count = 0
        if hasattr(train, "passenger_count") and train.passenger_count is not None:
            passenger_count = self._extract_position_value(train.passenger_count, self.env_index, default=0)
        passenger_label = self.topology_scene.addText(f"{passenger_count}p")
        passenger_label.setDefaultTextColor(QColor("grey"))
        passenger_label.setPos(node_position.x() - 10, node_position.y())
        self.train_passenger_items[train] = passenger_label

        self._display_alighting_info(train, node_position, arrival_state)

    def _compute_arrival_state(self, train, node_id):
        """Return arrival metadata (arrival flag, platform flag, alighting count)."""
        scenario = getattr(self.env, "scenario", None)
        world = getattr(scenario, "world", None)
        is_at_platform = bool(world and node_id in world.platform_nodes)

        previous_node = self.train_previous_positions.get(train)
        just_arrived = previous_node is not None and previous_node != node_id and is_at_platform

        alighting_count = 0
        if just_arrived:
            alighting_count = self._read_alighting_count(train, node_id)
            if alighting_count > 0:
                self._clear_existing_alighting_display(train)

        self.train_previous_positions[train] = node_id
        return TrainArrivalState(just_arrived, is_at_platform, alighting_count)

    def _read_alighting_count(self, train, node_id):
        """Read the number of passengers scheduled to alight at `node_id`."""
        passengers_by_destination = getattr(train, "passengers_by_destination", None)
        if passengers_by_destination is None:
            return 0
        if isinstance(passengers_by_destination, dict):
            destination_counts = passengers_by_destination.get(node_id)
        else:
            try:
                destination_counts = passengers_by_destination[node_id]
            except (TypeError, KeyError, IndexError):
                destination_counts = None
        if destination_counts is None:
            return 0
        return self._extract_position_value(destination_counts, self.env_index, default=0)

    def _clear_existing_alighting_display(self, train):
        """Remove any prior alighting label and stop its timer."""
        label = self.train_alighting_items.get(train)
        if label is not None and label.scene() is self.topology_scene:
            self.topology_scene.removeItem(label)
        self.train_alighting_items[train] = None

        timer = self.alighting_display_timers.get(train)
        if timer is not None:
            timer.stop()
        self.alighting_display_timers[train] = None

    def _update_timeline_path(self, train, node, segment, base_x, current_step):
        """Append the current position to the train's timeline trace."""
        startplat = segment.start_platform
        endplat = segment.end_platform
        if startplat not in self.platform_horiz_pos or endplat not in self.platform_horiz_pos:
            return
        offset = node_in_segment_percentage(node, segment, self.env.segment2nodes)
        y = self.platform_horiz_pos[startplat] - (
            self.platform_horiz_pos[startplat] - self.platform_horiz_pos[endplat]
        ) * offset
        max_x = self.timeline_width - self.timeline_margin
        clamped_x = min(max(base_x, self.timeline_margin), max_x)
        self._append_timeline_point(train, clamped_x, y, current_step)

    def _remove_previous_train_items(self, train, arrival_state):
        """Clear graphics associated with `train` before redraw."""
        item = self.train_items.pop(train, None)
        if item is not None and item.scene() is self.topology_scene:
            self.topology_scene.removeItem(item)

        label = self.train_label_items.pop(train, None)
        if label is not None and label.scene() is self.topology_scene:
            self.topology_scene.removeItem(label)

        passenger_label = self.train_passenger_items.pop(train, None)
        if passenger_label is not None and passenger_label.scene() is self.topology_scene:
            self.topology_scene.removeItem(passenger_label)

        self._remove_stale_alighting_label(train, arrival_state)

    def _remove_stale_alighting_label(self, train, arrival_state):
        """Remove alighting info once the display timer finishes."""
        label = self.train_alighting_items.get(train)
        if label is None or arrival_state.just_arrived:
            return
        timer = self.alighting_display_timers.get(train)
        if timer is not None and timer.isActive():
            return
        if label.scene() is self.topology_scene:
            self.topology_scene.removeItem(label)
        self.train_alighting_items[train] = None
        if timer is not None:
            timer.stop()
        self.alighting_display_timers[train] = None

    def _display_alighting_info(self, train, node_position, arrival_state):
        """Show a temporary label for passengers alighting from the train."""
        if not arrival_state.just_arrived or not arrival_state.is_at_platform:
            return
        if arrival_state.alighting_count <= 0:
            return

        alighting_label = self.topology_scene.addText(f"↓{arrival_state.alighting_count}p")
        alighting_label.setDefaultTextColor(QColor("darkgreen"))
        alighting_label.setPos(node_position.x() + 8, node_position.y() - 5)
        self.train_alighting_items[train] = alighting_label

        timer = QTimer(self)
        timer.setSingleShot(True)

        def remove_alighting_label(t=train):
            label_to_remove = self.train_alighting_items.get(t)
            if label_to_remove is not None and label_to_remove.scene() is self.topology_scene:
                self.topology_scene.removeItem(label_to_remove)
            self.train_alighting_items[t] = None
            self.alighting_display_timers[t] = None

        timer.timeout.connect(remove_alighting_label)
        timer.start(5000)
        self.alighting_display_timers[train] = timer

    def _cleanup_inactive_traces(self, active_trains):
        """Remove timeline traces for trains no longer present."""
        for train in list(self.timeline_traces.keys()):
            if train not in active_trains:
                trace = self.timeline_traces.pop(train)
                if trace["item"].scene() is self.timeline_scene:
                    self.timeline_scene.removeItem(trace["item"])
    def _on_start(self):
        """Handle start button click"""
        logger.info("Start button clicked")
        self.env.start()

    def _on_pause(self):
        """Handle pause button click"""
        logger.info("Pause button clicked")
        self.env.pause()

    def _on_stop(self):
        """Handle stop button click"""
        logger.info("Stop button clicked")
        self.env.reset()

    def refresh_platform_colors(self):
        for item in self.node_items.values():
            platform = get_platform_by_id(item.node.id)
            if platform:
                item.set_station_mode(platform, platform.mode)
