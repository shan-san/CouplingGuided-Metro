from __future__ import annotations
from calendar import c
import os
from typing import List, Tuple, Dict, Optional
import re

from facility.platform import Platform
from topology.route import Route, RouteType
from topology.node import set_passenger_whitelist
from topology.node_type import NodeType

# 全局暴露：其它模块引用
platforms: List[Platform] = []
routes: List[Route] = []
stations: List[Tuple[Platform, ...]] = []  # 新增：当前拓扑的“车站”平台分组
line_segments: List = []  # 留空 => 其它模块可从 routes 推导
ACTIVE_TOPOLOGY: Optional[int] = None

# ----------------------------
# 深圳地铁1号线（近似）客流OD配置
# 说明：用于 topology16 的硬编码客流生成（近似值，非官方数据）
# ----------------------------
LINE1_STATION_ORDER: List[str] = [
    "机场东", "后瑞", "固戍", "西乡", "坪洲", "宝体", "宝安中心", "新安", "前海湾",
    "鲤鱼门", "大新", "桃园", "深大", "高新园", "白石洲", "世界之窗", "华侨城", "侨城东",
    "竹子林", "车公庙", "香蜜湖", "购物公园", "会展中心", "岗厦", "华强路", "科学馆",
    "大剧院", "老街", "国贸", "罗湖",
]

# 站点基础到达率（近似，相对强度）
LINE1_STATION_BASE_RATE: Dict[str, float] = {
    "机场东": 0.55, "后瑞": 0.45, "固戍": 0.60, "西乡": 0.85, "坪洲": 0.65,
    "宝体": 0.80, "宝安中心": 1.00, "新安": 0.95, "前海湾": 1.05, "鲤鱼门": 0.70,
    "大新": 0.80, "桃园": 0.75, "深大": 1.10, "高新园": 1.20, "白石洲": 1.00,
    "世界之窗": 1.30, "华侨城": 1.00, "侨城东": 0.85, "竹子林": 1.05, "车公庙": 1.25,
    "香蜜湖": 0.90, "购物公园": 1.35, "会展中心": 1.30, "岗厦": 1.00, "华强路": 1.15,
    "科学馆": 1.10, "大剧院": 1.25, "老街": 1.40, "国贸": 1.20, "罗湖": 1.50,
}

# 站点吸引强度（用于目的地分布）
LINE1_STATION_ATTRACTION: Dict[str, float] = {
    "机场东": 0.70, "后瑞": 0.65, "固戍": 0.75, "西乡": 0.95, "坪洲": 0.80,
    "宝体": 0.90, "宝安中心": 1.10, "新安": 1.00, "前海湾": 1.10, "鲤鱼门": 0.85,
    "大新": 0.95, "桃园": 0.90, "深大": 1.30, "高新园": 1.35, "白石洲": 1.05,
    "世界之窗": 1.40, "华侨城": 1.10, "侨城东": 0.95, "竹子林": 1.15, "车公庙": 1.30,
    "香蜜湖": 1.00, "购物公园": 1.45, "会展中心": 1.35, "岗厦": 1.05, "华强路": 1.20,
    "科学馆": 1.15, "大剧院": 1.35, "老街": 1.55, "国贸": 1.30, "罗湖": 1.60,
}

# 深圳地铁1号线全 OD 矩阵（单位：万/小时，近似值）
# 每一行代表出发站，每一列代表到达站
LINE1_OD_MATRIX: Dict[str, Dict[str, float]] = {
    "机场东": {"机场东": 0, "后瑞": 0.0292, "固戍": 0.0295, "西乡": 0.0332, "坪洲": 0.0251, "宝体": 0.0257, "宝安中心": 0.0288, "新安": 0.0242, "前海湾": 0.0247, "鲤鱼门": 0.0178, "大新": 0.0187, "桃园": 0.0166, "深大": 0.0227, "高新园": 0.0223, "白石洲": 0.0165, "世界之窗": 0.0209, "华侨城": 0.0157, "侨城东": 0.013, "竹子林": 0.0151, "车公庙": 0.0163, "香蜜湖": 0.0121, "购物公园": 0.0169, "会展中心": 0.0151, "岗厦": 0.0114, "华强路": 0.0126, "科学馆": 0.0117, "大剧院": 0.0133, "老街": 0.0148, "国贸": 0.012, "罗湖": 0.0144},
    "后瑞": {"机场东": 0.0241, "后瑞": 0, "固戍": 0.0258, "西乡": 0.0286, "坪洲": 0.0214, "宝体": 0.0216, "宝安中心": 0.0241, "新安": 0.02, "前海湾": 0.0204, "鲤鱼门": 0.0146, "大新": 0.0152, "桃园": 0.0135, "深大": 0.0184, "高新园": 0.018, "白石洲": 0.0133, "世界之窗": 0.0168, "华侨城": 0.0126, "侨城东": 0.0104, "竹子林": 0.012, "车公庙": 0.013, "香蜜湖": 0.0096, "购物公园": 0.0134, "会展中心": 0.012, "岗厦": 0.009, "华强路": 0.01, "科学馆": 0.0092, "大剧院": 0.0105, "老街": 0.0117, "国贸": 0.0095, "罗湖": 0.0113},
    "固戍": {"机场东": 0.0268, "后瑞": 0.0284, "固戍": 0, "西乡": 0.0415, "坪洲": 0.0306, "宝体": 0.0306, "宝安中心": 0.0337, "新安": 0.0278, "前海湾": 0.0281, "鲤鱼门": 0.02, "大新": 0.0208, "桃园": 0.0184, "深大": 0.0249, "高新园": 0.0243, "白石洲": 0.0179, "世界之窗": 0.0225, "华侨城": 0.0168, "侨城东": 0.0138, "竹子林": 0.016, "车公庙": 0.0173, "香蜜湖": 0.0128, "购物公园": 0.0177, "会展中心": 0.0159, "岗厦": 0.0119, "华强路": 0.0131, "科学馆": 0.0121, "大剧院": 0.0138, "老街": 0.0153, "国贸": 0.0124, "罗湖": 0.0148},
    "西乡": {"机场东": 0.0326, "后瑞": 0.0341, "固戍": 0.0449, "西乡": 0, "坪洲": 0.0479, "宝体": 0.0472, "宝安中心": 0.0512, "新安": 0.0419, "前海湾": 0.0419, "鲤鱼门": 0.0297, "大新": 0.0306, "桃园": 0.027, "深大": 0.0363, "高新园": 0.0354, "白石洲": 0.0259, "世界之窗": 0.0326, "华侨城": 0.0243, "侨城东": 0.0199, "竹子林": 0.023, "车公庙": 0.0248, "香蜜湖": 0.0182, "购物公园": 0.0253, "会展中心": 0.0226, "岗厦": 0.0169, "华强路": 0.0186, "科学馆": 0.0172, "大剧院": 0.0195, "老街": 0.0217, "国贸": 0.0176, "罗湖": 0.021},
    "坪洲": {"机场东": 0.0213, "后瑞": 0.022, "固戍": 0.0285, "西乡": 0.0413, "坪洲": 0, "宝体": 0.0391, "宝安中心": 0.0418, "新安": 0.0338, "前海湾": 0.0334, "鲤鱼门": 0.0235, "大新": 0.0241, "桃园": 0.021, "深大": 0.0282, "高新园": 0.0274, "白石洲": 0.02, "世界之窗": 0.025, "华侨城": 0.0186, "侨城东": 0.0152, "竹子林": 0.0175, "车公庙": 0.0188, "香蜜湖": 0.0138, "购物公园": 0.0192, "会展中心": 0.0171, "岗厦": 0.0128, "华强路": 0.014, "科学馆": 0.0129, "大剧院": 0.0147, "老街": 0.0162, "国贸": 0.0132, "罗湖": 0.0157},
    "宝体": {"机场东": 0.0231, "后瑞": 0.0236, "固戍": 0.0302, "西乡": 0.0431, "坪洲": 0.0415, "宝体": 0, "宝安中心": 0.057, "新安": 0.0454, "前海湾": 0.0444, "鲤鱼门": 0.0308, "大新": 0.0313, "桃园": 0.0272, "深大": 0.0363, "高新园": 0.035, "白石洲": 0.0254, "世界之窗": 0.0318, "华侨城": 0.0235, "侨城东": 0.0192, "竹子林": 0.022, "车公庙": 0.0236, "香蜜湖": 0.0173, "购物公园": 0.0239, "会展中心": 0.0213, "岗厦": 0.0159, "华强路": 0.0174, "科学馆": 0.0161, "大剧院": 0.0181, "老街": 0.0201, "国贸": 0.0163, "罗湖": 0.0194},
    "宝安中心": {"机场东": 0.0259, "后瑞": 0.0263, "固戍": 0.0334, "西乡": 0.0469, "坪洲": 0.0445, "宝体": 0.0572, "宝安中心": 0, "新安": 0.0635, "前海湾": 0.0611, "鲤鱼门": 0.042, "大新": 0.0422, "桃园": 0.0364, "深大": 0.0482, "高新园": 0.0462, "白石洲": 0.0334, "世界之窗": 0.0415, "华侨城": 0.0306, "侨城东": 0.0249, "竹子林": 0.0284, "车公庙": 0.0304, "香蜜湖": 0.0222, "购物公园": 0.0307, "会展中心": 0.0273, "岗厦": 0.0203, "华强路": 0.0222, "科学馆": 0.0205, "大剧院": 0.0231, "老街": 0.0255, "国贸": 0.0206, "罗湖": 0.0245},
    "新安": {"机场东": 0.022, "后瑞": 0.0221, "固戍": 0.0278, "西乡": 0.0388, "坪洲": 0.0363, "宝体": 0.0459, "宝安中心": 0.0641, "新安": 0, "前海湾": 0.0641, "鲤鱼门": 0.0433, "大新": 0.0431, "桃园": 0.0367, "深大": 0.0482, "高新园": 0.0459, "白石洲": 0.0329, "世界之窗": 0.0408, "华侨城": 0.0299, "侨城东": 0.0242, "竹子林": 0.0276, "车公庙": 0.0295, "香蜜湖": 0.0215, "购物公园": 0.0296, "会展中心": 0.0262, "岗厦": 0.0195, "华强路": 0.0213, "科学馆": 0.0195, "大剧院": 0.022, "老街": 0.0243, "国贸": 0.0196, "罗湖": 0.0233},
    "前海湾": {"机场东": 0.0222, "后瑞": 0.0222, "固戍": 0.0277, "西乡": 0.0383, "坪洲": 0.0355, "宝体": 0.0443, "宝安中心": 0.0609, "新安": 0.0633, "前海湾": 0, "鲤鱼门": 0.0538, "大新": 0.0526, "桃园": 0.0443, "深大": 0.0576, "高新园": 0.0544, "白石洲": 0.0388, "世界之窗": 0.0477, "华侨城": 0.0348, "侨城东": 0.0281, "竹子林": 0.0319, "车公庙": 0.0339, "香蜜湖": 0.0246, "购物公园": 0.0338, "会展中心": 0.0299, "岗厦": 0.0222, "华强路": 0.0242, "科学馆": 0.0222, "大剧院": 0.0249, "老街": 0.0275, "国贸": 0.0222, "罗湖": 0.0263},
    "鲤鱼门": {"机场东": 0.0133, "后瑞": 0.0132, "固戍": 0.0164, "西乡": 0.0226, "坪洲": 0.0207, "宝体": 0.0256, "宝安中心": 0.0348, "新安": 0.0356, "前海湾": 0.0448, "鲤鱼门": 0, "大新": 0.0387, "桃园": 0.0321, "深大": 0.0412, "高新园": 0.0385, "白石洲": 0.0272, "世界之窗": 0.0332, "华侨城": 0.0241, "侨城东": 0.0193, "竹子林": 0.0218, "车公庙": 0.0232, "香蜜湖": 0.0168, "购物公园": 0.023, "会展中心": 0.0202, "岗厦": 0.015, "华强路": 0.0163, "科学馆": 0.0149, "大剧院": 0.0167, "老街": 0.0184, "国贸": 0.0148, "罗湖": 0.0175},
    "大新": {"机场东": 0.0141, "后瑞": 0.0139, "固戍": 0.0172, "西乡": 0.0235, "坪洲": 0.0214, "宝体": 0.0263, "宝安中心": 0.0354, "新安": 0.0357, "前海湾": 0.0442, "鲤鱼门": 0.039, "大新": 0, "桃园": 0.0413, "深大": 0.0522, "高新园": 0.0482, "白石洲": 0.0337, "世界之窗": 0.0409, "华侨城": 0.0295, "侨城东": 0.0235, "竹子林": 0.0264, "车公庙": 0.0279, "香蜜湖": 0.0201, "购物公园": 0.0274, "会展中心": 0.0241, "岗厦": 0.0178, "华强路": 0.0193, "科学馆": 0.0176, "大剧院": 0.0197, "老街": 0.0217, "国贸": 0.0174, "罗湖": 0.0206},
    "桃园": {"机场东": 0.0121, "后瑞": 0.012, "固戍": 0.0147, "西乡": 0.02, "坪洲": 0.0181, "宝体": 0.0221, "宝安中心": 0.0295, "新安": 0.0295, "前海湾": 0.036, "鲤鱼门": 0.0313, "大新": 0.04, "桃园": 0, "深大": 0.0547, "高新园": 0.0497, "白石洲": 0.0344, "世界之窗": 0.0412, "华侨城": 0.0295, "侨城东": 0.0233, "竹子林": 0.0261, "车公庙": 0.0274, "香蜜湖": 0.0196, "购物公园": 0.0267, "会展中心": 0.0234, "岗厦": 0.0172, "华强路": 0.0186, "科学馆": 0.0169, "大剧院": 0.0189, "老街": 0.0208, "国贸": 0.0167, "罗湖": 0.0196},
    "深大": {"机场东": 0.0169, "后瑞": 0.0166, "固戍": 0.0204, "西乡": 0.0275, "坪洲": 0.0248, "宝体": 0.0301, "宝安中心": 0.0398, "新安": 0.0395, "前海湾": 0.0478, "鲤鱼门": 0.041, "大新": 0.0516, "桃园": 0.0559, "深大": 0, "高新园": 0.0838, "白石洲": 0.057, "世界之窗": 0.0676, "华侨城": 0.0478, "侨城东": 0.0375, "竹子林": 0.0416, "车公庙": 0.0435, "香蜜湖": 0.031, "购物公园": 0.042, "会展中心": 0.0367, "岗厦": 0.0268, "华强路": 0.029, "科学馆": 0.0263, "大剧院": 0.0293, "老街": 0.0321, "国贸": 0.0257, "罗湖": 0.0302},
    "高新园": {"机场东": 0.0173, "后瑞": 0.0169, "固戍": 0.0207, "西乡": 0.0278, "坪洲": 0.025, "宝体": 0.0301, "宝安中心": 0.0397, "新安": 0.0391, "前海湾": 0.0469, "鲤鱼门": 0.0398, "大新": 0.0495, "桃园": 0.0527, "深大": 0.0871, "高新园": 0, "白石洲": 0.0703, "世界之窗": 0.082, "华侨城": 0.0573, "侨城东": 0.0445, "竹子林": 0.049, "车公庙": 0.0508, "香蜜湖": 0.0361, "购物公园": 0.0486, "会展中心": 0.0422, "岗厦": 0.0308, "华强路": 0.0331, "科学馆": 0.03, "大剧院": 0.0333, "老街": 0.0363, "国贸": 0.029, "罗湖": 0.0341},
    "白石洲": {"机场东": 0.0133, "后瑞": 0.013, "固戍": 0.0158, "西乡": 0.0213, "坪洲": 0.019, "宝体": 0.0228, "宝安中心": 0.0299, "新安": 0.0293, "前海湾": 0.0349, "鲤鱼门": 0.0294, "大新": 0.0361, "桃园": 0.038, "深大": 0.0618, "高新园": 0.0733, "白石洲": 0, "世界之窗": 0.0761, "华侨城": 0.0523, "侨城东": 0.0401, "竹子林": 0.0437, "车公庙": 0.0449, "香蜜湖": 0.0317, "购物公园": 0.0424, "会展中心": 0.0367, "岗厦": 0.0266, "华强路": 0.0285, "科学馆": 0.0257, "大剧院": 0.0285, "老街": 0.031, "国贸": 0.0247, "罗湖": 0.029},
    "世界之窗": {"机场东": 0.0167, "后瑞": 0.0163, "固戍": 0.0198, "西乡": 0.0265, "坪洲": 0.0236, "宝体": 0.0282, "宝安中心": 0.0368, "新安": 0.0358, "前海湾": 0.0425, "鲤鱼门": 0.0355, "大新": 0.0433, "桃园": 0.0452, "深大": 0.0725, "高新园": 0.0847, "白石洲": 0.0753, "世界之窗": 0, "华侨城": 0.0788, "侨城东": 0.0596, "竹子林": 0.0641, "车公庙": 0.0652, "香蜜湖": 0.0456, "购物公园": 0.0606, "会展中心": 0.0521, "岗厦": 0.0376, "华强路": 0.0401, "科学馆": 0.0361, "大剧院": 0.0398, "老街": 0.0432, "国贸": 0.0343, "罗湖": 0.0401},
    "华侨城": {"机场东": 0.012, "后瑞": 0.0117, "固戍": 0.0142, "西乡": 0.0189, "坪洲": 0.0168, "宝体": 0.02, "宝安中心": 0.026, "新安": 0.0252, "前海湾": 0.0297, "鲤鱼门": 0.0247, "大新": 0.030, "桃园": 0.031, "深大": 0.0492, "高新园": 0.0568, "白石洲": 0.0497, "世界之窗": 0.0757, "华侨城": 0, "侨城东": 0.0514, "竹子林": 0.0544, "车公庙": 0.0547, "香蜜湖": 0.0378, "购物公园": 0.0499, "会展中心": 0.0426, "岗厦": 0.0306, "华强路": 0.0324, "科学馆": 0.029, "大剧院": 0.0319, "老街": 0.0345, "国贸": 0.0273, "罗湖": 0.0319},
    "侨城东": {"机场东": 0.0097, "后瑞": 0.0094, "固戍": 0.0114, "西乡": 0.0151, "坪洲": 0.0134, "宝体": 0.0159, "宝安中心": 0.0206, "新安": 0.0199, "前海湾": 0.0234, "鲤鱼门": 0.0194, "大新": 0.0233, "桃园": 0.0239, "深大": 0.0377, "高新园": 0.043, "白石洲": 0.0372, "世界之窗": 0.0558, "华侨城": 0.0501, "侨城东": 0, "竹子林": 0.0524, "车公庙": 0.0518, "香蜜湖": 0.0354, "购物公园": 0.0462, "会展中心": 0.0391, "岗厦": 0.0279, "华强路": 0.0294, "科学馆": 0.0262, "大剧院": 0.0287, "老街": 0.0309, "国贸": 0.0244, "罗湖": 0.0283},
    "竹子林": {"机场东": 0.0116, "后瑞": 0.0113, "固戍": 0.0136, "西乡": 0.018, "坪洲": 0.0159, "宝体": 0.0189, "宝安中心": 0.0243, "新安": 0.0234, "前海湾": 0.0274, "鲤鱼门": 0.0226, "大新": 0.027, "桃园": 0.0276, "深大": 0.0431, "高新园": 0.0489, "白石洲": 0.0418, "世界之窗": 0.0619, "华侨城": 0.0547, "侨城东": 0.054, "竹子林": 0, "车公庙": 0.0739, "香蜜湖": 0.0498, "购物公园": 0.0641, "会展中心": 0.0537, "岗厦": 0.038, "华强路": 0.0398, "科学馆": 0.0352, "大剧院": 0.0384, "老街": 0.0411, "国贸": 0.0323, "罗湖": 0.0375},
    "车公庙": {"机场东": 0.0134, "后瑞": 0.013, "固戍": 0.0156, "西乡": 0.0207, "坪洲": 0.0182, "宝体": 0.0215, "宝安中心": 0.0277, "新安": 0.0266, "前海湾": 0.031, "鲤鱼门": 0.0254, "大新": 0.0303, "桃园": 0.0308, "深大": 0.0479, "高新园": 0.0539, "白石洲": 0.0457, "世界之窗": 0.067, "华侨城": 0.0585, "侨城东": 0.0568, "竹子林": 0.0786, "车公庙": 0, "香蜜湖": 0.0684, "购物公园": 0.0868, "会展中心": 0.0718, "岗厦": 0.0503, "华强路": 0.0522, "科学馆": 0.0459, "大剧院": 0.0497, "老街": 0.053, "国贸": 0.0415, "罗湖": 0.0479},
    "香蜜湖": {"机场东": 0.0092, "后瑞": 0.0088, "固戍": 0.0106, "西乡": 0.014, "坪洲": 0.0124, "宝体": 0.0146, "宝安中心": 0.0187, "新安": 0.0179, "前海湾": 0.0208, "鲤鱼门": 0.017, "大新": 0.0202, "桃园": 0.0204, "深大": 0.0316, "高新园": 0.0353, "白石洲": 0.0297, "世界之窗": 0.0433, "华侨城": 0.0374, "侨城东": 0.0359, "竹子林": 0.0489, "车公庙": 0.0631, "香蜜湖": 0, "购物公园": 0.0704, "会展中心": 0.0574, "岗厦": 0.0397, "华强路": 0.0408, "科学馆": 0.0355, "大剧院": 0.0382, "老街": 0.0405, "国贸": 0.0316, "罗湖": 0.0363},
    "购物公园": {"机场东": 0.0137, "后瑞": 0.0132, "固戍": 0.0158, "西乡": 0.0208, "坪洲": 0.0183, "宝体": 0.0215, "宝安中心": 0.0276, "新安": 0.0263, "前海湾": 0.0305, "鲤鱼门": 0.0249, "大新": 0.0294, "桃园": 0.0296, "深大": 0.0456, "高新园": 0.0508, "白石洲": 0.0425, "世界之窗": 0.0614, "华侨城": 0.0527, "侨城东": 0.05, "竹子林": 0.0673, "车公庙": 0.0856, "香蜜湖": 0.0752, "购物公园": 0, "会展中心": 0.1016, "岗厦": 0.0691, "华强路": 0.0702, "科学馆": 0.0606, "大剧院": 0.0646, "老街": 0.068, "国贸": 0.0527, "罗湖": 0.0602},
    "会展中心": {"机场东": 0.0127, "后瑞": 0.0123, "固戍": 0.0147, "西乡": 0.0194, "坪洲": 0.017, "宝体": 0.0199, "宝安中心": 0.0255, "新安": 0.0243, "前海湾": 0.028, "鲤鱼门": 0.0228, "大新": 0.0269, "桃园": 0.027, "深大": 0.0414, "高新园": 0.0459, "白石洲": 0.0382, "世界之窗": 0.0549, "华侨城": 0.0467, "侨城东": 0.044, "竹子林": 0.0586, "车公庙": 0.0736, "香蜜湖": 0.0637, "购物公园": 0.1056, "会展中心": 0, "岗厦": 0.0765, "华强路": 0.0765, "科学馆": 0.0651, "大剧院": 0.0688, "老街": 0.0718, "国贸": 0.0552, "罗湖": 0.0627},
    "岗厦": {"机场东": 0.0094, "后瑞": 0.0091, "固戍": 0.0109, "西乡": 0.0143, "坪洲": 0.0125, "宝体": 0.0147, "宝安中心": 0.0187, "新安": 0.0178, "前海湾": 0.0205, "鲤鱼门": 0.0166, "大新": 0.0196, "桃园": 0.0196, "深大": 0.0299, "高新园": 0.033, "白石洲": 0.0274, "世界之窗": 0.0391, "华侨城": 0.0331, "侨城东": 0.031, "竹子林": 0.0409, "车公庙": 0.0509, "香蜜湖": 0.0435, "购物公园": 0.0709, "会展中心": 0.0755, "岗厦": 0, "华强路": 0.0671, "科学馆": 0.0563, "大剧院": 0.0587, "老街": 0.0607, "国贸": 0.0462, "罗湖": 0.0522},
    "华强路": {"机场东": 0.0108, "后瑞": 0.0104, "固戍": 0.0124, "西乡": 0.0163, "坪洲": 0.0142, "宝体": 0.0167, "宝安中心": 0.0212, "新安": 0.0201, "前海湾": 0.0231, "鲤鱼门": 0.0187, "大新": 0.022, "桃园": 0.0219, "深大": 0.0334, "高新园": 0.0367, "白石洲": 0.0304, "世界之窗": 0.0432, "华侨城": 0.0363, "侨城东": 0.0338, "竹子林": 0.0443, "车公庙": 0.0547, "香蜜湖": 0.0463, "购物公园": 0.0745, "会展中心": 0.0781, "岗厦": 0.0694, "华强路": 0, "科学馆": 0.076, "大剧院": 0.0781, "老街": 0.0797, "国贸": 0.0601, "罗湖": 0.0673},
    "科学馆": {"机场东": 0.0102, "后瑞": 0.0098, "固戍": 0.0117, "西乡": 0.0153, "坪洲": 0.0134, "宝体": 0.0156, "宝安中心": 0.0198, "新安": 0.0188, "前海湾": 0.0216, "鲤鱼门": 0.0174, "大新": 0.0204, "桃园": 0.0203, "深大": 0.0308, "高新园": 0.0338, "白石洲": 0.0278, "世界之窗": 0.0394, "华侨城": 0.033, "侨城东": 0.0306, "竹子林": 0.0399, "车公庙": 0.0488, "香蜜湖": 0.041, "购物公园": 0.0653, "会展中心": 0.0676, "岗厦": 0.0591, "华强路": 0.0773, "科学馆": 0, "大剧院": 0.0869, "老街": 0.0873, "国贸": 0.0651, "罗湖": 0.0721},
    "大剧院": {"机场东": 0.0116, "后瑞": 0.0112, "固戍": 0.0133, "西乡": 0.0174, "坪洲": 0.0152, "宝体": 0.0177, "宝安中心": 0.0225, "新安": 0.0213, "前海湾": 0.0244, "鲤鱼门": 0.0197, "大新": 0.023, "桃园": 0.0228, "深大": 0.0346, "高新园": 0.0378, "白石洲": 0.0311, "世界之窗": 0.0439, "华侨城": 0.0366, "侨城东": 0.0337, "竹子林": 0.0437, "车公庙": 0.0532, "香蜜湖": 0.0444, "购物公园": 0.0702, "会展中心": 0.0719, "岗厦": 0.0621, "华强路": 0.0799, "科学馆": 0.0875, "大剧院": 0, "老街": 0.1179, "国贸": 0.0865, "罗湖": 0.0947},
    "老街": {"机场东": 0.0133, "后瑞": 0.0127, "固戍": 0.0152, "西乡": 0.0198, "坪洲": 0.0173, "宝体": 0.0201, "宝安中心": 0.0255, "新安": 0.0241, "前海湾": 0.0276, "鲤鱼门": 0.0222, "大新": 0.0259, "桃园": 0.0256, "深大": 0.0388, "高新园": 0.0423, "白石洲": 0.0346, "世界之窗": 0.0487, "华侨城": 0.0405, "侨城东": 0.0372, "竹子林": 0.048, "车公庙": 0.0582, "香蜜湖": 0.0482, "购物公园": 0.0757, "会展中心": 0.0769, "岗厦": 0.0658, "华强路": 0.0835, "科学馆": 0.09, "大剧院": 0.1208, "老街": 0, "国贸": 0.1163, "罗湖": 0.1253},
    "国贸": {"机场东": 0.0114, "后瑞": 0.0109, "固戍": 0.013, "西乡": 0.017, "坪洲": 0.0148, "宝体": 0.0172, "宝安中心": 0.0218, "新安": 0.0205, "前海湾": 0.0234, "鲤鱼门": 0.0188, "大新": 0.0219, "桃园": 0.0217, "深大": 0.0327, "高新园": 0.0356, "白石洲": 0.0291, "世界之窗": 0.0408, "华侨城": 0.0339, "侨城东": 0.031, "竹子林": 0.0398, "车公庙": 0.048, "香蜜湖": 0.0396, "购物公园": 0.0618, "会展中心": 0.0623, "岗厦": 0.0529, "华强路": 0.0665, "科学馆": 0.0708, "大剧院": 0.0935, "老街": 0.1227, "国贸": 0, "罗湖": 0.1266},
    "罗湖": {"机场东": 0.0151, "后瑞": 0.0145, "固戍": 0.0172, "西乡": 0.0225, "坪洲": 0.0195, "宝体": 0.0227, "宝安中心": 0.0287, "新安": 0.027, "前海湾": 0.0308, "鲤鱼门": 0.0247, "大新": 0.0288, "桃园": 0.0284, "深大": 0.0428, "高新园": 0.0464, "白石洲": 0.0378, "世界之窗": 0.053, "华侨城": 0.0438, "侨城东": 0.0399, "竹子林": 0.0512, "车公庙": 0.0615, "香蜜湖": 0.0504, "购物公园": 0.0784, "会展中心": 0.0786, "岗厦": 0.0662, "华强路": 0.0825, "科学馆": 0.087, "大剧院": 0.1135, "老街": 0.1466, "国贸": 0.1405, "罗湖": 0},
}

# ----------------------------
# 深圳地铁1号线（现实性改进）
# ----------------------------

# 站点类型定义：用于模拟潮汐流和不同的峰平比
LINE1_STATION_ROLES: Dict[str, str] = {
    "机场东": "Hub", "后瑞": "Residential", "固戍": "Residential", "西乡": "Residential", 
    "坪洲": "Residential", "宝体": "Residential", "宝安中心": "Hub", "新安": "Residential", 
    "前海湾": "Hub", "鲤鱼门": "Residential", "大新": "Residential", "桃园": "Residential", 
    "深大": "Employment", "高新园": "Employment", "白石洲": "Residential", "世界之窗": "Hub", 
    "华侨城": "Hub", "侨城东": "Employment", "竹子林": "Employment", "车公庙": "Employment", 
    "香蜜湖": "Employment", "购物公园": "Employment", "会展中心": "Employment", "岗厦": "Employment", 
    "华强路": "Employment", "科学馆": "Employment", "大剧院": "Hub", "老街": "Hub", 
    "国贸": "Hub", "罗湖": "Hub",
}

# 角色对应的峰平比 (典型范围 3.0 - 5.0+)
STATION_ROLE_PEAK_RATIO = {
    "Residential": 5.0,  # 居住区峰值极高
    "Employment": 3.0,   # 商务区峰值相对平缓但持续
    "Hub": 2.2,          # 交通枢纽全天流量较稳
}

def _build_line1_od_profile(station: str, index: int) -> Dict[str, float | int | str | Dict]:
    """构建深圳地铁1号线的OD配置（引入现实潮汐逻辑）"""
    role = LINE1_STATION_ROLES.get(station, "Residential")
    base_row = LINE1_OD_MATRIX.get(station, {})
    
    # 基础到达率 (平峰参考值，单位：人/小时)
    # 我们根据角色初始化一个基础强度
    intensity_map = {"Residential": 1200, "Employment": 800, "Hub": 1500}
    base_rate_offpeak = intensity_map.get(role, 1000)
    
    return {
        "line": "shenzhen_line1",
        "station": station,
        "role": role,
        "index": index,
        "base_rate_offpeak": base_rate_offpeak,
        "od_matrix_ref": LINE1_OD_MATRIX, # 这里保留全矩阵引用，逻辑由 passenger_flow 处理
    }

# ----------------------------
# 拓扑1：原简单环 + 内侧分支（一条环路 + 两个局部跨接）
# ----------------------------
def _build_topology1() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    d2 = Platform(8, "d2")
    s26 = Platform(14, "s26")
    s25 = Platform(13, "s25")
    s24 = Platform(12, "s24")
    s23 = Platform(11, "s23")
    s22 = Platform(10, "s22")
    s21 = Platform(9, "s21")

    # 一圈 + 中部斜连（这里用一条环路 + 两个局部跨接）
    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, d2, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )
    # 分支
    branch_1 = Route(2, "s15_to_s25_shortcut", [s15, s25], route_type=RouteType.NON_CIRCULAR)
    branch_2 = Route(3, "s22_to_s12_shortcut", [s22, s12], route_type=RouteType.NON_CIRCULAR)

    plats = [d1, s11, s12, s13, s14, s15, s16, d2, s26, s25, s24, s23, s22, s21]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    # 简单：每个平台独立为一站（可按需要再聚合）
    sts = [(p,) for p in plats]
    return plats, [outer_loop, branch_1, branch_2], sts

# ----------------------------
# 拓扑2：两个环 + 三条连接（抽象为两个循环+若干跳线）
# ----------------------------
def _build_topology2() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    d2 = Platform(9, "d2")

    s21 = Platform(10, "s21")
    s22 = Platform(11, "s22")
    s23 = Platform(12, "s23")
    s24 = Platform(13, "s24")
    s25 = Platform(14, "s25")
    s26 = Platform(15, "s26")
    s27 = Platform(16, "s27")

    


    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, d2, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )
    connect1 = Route(2, "s15_to_s17", [s15, s16, s17], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s17_to_s15", [s17, s16, s15], route_type=RouteType.NON_CIRCULAR)
    connect3 = Route(4, "s27_to_s25", [s27, s26, s25], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s25_to_s27", [s25, s26, s27], route_type=RouteType.NON_CIRCULAR)
    connect5 = Route(9, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)
    # 返流方向可选（如需要再添加反向 routes）

    plats = [d1, s11, s12, s13, s14, s15, s16, s17, d2, s21, s22, s23, s24, s25, s26, s27]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4, connect5], sts

# ----------------------------
# 拓扑4：车站数量灵敏度分析，五个车站，2个中间连接，10列车
# ----------------------------
def _build_topology4() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11, s12, (s13 del), s14->s13, s15->s14, s16->s15
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    
    d2 = Platform(7, "d2")
    
    # Bottom: s26->s25, s25->s24, s24->s23, (s23 del), s22, s21
    # Renumbering s21..s25
    s25 = Platform(12, "s25")
    s24 = Platform(11, "s24")
    s23 = Platform(10, "s23")
    s22 = Platform(9, "s22")
    s21 = Platform(8, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, d2, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)
    connect3 = Route(4, "s14_to_s24", [s14, s24], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s24_to_s14", [s24, s14], route_type=RouteType.NON_CIRCULAR)


    plats = [d1, s11, s12, s13, s14, s15, d2, s25, s24, s23, s22, s21]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4], sts

# ----------------------------
# 拓扑5：车站数量灵敏度分析，七个车站，2个中间连接，10列车
# ----------------------------
def _build_topology5() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11, s12, s13, NEW->s14, s14->s15, s15->s16, s16->s17
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    
    d2 = Platform(9, "d2")
    
    # Bottom: s21, s22, s23, NEW->s24, s24->s25, s25->s26, s26->s27
    s27 = Platform(16, "s27")
    s26 = Platform(15, "s26")
    s25 = Platform(14, "s25")
    s24 = Platform(13, "s24")
    s23 = Platform(12, "s23")
    s22 = Platform(11, "s22")
    s21 = Platform(10, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, d2, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)
    connect3 = Route(4, "s16_to_s26", [s16, s26], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s26_to_s16", [s26, s16], route_type=RouteType.NON_CIRCULAR)

    plats = [d1, s11, s12, s13, s14, s15, s16, s17, d2, s27, s26, s25, s24, s23, s22, s21]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4], sts

# ----------------------------
# 拓扑6：车站数量灵敏度分析，九个车站，2中间连接，10列车
# ----------------------------
def _build_topology6() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    s19 = Platform(10, "s19")
    
    d2 = Platform(11, "d2")
    
    # Bottom: s21, s22, s23, NEW1->s24, NEW2->s25, s24->s26, s25->s27, s26->s28
    s29 = Platform(20, "s29")
    s28 = Platform(19, "s28")
    s27 = Platform(18, "s27")
    s26 = Platform(17, "s26")
    s25 = Platform(16, "s25")
    s24 = Platform(15, "s24")
    s23 = Platform(14, "s23")
    s22 = Platform(13, "s22")
    s21 = Platform(12, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, d2, s29, s28, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)
    connect3 = Route(4, "s18_to_s28", [s18, s28], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s28_to_s18", [s28, s18], route_type=RouteType.NON_CIRCULAR)

    plats = [d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, d2, s29, s28, s27, s26, s25, s24, s23, s22, s21]
    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4], sts

# ----------------------------
# 拓扑7：列车数量灵敏度分析，八个车站，2中间连接，10列车
# ----------------------------
def _build_topology7() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11..s18
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    
    d2 = Platform(10, "d2")
    
    # Bottom: s21, s22, s23, NEW1->s24, NEW2->s25, NEW3->s26, s24->s27, s25->s28, s26->s29
    s28 = Platform(18, "s28")
    s27 = Platform(17, "s27")
    s26 = Platform(16, "s26")
    s25 = Platform(15, "s25")
    s24 = Platform(14, "s24")
    s23 = Platform(13, "s23")
    s22 = Platform(12, "s22")
    s21 = Platform(11, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, s18, d2, s28, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)
    connect3 = Route(4, "s17_to_s27", [s17, s27], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s27_to_s17", [s27, s17], route_type=RouteType.NON_CIRCULAR)

    plats = [d1, s11, s12, s13, s14, s15, s16, s17, s18, d2, s28, s27, s26, s25, s24, s23, s22, s21]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4], sts

# ----------------------------
# 拓扑8：列车数量灵敏度分析，八个车站，2中间连接，12列车
# ----------------------------
def _build_topology8() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11..s18
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    
    d2 = Platform(10, "d2")
    
    # Bottom: s21, s22, s23, NEW1->s24, NEW2->s25, NEW3->s26, s24->s27, s25->s28, s26->s29
    s28 = Platform(18, "s28")
    s27 = Platform(17, "s27")
    s26 = Platform(16, "s26")
    s25 = Platform(15, "s25")
    s24 = Platform(14, "s24")
    s23 = Platform(13, "s23")
    s22 = Platform(12, "s22")
    s21 = Platform(11, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, s18, d2, s28, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)
    connect3 = Route(4, "s17_to_s27", [s17, s27], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s27_to_s17", [s27, s17], route_type=RouteType.NON_CIRCULAR)

    plats = [d1, s11, s12, s13, s14, s15, s16, s17, s18, d2, s28, s27, s26, s25, s24, s23, s22, s21]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4], sts

# ----------------------------
# 拓扑9：列车数量灵敏度分析，八个车站，2中间连接，14列车
# ----------------------------
def _build_topology9() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11..s18
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    
    d2 = Platform(10, "d2")
    
    # Bottom: s21, s22, s23, NEW1->s24, NEW2->s25, NEW3->s26, s24->s27, s25->s28, s26->s29
    s28 = Platform(18, "s28")
    s27 = Platform(17, "s27")
    s26 = Platform(16, "s26")
    s25 = Platform(15, "s25")
    s24 = Platform(14, "s24")
    s23 = Platform(13, "s23")
    s22 = Platform(12, "s22")
    s21 = Platform(11, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, s18, d2, s28, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)
    connect3 = Route(4, "s17_to_s27", [s17, s27], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s27_to_s17", [s27, s17], route_type=RouteType.NON_CIRCULAR)


    plats = [d1, s11, s12, s13, s14, s15, s16, s17, s18, d2, s28, s27, s26, s25, s24, s23, s22, s21]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4], sts

# ----------------------------
# 拓扑10：辅助线数量灵敏度分析，9个车站，14列车，2中间连接
# ----------------------------
def _build_topology10() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    s19 = Platform(10, "s19")
    
    d2 = Platform(11, "d2")
    
    # Bottom: s21, s22, s23, NEW1->s24, NEW2->s25, s24->s26, s25->s27, s26->s28
    s29 = Platform(20, "s29")
    s28 = Platform(19, "s28")
    s27 = Platform(18, "s27")
    s26 = Platform(17, "s26")
    s25 = Platform(16, "s25")
    s24 = Platform(15, "s24")
    s23 = Platform(14, "s23")
    s22 = Platform(13, "s22")
    s21 = Platform(12, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, d2, s29, s28, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)
    connect3 = Route(4, "s18_to_s28", [s18, s28], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s28_to_s18", [s28, s18], route_type=RouteType.NON_CIRCULAR)

    plats = [d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, d2, s29, s28, s27, s26, s25, s24, s23, s22, s21]
    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4], sts

# ----------------------------
# 拓扑11：辅助线数量灵敏度分析，9个车站，14列车，4条中间连接
# ----------------------------
def _build_topology11() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    s19 = Platform(10, "s19")
    
    d2 = Platform(11, "d2")
    
    # Bottom: s21, s22, s23, NEW1->s24, NEW2->s25, s24->s26, s25->s27, s26->s28
    s29 = Platform(20, "s29")
    s28 = Platform(19, "s28")
    s27 = Platform(18, "s27")
    s26 = Platform(17, "s26")
    s25 = Platform(16, "s25")
    s24 = Platform(15, "s24")
    s23 = Platform(14, "s23")
    s22 = Platform(13, "s22")
    s21 = Platform(12, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, d2, s29, s28, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22_shortcut", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12_shortcut", [s22, s12], route_type=RouteType.NON_CIRCULAR)

    connect3 = Route(4, "s18_to_s28_shortcut", [s18, s28], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s28_to_s18_shortcut", [s28, s18], route_type=RouteType.NON_CIRCULAR)

    connect5 = Route(6, "s14_to_s24_shortcut", [s14, s24], route_type=RouteType.NON_CIRCULAR)
    connect6 = Route(7, "s24_to_s14_shortcut", [s24, s14], route_type=RouteType.NON_CIRCULAR)

    connect7 = Route(8, "s16_to_s26_shortcut", [s16, s26], route_type=RouteType.NON_CIRCULAR)
    connect8 = Route(9, "s26_to_s16_shortcut", [s26, s16], route_type=RouteType.NON_CIRCULAR)


    plats = [d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, d2, s29, s28, s27, s26, s25, s24, s23, s22, s21]
    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4, connect5, connect6, connect7, connect8], sts

# ----------------------------
# 拓扑12：辅助线数量灵敏度分析，9个车站，14列车，6条中间连接
# ----------------------------
def _build_topology12() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    s19 = Platform(10, "s19")
    
    d2 = Platform(11, "d2")
    
    # Bottom: s21, s22, s23, NEW1->s24, NEW2->s25, s24->s26, s25->s27, s26->s28
    s29 = Platform(20, "s29")
    s28 = Platform(19, "s28")
    s27 = Platform(18, "s27")
    s26 = Platform(17, "s26")
    s25 = Platform(16, "s25")
    s24 = Platform(15, "s24")
    s23 = Platform(14, "s23")
    s22 = Platform(13, "s22")
    s21 = Platform(12, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, d2, s29, s28, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22_shortcut", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12_shortcut", [s22, s12], route_type=RouteType.NON_CIRCULAR)

    connect3 = Route(4, "s13_to_s23_shortcut", [s13, s23], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s23_to_s13_shortcut", [s23, s13], route_type=RouteType.NON_CIRCULAR)

    connect5 = Route(6, "s14_to_s24_shortcut", [s14, s24], route_type=RouteType.NON_CIRCULAR)
    connect6 = Route(7, "s24_to_s14_shortcut", [s24, s14], route_type=RouteType.NON_CIRCULAR)

    connect7 = Route(10, "s16_to_s26_shortcut", [s16, s26], route_type=RouteType.NON_CIRCULAR)
    connect8 = Route(11, "s26_to_s16_shortcut", [s26, s16], route_type=RouteType.NON_CIRCULAR)

    connect9 = Route(12, "s17_to_s27_shortcut", [s17, s27], route_type=RouteType.NON_CIRCULAR)
    connect10 = Route(13, "s27_to_s17_shortcut", [s27, s17], route_type=RouteType.NON_CIRCULAR)

    connect11 = Route(14, "s18_to_s28_shortcut", [s18, s28], route_type=RouteType.NON_CIRCULAR)
    connect12 = Route(15, "s28_to_s18_shortcut", [s28, s18], route_type=RouteType.NON_CIRCULAR)


    plats = [d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, d2, s29, s28, s27, s26, s25, s24, s23, s22, s21]
    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4, connect5, connect6, connect7, connect8, connect9, connect10, connect11, connect12], sts

# ----------------------------
# 拓扑13：Urban Branch, 8个车站，12列车，3条中间连接
# ----------------------------
def _build_topology13() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11..s18
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")

    d2 = Platform(10, "d2")

    # Bottom: s21..s28
    s28 = Platform(18, "s28")
    s27 = Platform(17, "s27")
    s26 = Platform(16, "s26")
    s25 = Platform(15, "s25")
    s24 = Platform(14, "s24")
    s23 = Platform(13, "s23")
    s22 = Platform(12, "s22")
    s21 = Platform(11, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [d1, s11, s12, s13, s14, s15, s16, s17, s18, d2, s28, s27, s26, s25, s24, s23, s22, s21],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)

    connect3 = Route(4, "s14_to_s24", [s14, s24], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s24_to_s14", [s24, s14], route_type=RouteType.NON_CIRCULAR)

    connect5 = Route(12, "s17_to_s27", [s17, s27], route_type=RouteType.NON_CIRCULAR)
    connect6 = Route(13, "s27_to_s17", [s27, s17], route_type=RouteType.NON_CIRCULAR)

    plats = [d1, s11, s12, s13, s14, s15, s16, s17, s18, d2, s28, s27, s26, s25, s24, s23, s22, s21]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4, connect5, connect6], sts


# ----------------------------
# 拓扑14：Suburban Express, 13个车站，20列车，4条中间连接
# ----------------------------
def _build_topology14() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11..s113
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    s19 = Platform(10, "s19")
    s110 = Platform(11, "s110")
    s111 = Platform(12, "s111")
    s112 = Platform(13, "s112")
    s113 = Platform(14, "s113")

    d2 = Platform(15, "d2")

    # Bottom: s21..s213
    s213 = Platform(28, "s213")
    s212 = Platform(27, "s212")
    s211 = Platform(26, "s211")
    s210 = Platform(25, "s210")
    s29 = Platform(24, "s29")
    s28 = Platform(23, "s28")
    s27 = Platform(22, "s27")
    s26 = Platform(21, "s26")
    s25 = Platform(20, "s25")
    s24 = Platform(19, "s24")
    s23 = Platform(18, "s23")
    s22 = Platform(17, "s22")
    s21 = Platform(16, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [
            d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, s110, s111, s112, s113,
            d2, s213, s212, s211, s210, s29, s28, s27, s26, s25, s24, s23, s22, s21,
        ],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)

    connect3 = Route(4, "s15_to_s25", [s15, s25], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s25_to_s15", [s25, s15], route_type=RouteType.NON_CIRCULAR)

    connect5 = Route(12, "s18_to_s28", [s18, s28], route_type=RouteType.NON_CIRCULAR)
    connect6 = Route(13, "s28_to_s18", [s28, s18], route_type=RouteType.NON_CIRCULAR)

    connect7 = Route(16, "s112_to_s212", [s112, s212], route_type=RouteType.NON_CIRCULAR)
    connect8 = Route(17, "s212_to_s112", [s212, s112], route_type=RouteType.NON_CIRCULAR)
    plats = [
        d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, s110, s111, s112, s113,
        d2, s213, s212, s211, s210, s29, s28, s27, s26, s25, s24, s23, s22, s21,
    ]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4, connect5, connect6, connect7, connect8], sts


# ----------------------------
# 拓扑15：Mainstream Commuter, 辅助线数量灵敏度分析，24个车站，30列车，6条中间连接
# ----------------------------
def _build_topology15() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11..s124
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    s19 = Platform(10, "s19")
    s110 = Platform(11, "s110")
    s111 = Platform(12, "s111")
    s112 = Platform(13, "s112")
    s113 = Platform(14, "s113")
    s114 = Platform(15, "s114")
    s115 = Platform(16, "s115")
    s116 = Platform(17, "s116")
    s117 = Platform(18, "s117")
    s118 = Platform(19, "s118")
    s119 = Platform(20, "s119")
    s120 = Platform(21, "s120")
    s121 = Platform(22, "s121")
    s122 = Platform(23, "s122")
    s123 = Platform(24, "s123")
    s124 = Platform(25, "s124")

    d2 = Platform(26, "d2")

    # Bottom: s21..s224
    s224 = Platform(50, "s224")
    s223 = Platform(49, "s223")
    s222 = Platform(48, "s222")
    s221 = Platform(47, "s221")
    s220 = Platform(46, "s220")
    s219 = Platform(45, "s219")
    s218 = Platform(44, "s218")
    s217 = Platform(43, "s217")
    s216 = Platform(42, "s216")
    s215 = Platform(41, "s215")
    s214 = Platform(40, "s214")
    s213 = Platform(39, "s213")
    s212 = Platform(38, "s212")
    s211 = Platform(37, "s211")
    s210 = Platform(36, "s210")
    s29 = Platform(35, "s29")
    s28 = Platform(34, "s28")
    s27 = Platform(33, "s27")
    s26 = Platform(32, "s26")
    s25 = Platform(31, "s25")
    s24 = Platform(30, "s24")
    s23 = Platform(29, "s23")
    s22 = Platform(28, "s22")
    s21 = Platform(27, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [
            d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, s110, s111, s112, s113, s114, s115, s116, s117,
            s118, s119, s120, s121, s122, s123, s124, d2, s224, s223, s222, s221, s220, s219, s218, s217, s216,
            s215, s214, s213, s212, s211, s210, s29, s28, s27, s26, s25, s24, s23, s22, s21,
        ],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)

    connect3 = Route(4, "s16_to_s26", [s16, s26], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s26_to_s16", [s26, s16], route_type=RouteType.NON_CIRCULAR)

    connect5 = Route(8, "s110_to_s210", [s110, s210], route_type=RouteType.NON_CIRCULAR)
    connect6 = Route(9, "s210_to_s110", [s210, s110], route_type=RouteType.NON_CIRCULAR)

    connect7 = Route(16, "s114_to_s214", [s114, s214], route_type=RouteType.NON_CIRCULAR)
    connect8 = Route(17, "s214_to_s114", [s214, s114], route_type=RouteType.NON_CIRCULAR)

    connect9 = Route(20, "s118_to_s218", [s118, s218], route_type=RouteType.NON_CIRCULAR)
    connect10 = Route(21, "s218_to_s118", [s218, s118], route_type=RouteType.NON_CIRCULAR)

    connect11 = Route(24, "s123_to_s223", [s123, s223], route_type=RouteType.NON_CIRCULAR)
    connect12 = Route(25, "s223_to_s123", [s223, s123], route_type=RouteType.NON_CIRCULAR)

    plats = [
        d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, s110, s111, s112, s113, s114, s115, s116, s117,
        s118, s119, s120, s121, s122, s123, s124, d2, s224, s223, s222, s221, s220, s219, s218, s217, s216,
        s215, s214, s213, s212, s211, s210, s29, s28, s27, s26, s25, s24, s23, s22, s21,
    ]

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [outer_loop, connect1, connect2, connect3, connect4, connect5, connect6, connect7, connect8, connect9, connect10, connect11, connect12], sts


# ----------------------------
# 拓扑16：Core Artery，30个车站，45列车，10条中间连接
# ----------------------------
def _build_topology16() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    d1 = Platform(1, "d1")
    # Top: s11..s130
    s11 = Platform(2, "s11")
    s12 = Platform(3, "s12")
    s13 = Platform(4, "s13")
    s14 = Platform(5, "s14")
    s15 = Platform(6, "s15")
    s16 = Platform(7, "s16")
    s17 = Platform(8, "s17")
    s18 = Platform(9, "s18")
    s19 = Platform(10, "s19")
    s110 = Platform(11, "s110")
    s111 = Platform(12, "s111")
    s112 = Platform(13, "s112")
    s113 = Platform(14, "s113")
    s114 = Platform(15, "s114")
    s115 = Platform(16, "s115")
    s116 = Platform(17, "s116")
    s117 = Platform(18, "s117")
    s118 = Platform(19, "s118")
    s119 = Platform(20, "s119")
    s120 = Platform(21, "s120")
    s121 = Platform(22, "s121")
    s122 = Platform(23, "s122")
    s123 = Platform(24, "s123")
    s124 = Platform(25, "s124")
    s125 = Platform(26, "s125")
    s126 = Platform(27, "s126")
    s127 = Platform(28, "s127")
    s128 = Platform(29, "s128")
    s129 = Platform(30, "s129")
    s130 = Platform(31, "s130")

    d2 = Platform(32, "d2")

    # Bottom: s21..s230
    s230 = Platform(62, "s230")
    s229 = Platform(61, "s229")
    s228 = Platform(60, "s228")
    s227 = Platform(59, "s227")
    s226 = Platform(58, "s226")
    s225 = Platform(57, "s225")
    s224 = Platform(56, "s224")
    s223 = Platform(55, "s223")
    s222 = Platform(54, "s222")
    s221 = Platform(53, "s221")
    s220 = Platform(52, "s220")
    s219 = Platform(51, "s219")
    s218 = Platform(50, "s218")
    s217 = Platform(49, "s217")
    s216 = Platform(48, "s216")
    s215 = Platform(47, "s215")
    s214 = Platform(46, "s214")
    s213 = Platform(45, "s213")
    s212 = Platform(44, "s212")
    s211 = Platform(43, "s211")
    s210 = Platform(42, "s210")
    s29 = Platform(41, "s29")
    s28 = Platform(40, "s28")
    s27 = Platform(39, "s27")
    s26 = Platform(38, "s26")
    s25 = Platform(37, "s25")
    s24 = Platform(36, "s24")
    s23 = Platform(35, "s23")
    s22 = Platform(34, "s22")
    s21 = Platform(33, "s21")

    outer_loop = Route(
        1,
        "outer_loop",
        [
            d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, s110, s111, s112, s113, s114, s115, s116, s117,
            s118, s119, s120, s121, s122, s123, s124, s125, s126, s127, s128, s129, s130, d2, s230, s229, s228,
            s227, s226, s225, s224, s223, s222, s221, s220, s219, s218, s217, s216, s215, s214, s213, s212, s211,
            s210, s29, s28, s27, s26, s25, s24, s23, s22, s21,
        ],
        route_type=RouteType.CIRCULAR,
    )

    connect1 = Route(2, "s12_to_s22", [s12, s22], route_type=RouteType.NON_CIRCULAR)
    connect2 = Route(3, "s22_to_s12", [s22, s12], route_type=RouteType.NON_CIRCULAR)

    connect3 = Route(4, "s15_to_s25", [s15, s25], route_type=RouteType.NON_CIRCULAR)
    connect4 = Route(5, "s25_to_s15", [s25, s15], route_type=RouteType.NON_CIRCULAR)

    connect5 = Route(8, "s18_to_s28", [s18, s28], route_type=RouteType.NON_CIRCULAR)
    connect6 = Route(9, "s28_to_s18", [s28, s18], route_type=RouteType.NON_CIRCULAR)

    connect7 = Route(12, "s111_to_s211", [s111, s211], route_type=RouteType.NON_CIRCULAR)
    connect8 = Route(13, "s211_to_s111", [s211, s111], route_type=RouteType.NON_CIRCULAR)

    connect9 = Route(16, "s113_to_s213", [s113, s213], route_type=RouteType.NON_CIRCULAR)
    connect10 = Route(17, "s213_to_s113", [s213, s113], route_type=RouteType.NON_CIRCULAR)

    connect11 = Route(24, "s116_to_s216", [s116, s216], route_type=RouteType.NON_CIRCULAR)
    connect12 = Route(25, "s216_to_s116", [s216, s116], route_type=RouteType.NON_CIRCULAR)

    connect13 = Route(28, "s119_to_s219", [s119, s219], route_type=RouteType.NON_CIRCULAR)
    connect14 = Route(29, "s219_to_s119", [s219, s119], route_type=RouteType.NON_CIRCULAR)

    connect15 = Route(32, "s122_to_s222", [s122, s222], route_type=RouteType.NON_CIRCULAR)
    connect16 = Route(33, "s222_to_s122", [s222, s122], route_type=RouteType.NON_CIRCULAR)

    connect17 = Route(36, "s125_to_s225", [s125, s225], route_type=RouteType.NON_CIRCULAR)
    connect18 = Route(37, "s225_to_s125", [s225, s125], route_type=RouteType.NON_CIRCULAR)

    connect19 = Route(40, "s129_to_s229", [s129, s229], route_type=RouteType.NON_CIRCULAR)
    connect20 = Route(41, "s229_to_s129", [s229, s129], route_type=RouteType.NON_CIRCULAR)
    plats = [
        d1, s11, s12, s13, s14, s15, s16, s17, s18, s19, s110, s111, s112, s113, s114, s115, s116, s117,
        s118, s119, s120, s121, s122, s123, s124, s125, s126, s127, s128, s129, s130, d2, s230, s229, s228,
        s227, s226, s225, s224, s223, s222, s221, s220, s219, s218, s217, s216, s215, s214, s213, s212, s211,
        s210, s29, s28, s27, s26, s25, s24, s23, s22, s21,
    ]

    # 为 topology16 注入深圳地铁1号线近似OD配置
    # 站台映射约定：s11 为罗湖下行，s21 为罗湖上行；
    # 因此 s11..s130 对应“下行”(罗湖 -> 机场东)，s230..s21 对应“上行”(机场东 -> 罗湖)。
    line1_order = LINE1_STATION_ORDER  # 机场东 -> 罗湖
    down_platforms = [
        s11, s12, s13, s14, s15, s16, s17, s18, s19, s110, s111, s112, s113, s114, s115, s116, s117, s118,
        s119, s120, s121, s122, s123, s124, s125, s126, s127, s128, s129, s130,
    ]
    up_platforms = [
        s230, s229, s228, s227, s226, s225, s224, s223, s222, s221, s220, s219, s218, s217, s216, s215, s214,
        s213, s212, s211, s210, s29, s28, s27, s26, s25, s24, s23, s22, s21,
    ]

    if len(line1_order) == len(up_platforms) == len(down_platforms):
        rev_order = list(reversed(line1_order))  # 罗湖 -> 机场东
        for idx, p in enumerate(down_platforms):
            station = rev_order[idx]
            station_idx = line1_order.index(station)
            p.station_name = f"{station}下行"
            p.od_profile = _build_line1_od_profile(station, station_idx)

        for idx, p in enumerate(up_platforms):
            station = line1_order[idx]
            p.station_name = f"{station}上行"
            p.od_profile = _build_line1_od_profile(station, idx)

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d"):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in plats:
        p.node_type = infer_node_type(p)

    sts = [(p,) for p in plats]
    return plats, [
        outer_loop, connect1, connect2, connect3, connect4, connect5, connect6, connect7, connect8, connect9,
        connect10, connect11, connect12, connect13, connect14, connect15, connect16, connect17, connect18,
        connect19, connect20,
    ], sts

# ----------------------------
# 拓扑3：深圳地铁1号线
# 合并主线为两条环形 route（上行/下行），其余全部按单线段辅助 route
# ----------------------------
def _build_topology3() -> Tuple[List[Platform], List[Route], List[Tuple[Platform, ...]]]:
    def P(pid, name, dwell=30): return Platform(pid, name, dwell)
    # 上行 / 上行相关
    s11 = P (1, "机场东上行折返线", 10)
    s12 = P (2, "机场东上行")
    s13 = P (3, "后瑞上行")
    s14 = P (4, "固戍上行")
    s15 = P (5, "固戍备用线", 10)
    s16 = P (6, "西乡上行")
    s17 = P (7, "坪洲上行")
    s18 = P (8, "宝体上行")
    s19 = P (9, "宝安中心上行")
    s20 = P (10, "新安上行")
    s21 = P (11, "前海出段线", 10)
    s22 = P (12, "前海入段线", 10)
    s23 = P (13, "前海湾上行")
    s24 = P (14, "鲤鱼门上行")
    s25 = P (15, "大新上行")
    s26 = P (16, "桃园上行")
    s27 = P (17, "深大上行")
    s28 = P (18, "深大存车线", 10)
    s29 = P (19, "高新园上行")
    s30 = P (20, "白石洲上行")
    s31 = P (21, "世界之窗上行")
    s32 = P (22, "华侨城上行")
    s33 = P (23, "侨城东上行")
    s34 = P (24, "竹子林上行")
    s35 = P (25, "竹子林备用线", 10)
    s36 = P (26, "竹子林出段线", 10)
    s37 = P (27, "竹子林入段线", 10)
    s38 = P (28, "车公庙上行")
    s39 = P (29, "香蜜湖上行")
    s40 = P (30, "购物公园上行")
    s41 = P (31, "会展中心上行")
    s42 = P (32, "岗厦上行")
    s43 = P (33, "华强路上行")
    s44 = P (34, "科学馆上行")
    s45 = P (35, "大剧院上行")
    s46 = P (36, "老街上行")
    s47 = P (37, "国贸上行")
    s48 = P (38, "罗湖 1 道")
    s49 = P (39, "罗湖 2 道")
    # 下行
    s50 = P (40, "罗湖 3 道")
    s51 = P (41, "国贸下行")
    s52 = P (42, "老街下行")
    s53 = P (43, "大剧院下行")
    s54 = P (44, "科学馆下行")
    s55 = P (45, "华强路下行")
    s56 = P (46, "岗厦下行")
    s57 = P (47, "会展中心下行")
    s58 = P (48, "购物公园下行")
    s59 = P (49, "香蜜湖下行")
    s60 = P (50, "车公庙下行")
    s61 = P (51, "竹子林下行")
    s62 = P (52, "侨城东下行")
    s63 = P (53, "华侨城下行")
    s64 = P (54, "世界之窗下行")
    s65 = P (55, "白石洲下行")
    s66 = P (56, "高新园下行")
    s67 = P (57, "深大下行")
    s68 = P (58, "桃园下行")
    s69 = P (59, "大新下行")
    s70 = P (60, "鲤鱼门下行")
    s71 = P (61, "前海湾下行")
    s72 = P (62, "新安下行")
    s73 = P (63, "宝安中心下行")
    s74 = P (64, "宝体下行")
    s75 = P (65, "坪洲下行")
    s76 = P (66, "西乡下行")
    s77 = P (67, "固戍下行")
    s78 = P (68, "后瑞下行")
    s79 = P (69, "机场东下行")
    s80 = P (70, "机场东下行折返线", 10)
    # 道岔 / 连接节点（保持 ID）
    d11 = P (71, "罗湖 1 道连接上行线", 10); d12 = P (72, "罗湖交叉渡线 1", 10); d13 = P (73, "罗湖交叉渡线 2", 10)
    d14 = P (74, "罗湖交叉渡线 3", 10); d15 = P (75, "罗湖交叉渡线 4", 10)
    d16 = P (76, "大剧院渡线 1", 10); d17 = P (77, "大剧院渡线 2", 10)
    d18 = P (78, "会展中心上行连接存车线 1", 10); d19 = P (79, "会展中心上行连接存车线 2", 10)
    d20 = P (80, "会展中心上行连接存车线 3", 10); d21 = P (81, "会展中心上行连接存车线 4", 10)
    d22 = P (82, "会展中心下行连接存车线 1", 10); d23 = P (83, "会展中心下行连接存车线 2", 10)
    d24 = P (84, "会展中心下行连接存车线 3", 10); d25 = P (85, "会展中心下行连接存车线 4", 10)
    d26 = P (86, "会展中心渡线 1", 10); d27 = P (87, "会展中心渡线 2", 10)
    d28 = P (88, "竹子林上行连接备用线", 10); d29 = P (89, "竹子林下行连接备用线 1", 10)
    d30 = P (90, "竹子林下行连接备用线 2", 10); d31 = P (91, "竹子林上行连接出段线 1", 10)
    d32 = P (92, "竹子林上行连接出段线 2", 10); d33 = P (93, "竹子林上行连接出段线 3", 10)
    d34 = P (94, "竹子林上行连接出段线 4", 10); d35 = P (95, "竹子林下行连接入段线 1", 10)
    d36 = P (96, "竹子林下行连接入段线 2", 10); d37 = P (97, "竹子林下行连接入段线 3", 10)
    d38 = P (98, "竹子林下行连接入段线 4", 10); d39 = P (99, "竹子林出段线连接备用线", 10)
    d40 = P (100, "世界之窗渡线 1", 10); d41 = P (101, "世界之窗渡线 2", 10)
    d42 = P (102, "深大上行连接存车线 1", 10); d43 = P (103, "深大上行连接存车线 2", 10)
    d44 = P (104, "深大上行连接存车线 3", 10); d45 = P (105, "深大上行连接存车线 4", 10)
    d46 = P (106, "深大下行连接存车线 1", 10); d47 = P (107, "深大下行连接存车线 2", 10)
    d48 = P (108, "前海湾上行连接入段线 1", 10); d49 = P (109, "前海湾上行连接入段线 2", 10)
    d50 = P (110, "前海湾下行连接入段线 1", 10); d51 = P (111, "前海湾下行连接入段线 2", 10)
    d52 = P (112, "新安上行连接出段线 1", 10); d53 = P (113, "新安上行连接出段线 2", 10)
    d54 = P (114, "新安下行连接出段线 1", 10); d55 = P (115, "新安下行连接出段线 2", 10)
    d56 = P (116, "宝体渡线 1", 10); d57 = P (117, "宝体渡线 2", 10)
    d58 = P (118, "西乡上行连接折返线 1", 10); d59 = P (119, "西乡上行连接折返线 2", 10)
    d60 = P (120, "西乡上行连接折返线 3", 10); d61 = P (121, "西乡上行连接折返线 4", 10)
    d62 = P (122, "西乡下行连接折返线 1", 10); d63 = P (123, "西乡下行连接折返线 2", 10)
    d64 = P (124, "固戍上行连接备用线 1", 10); d65 = P (125, "固戍上行连接备用线 2", 10)
    d66 = P (126, "固戍上行连接备用线 3", 10); d67 = P (127, "固戍上行连接备用线 4", 10)
    d68 = P (128, "固戍下行连接备用线 1", 10); d69 = P (129, "固戍下行连接备用线 2", 10)
    d70 = P (130, "固戍下行连接备用线 3", 10); d71 = P (131, "固戍下行连接备用线 4", 10)
    d72 = P (132, "机场东交叉渡线 1", 10); d73 = P (133, "机场东交叉渡线 2", 10)
    d74 = P (134, "机场东交叉渡线 3", 10); d75 = P (135, "机场东交叉渡线 4", 10)

    # 上行交路（机场东到罗湖）
    up_main_seq = [
        s11, d73, d72, s12, s13, d66, s14, d64, d60, s16, d58, s17, d56, s18,
        s19, s20, d52, d48, s23, s24, s25, s26, d44, d42, s27, s29, s30, d40,
        s31, s32, s33, d33, d31, s34, d28, s38, s39, s40, d26, s41, d20, d18, 
        s42, s43, s44, d16, s45, s46, s47, d13, d12, d11, s48, d11, s49
    ]
    # 下行交路（罗湖到机场东）
    down_main_seq = [
        s50, d14, d15, s51, s52, s53, d17, s54, s55, s56, d22, d24, s57, d27,
        s58, s59, s60, d29, s61, d35, d37, s62, s63, s64, d41, s65, s66, s67,
        d46, s68, s69, s70, s71, d50, d54, s72, s73, s74, d57, s75, d62, s76,
        d68, s77, d70, s78, s79, d74, d75, s80
    ]

    routes: List[Route] = []
    rid = 1
    routes.append(Route(rid, "main_up_ring", up_main_seq, route_type=RouteType.NON_CIRCULAR)); rid += 1
    routes.append(Route(rid, "main_down_ring", down_main_seq, route_type=RouteType.NON_CIRCULAR)); rid += 1


    # 辅助线route
    routes.append(Route(rid, "luohu1_arrive", [d11, s48], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 罗湖1道接车进路
    routes.append(Route(rid, "luohu1_depart", [s48, d11, d12, d15], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 罗湖1道发车进路
    routes.append(Route(rid, "luohu2_arrive", [d15, d12, d11, s49], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 罗湖2道接车进路
    routes.append(Route(rid, "luohu2_depart", [s49, d11, d12, d15], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 罗湖2道发车进路
    routes.append(Route(rid, "luohu3_arrive", [d13, d14, s50], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 罗湖3道接车进路
    routes.append(Route(rid, "luohu3_depart", [s50, d14, d13], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 罗湖3道发车进路
    routes.append(Route(rid, "dajuyuan_turnback", [d17, d16], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 大剧院渡线折返
    routes.append(Route(rid, "huizhanzhongxin_turnback_up", [d20, d21, d25, d24], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 大剧院上行折返
    routes.append(Route(rid, "huizhanzhongxin_turnback_down", [d22, d23, d19, d18], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 大剧院下行折返
    routes.append(Route(rid, "zhuzilin_turnback_down", [d29, d30, s35, d30, d28], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 竹子林下行折返
    routes.append(Route(rid, "zhuzilin_turnback_up", [d33, d31, d32, d36, d37], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 竹子林上行折返
    routes.append(Route(rid, "shijiezhichuang_turnback", [d41, d40], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 世界之窗渡线折返
    routes.append(Route(rid, "shenda_turnback", [d46, d47, d43, d42], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 深大存车线折返
    routes.append(Route(rid, "qianhaiwan_turnback", [d50, d51, d49, d48], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 前海湾入段线折返
    routes.append(Route(rid, "xinan_turnback", [d52, d53, d55 ,d54], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 新安出段线折返
    routes.append(Route(rid, "baoti_turnback", [d56, d57], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 宝体渡线折返
    routes.append(Route(rid, "xixiang_turnback_up", [d58, d59, d63, d62], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 西乡折返线上行折返
    routes.append(Route(rid, "baoti_turnback_down", [d62, d63, d61, d60], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 西乡折返线下行折返
    routes.append(Route(rid, "gushu_turnback_up", [d66, d67, d71, d70], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 固戍备用线上行折返
    routes.append(Route(rid, "gushu_turnback_down", [d68, d69, d65, d64], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 固戍备用线下行折返
    routes.append(Route(rid, "jichangdong_depart", [s80, d75, d72, d73, s11], route_type=RouteType.NON_CIRCULAR)); rid += 1 # 机场东上行发车进路



    platforms_all = [
        s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29,
        s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48,
        s49, s50, s51, s52, s53, s54, s55, s56, s57, s58, s59, s60, s61, s62, s63, s64, s65, s66, s67,
        s68, s69, s70, s71, s72, s73, s74, s75, s76, s77, s78, s79, s80,
        d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29,
        d30, d31, d32, d33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48,
        d49, d50, d51, d52, d53, d54, d55, d56, d57, d58, d59, d60, d61, d62, d63, d64, d65, d66, d67,
        d68, d69, d70, d71, d72, d73, d74, d75,
    ]

    # 自动构建 stations：聚合主线“上行/下行/道”平台（忽略含关键字的辅助节点）
    aux_kw = re.compile(r"(连接|渡线|出段线|入段线|备用线|存车线|交叉渡线|折返线)")
    base_map: Dict[str, List[Platform]] = {}
    def base_name(n: str) -> str:
        n2 = re.sub(r"\s*\d+\s*道", " 道", n)
        n2 = n2.replace("上行", "").replace("下行", "")
        n2 = re.sub(r"\s*道$", " 道", n2)  # 统一“罗湖 道”样式
        n2 = re.sub(r"\s+", "", n2)
        return n2

    def infer_node_type(p: Platform) -> NodeType:
        if p.name.startswith("d") or re.search(r"(渡线|连接|出段线|入段线|备用线|存车线|交叉渡线|折返线)", p.name):
            return NodeType.SWITCH
        return NodeType.PLATFORM

    for p in platforms_all:
        p.node_type = infer_node_type(p)
        if aux_kw.search(p.name):
            continue
        if "道连接" in p.name:  # 排除“罗湖 1 道连接上行线”
            continue
        b = base_name(p.name)
        base_map.setdefault(b, []).append(p)

    # “罗湖”特殊：把 1/2/3 道都并入同站
    # 上一步 base_name 会把“罗湖1道/2道/3道”统一为“罗湖道” => 用它再合并
    stations_list: List[Tuple[Platform, ...]] = []
    for b, ps in base_map.items():
        stations_list.append(tuple(sorted(ps, key=lambda x: x.id)))

    return platforms_all, routes, stations_list

def get_platform_by_id(platform_id):
    """Find a platform by its id from the platforms list."""
    for p in platforms:
        if p.id == platform_id:
            return p
    return None

# ----------------------------
# 中断管理器（保持接口）
# ----------------------------
from delay_events import DelayEvents


class LineDisruption(DelayEvents):
    """Shared event manager used by the existing simulation and MILP solver."""



disruption_manager = LineDisruption()

# ----------------------------
# 统一拓扑加载
# ----------------------------
_TOPOLOGY_BUILDERS = {
    1: _build_topology1,
    2: _build_topology2,
    3: _build_topology3,
    4: _build_topology4,
    5: _build_topology5,
    6: _build_topology6,
    7: _build_topology7,
    8: _build_topology8,
    9: _build_topology9,
    10: _build_topology10,
    11: _build_topology11,
    12: _build_topology12,
    13: _build_topology13,
    14: _build_topology14,
    15: _build_topology15,
    16: _build_topology16,
}
from topology.metro_lines import LINE_SPECS, build_line
_TOPOLOGY_BUILDERS.update({tid: (lambda tid=tid: build_line(tid)) for tid in LINE_SPECS})


def load_topology(topology_id: int):
    """选择拓扑，填充全局 platforms / routes / stations，line_segments 置空（交由其它流程动态生成）"""
    global ACTIVE_TOPOLOGY, platforms, routes, stations, line_segments
    if topology_id not in _TOPOLOGY_BUILDERS:
        raise ValueError(f"未知拓扑: {topology_id}")

    result = _TOPOLOGY_BUILDERS[topology_id]()
    # 兼容旧格式（2元组）
    if len(result) == 2:
        plats, rts = result  # type: ignore
        sts = [(p,) for p in plats]
    else:
        plats, rts, sts = result  # type: ignore

    platforms.clear()
    platforms.extend(plats)

    routes.clear(); routes.extend(rts)

    stations.clear(); stations.extend(sts)

    line_segments.clear()  # 动态推导
    ACTIVE_TOPOLOGY = topology_id
    set_passenger_whitelist(topology_id)
    return


def calc_platform_positions(platforms, 
                            horizontal_platform_interval, 
                            vertical_platform_interval, 
                            start_positon = (0, 0),
                            offset = (0, 0),
                            topology_id: int | None = None):
    """Calculate the positions of the platforms"""
    topology_id = topology_id or ACTIVE_TOPOLOGY or int(os.environ.get("METRO_TOPOLOGY", 1))

    if topology_id in LINE_SPECS:
        from topology.metro_lines import depot_sites
        sx, sy = start_positon[0] + offset[0], start_positon[1] + offset[1]
        depot_positions = {site["node_id"]: sum(site["station_indices"])/len(site["station_indices"])
                           for site in depot_sites(topology_id)}
        return {p: (sx + (int(p.name.split("_")[1]) if p.name.startswith("s") else depot_positions[p.id])*horizontal_platform_interval,
                    sy + (vertical_platform_interval if p.name.startswith("s1_") else
                          -vertical_platform_interval if p.name.startswith("s2_") else -3*vertical_platform_interval))
                for p in platforms}

    if topology_id in (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        ox, oy = offset
        def add_off(x, y): return (x + ox, y + oy)
        positions = {}
        for p in platforms:
            if p.name == "d1":
                positions[p] = add_off(start_positon[0], start_positon[1])
            elif p.name == "d2":
                positions[p] = add_off(start_positon[0] + (len(platforms) / 2) * horizontal_platform_interval,
                                       start_positon[1])
            elif p.name.startswith("s1"):
                positions[p] = add_off(start_positon[0] + horizontal_platform_interval * int(p.name[2:]),
                                       start_positon[1] + vertical_platform_interval)
            elif p.name.startswith("s2"):
                positions[p] = add_off(start_positon[0] + horizontal_platform_interval * int(p.name[2:]),
                                       start_positon[1] - vertical_platform_interval)
        return positions

    if topology_id != 3:
        return {}

    # Topology 3
    """Calculate platform positions
    - 上行路径放在上方直线；下行路径放在下方直线（顺序按路径段首次出现）
    - 辅助/道岔/连接/折返/入段/出段/备用/存车/渡线/交叉渡线等点放在中间直线，x 贴近其所属主站（上/下行）
    - 对齐：罗湖1/2/3道同列；机场东上/下行同列
    返回: dict {Platform对象: (x, y)}
    """
    sx, sy = float(start_positon[0] + offset[0]), float(start_positon[1] + offset[1])
    dx, dy = float(horizontal_platform_interval), float(vertical_platform_interval)
    name_map = {p.name: p for p in platforms}

    # 路径顺序（按你给的清单）
    up_names = [
        "机场东上行折返线","机场东交叉渡线 2","机场东交叉渡线 1","机场东上行","后瑞上行",
        "固戍上行连接备用线 3","固戍上行","固戍上行连接备用线 1","西乡上行连接折返线 3","西乡上行",
        "西乡上行连接折返线 1","坪洲上行","宝体渡线 1","宝体上行","宝安中心上行","新安上行","新安上行连接出段线 1",
        "前海湾上行连接入段线 1","前海湾上行","鲤鱼门上行","大新上行","桃园上行","深大上行连接存车线 3",
        "深大上行连接存车线 1","深大上行","高新园上行","白石洲上行","世界之窗渡线 1","世界之窗上行",
        "华侨城上行","侨城东上行","竹子林上行连接出段线 3","竹子林上行连接出段线 1","竹子林上行",
        "竹子林上行连接备用线","车公庙上行","香蜜湖上行","购物公园上行","会展中心渡线 1","会展中心上行",
        "会展中心上行连接存车线 3","会展中心上行连接存车线 1","岗厦上行","华强路上行","科学馆上行",
        "大剧院渡线 1","大剧院上行","老街上行","国贸上行","罗湖交叉渡线 2","罗湖交叉渡线 1",
        "罗湖 1 道连接上行线","罗湖 2 道"
    ]
    down_names = [
        "罗湖 3 道","罗湖交叉渡线 3","罗湖交叉渡线 4","国贸下行","老街下行","大剧院下行","大剧院渡线 2",
        "科学馆下行","华强路下行","岗厦下行","会展中心下行连接存车线 1","会展中心下行连接存车线 3",
        "会展中心下行","会展中心渡线 2","购物公园下行","香蜜湖下行","车公庙下行","竹子林下行连接备用线 1",
        "竹子林下行","竹子林下行连接入段线 1","竹子林下行连接入段线 3","侨城东下行","华侨城下行",
        "世界之窗下行","世界之窗渡线 2","白石洲下行","高新园下行","深大下行","深大下行连接存车线 1",
        "桃园下行","大新下行","鲤鱼门下行","前海湾下行","前海湾下行连接入段线 1","新安下行连接出段线 1",
        "新安下行","宝安中心下行","宝体下行","宝体渡线 2","坪洲下行","西乡下行连接折返线 1","西乡下行",
        "固戍下行连接备用线 1","固戍下行","固戍下行连接备用线 3","后瑞下行","机场东下行","机场东交叉渡线 3",
        "机场东交叉渡线 4","机场东下行折返线"
    ]

    def unique_seq(names):
        seen, out = set(), []
        for n in names:
            p = name_map.get(n)
            if p and p not in seen:
                out.append(p); seen.add(p)
        return out

    # 上行反向（左=罗湖，右=机场），下行按列出顺序
    up_order = list(reversed(unique_seq(up_names)))
    down_order = unique_seq(down_names)

    aux_keywords = r"(连接|渡线|出段线|入段线|备用线|存车线|交叉渡线|折返线)"
    def is_main_up(p): return ("上行" in p.name) and (re.search(aux_keywords, p.name) is None)
    def is_main_down(p): return ("下行" in p.name) and (re.search(aux_keywords, p.name) is None)
    def base_name(name):
        n = re.sub(r"\s*\d+\s*道", " 道", name)
        n = re.sub(r"[上下]行", "", n)
        n = re.sub(aux_keywords + r".*", "", n)
        n = re.sub(r"\s*\d+\s*$", "", n)
        return n.strip()

    # 初始列索引（各自按顺序）
    up_idx0 = {p: i for i, p in enumerate(up_order)}
    dn_idx0 = {p: i for i, p in enumerate(down_order)}

    # 纵向对齐目标：同名车站取“更靠右”的列
    base_up_idx = {base_name(p.name): up_idx0[p] for p in up_order if is_main_up(p)}
    base_dn_idx = {base_name(p.name): dn_idx0[p] for p in down_order if is_main_down(p)}
    bases = set(base_up_idx) | set(base_dn_idx)

    # 约束集合（指定最小列索引）
    assign_up = {}
    assign_dn = {}

    # 罗湖 1/2/3 道固定在最左列（0）
    for n in ["罗湖 1 道", "罗湖 2 道"]:
        p = name_map.get(n)
        if p in up_idx0:
            assign_up[p] = 0
    p3 = name_map.get("罗湖 3 道")
    if p3 in dn_idx0:
        assign_dn[p3] = 0

    # 同名车站：取更靠右的列
    for b in bases:
        t = max(base_up_idx.get(b, -1), base_dn_idx.get(b, -1))
        pu = name_map.get(b + "上行") or None
        pd = name_map.get(b + "下行") or None
        if pu in up_idx0:
            assign_up[pu] = max(assign_up.get(pu, -1), t)
        if pd in dn_idx0:
            assign_dn[pd] = max(assign_dn.get(pd, -1), t)

    # 压实函数：保持顺序单调并满足最小列约束
    def compact(order_list, assigned_min):
        out = {}
        cur = -1
        for item in order_list:
            want = assigned_min.get(item, cur + 1)
            if want <= cur:
                want = cur + 1
            out[item] = want
            cur = want
        return out, cur  # cur 是最大列

    # 迭代压实，直到上下行同名列稳定
    for _ in range(5):
        up_idx, up_max = compact(up_order, assign_up)
        dn_idx, dn_max = compact(down_order, assign_dn)

        changed = False
        for b in bases:
            # 找到上下行对应平台
            pu = name_map.get(b + "上行")
            pd = name_map.get(b + "下行")
            if (pu in up_idx) or (pd in dn_idx):
                xu = up_idx.get(pu, -1)
                xd = dn_idx.get(pd, -1)
                target = max(xu, xd)
                if pu in up_idx and assign_up.get(pu, -1) != target:
                    assign_up[pu] = target; changed = True
                if pd in dn_idx and assign_dn.get(pd, -1) != target:
                    assign_dn[pd] = target; changed = True
        if not changed:
            break

    # 最终索引与最右列
    up_idx, up_max = compact(up_order, assign_up)
    dn_idx, dn_max = compact(down_order, assign_dn)
    right_col = max(up_max, dn_max)

    # 机场东上/下行折返线固定在最右列
    ae_up_turn = name_map.get("机场东上行折返线")
    ae_dn_turn = name_map.get("机场东下行折返线")
    if ae_up_turn in up_idx:
        up_idx[ae_up_turn] = right_col
    if ae_dn_turn in dn_idx:
        dn_idx[ae_dn_turn] = right_col

    # 生成坐标
    pos = {}
    x_map = {}

    up_y = sy + dy
    for p in up_order:
        if p in up_idx:
            x = sx + up_idx[p] * dx
            pos[p] = (x, up_y); x_map[p] = x

    down_y = sy - dy
    for p in down_order:
        if p in dn_idx:
            x = sx + dn_idx[p] * dx
            pos[p] = (x, down_y); x_map[p] = x

    # 邻接用于锚定中线辅助点
    from collections import defaultdict
    neighbors = defaultdict(set)
    for seg in line_segments:
        a, b = seg.start_platform, seg.end_platform
        neighbors[a].add(b); neighbors[b].add(a)

    up_set, down_set = set(up_order), set(down_order)
    mid_y = sy

    # 不在两条路径中的平台作为中线辅助点
    aux_nodes = [p for p in platforms if p not in up_set and p not in down_set]

    # 锚定：优先用名称基地找到对应主站的列，否则用邻居已有列的均值
    def base_of(p): return base_name(p.name)
    anchor_use_count = defaultdict(int)

    for p in aux_nodes:
        anchor_x = None
        b = base_of(p)
        # 优先有列的上/下行主站
        pu = name_map.get(b + "上行")
        pd = name_map.get(b + "下行")
        if pu in x_map:
            anchor_x = x_map[pu]
        if pd in x_map:
            anchor_x = x_map[pd] if anchor_x is None else max(anchor_x, x_map[pd])
        # 否则用邻居
        if anchor_x is None:
            xs = [x_map[q] for q in neighbors.get(p, ()) if q in x_map]
            if xs:
                anchor_x = sum(xs) / len(xs)
        # 兜底
        if anchor_x is None:
            anchor_x = sx

        k = anchor_use_count[anchor_x]  # 使用 x 值作为桶，微偏移避免重叠
        offset_x = ((k // 2) + 1) * (dx * 0.25)
        offset_x = offset_x if (k % 2 == 0) else -offset_x
        x = max(sx, min(sx + right_col * dx, anchor_x + offset_x))
        pos[p] = (float(x), float(mid_y))
        anchor_use_count[anchor_x] += 1

    # 特殊摆位：前海出/入段线在“前海湾上行”正下方
    if "前海湾上行" in name_map:
        qh = name_map["前海湾上行"]
        if qh in pos:
            qx, qy = pos[qh]
            by = qy + dy * 0.8
            if "前海出段线" in name_map:
                pos[name_map["前海出段线"]] = (qx + dx * 0.18, by)
            if "前海入段线" in name_map:
                pos[name_map["前海入段线"]] = (qx - dx * 0.18, by)

    # 特殊摆位：竹子林出/入段线在“竹子林（上/下）”正下方（以上行优先）
    anchor_zzl = None
    if "竹子林上行" in name_map and name_map["竹子林上行"] in pos:
        anchor_zzl = name_map["竹子林上行"]
    elif "竹子林下行" in name_map and name_map["竹子林下行"] in pos:
        anchor_zzl = name_map["竹子林下行"]
    if anchor_zzl:
        ax, ay = pos[anchor_zzl]
        by2 = ay + dy * 0.8
        if "竹子林出段线" in name_map:
            pos[name_map["竹子林出段线"]] = (ax + dx * 0.18, by2)
        if "竹子林入段线" in name_map:
            pos[name_map["竹子林入段线"]] = (ax - dx * 0.18, by2)


    # 强制将“罗湖 1 道”固定在最左侧
    p_lh1 = name_map.get("罗湖 1 道")
    if p_lh1 is not None:
        # 保持其当前的 y（辅助点通常在中线），仅将 x 固定为最左列
        y = pos.get(p_lh1, (sx, sy))[1]
        pos[p_lh1] = (sx, y)


    # 会展中心上/下行连接存车线 2/4从左至右的顺序为：会展中心上行连接存车线2 - 会展中心下行连接存车线2 - 会展中心下行连接存车线4 - 会展中心上行连接存车线4
    names_hz = [
        "会展中心上行连接存车线 2",
        "会展中心下行连接存车线 2",
        "会展中心下行连接存车线 4",
        "会展中心上行连接存车线 4"
    ]
    hz_points = [name_map.get(n) for n in names_hz if name_map.get(n) in pos]
    if len(hz_points) == 4:
        x_start = min(pos[p][0] for p in hz_points)
        x_end = max(pos[p][0] for p in hz_points)
        y_avg = sum(pos[p][1] for p in hz_points) / 4.0
        for i, p in enumerate(hz_points):
            x = x_start + i * (x_end - x_start) / 3.0
            # 向左移动一个区间距离
            x = x - dx
            pos[p] = (x, y_avg)

    # 竹子林下行入段线2/4应分别与竹子林下行入段线1/3对齐，位于其正下方约1/4的上下行线路间距处；
    d35p = name_map.get("竹子林下行连接入段线 1")
    d36p = name_map.get("竹子林下行连接入段线 2")
    d37p = name_map.get("竹子林下行连接入段线 3")
    d38p = name_map.get("竹子林下行连接入段线 4")
    y_down_quarter = sy - 0.5 * dy  # 1/4 的上下行线路间距（上下行相距 2*dy）

    if d36p and d35p and d35p in pos:
        pos[d36p] = (pos[d35p][0], y_down_quarter)
    if d38p and d37p and d37p in pos:
        pos[d38p] = (pos[d37p][0], y_down_quarter)

    # 竹子林上行出段线2/4应分别与竹子林上行出段线1/3对齐，位于其正上方约1/4的上下行线路间距处；
    d31p = name_map.get("竹子林上行连接出段线 1")
    d32p = name_map.get("竹子林上行连接出段线 2")
    d33p = name_map.get("竹子林上行连接出段线 3")
    d34p = name_map.get("竹子林上行连接出段线 4")
    y_up_quarter = sy + 0.5 * dy

    if d32p and d31p and d31p in pos:
        pos[d32p] = (pos[d31p][0], y_up_quarter)
    if d34p and d33p and d33p in pos:
        pos[d34p] = (pos[d33p][0], y_up_quarter)

    # 竹子林出段线连接备用线应位于竹子林下行入段线2/4左侧一个区间距离处，
    # 纵向位于1/2的上下行线路间距处，从右至左顺序为：
    # 竹子林出段线连接备用线 - 竹子林备用线 - 竹子林下行连接备用线2
    d39p = name_map.get("竹子林出段线连接备用线")
    s35p = name_map.get("竹子林备用线")
    d30p = name_map.get("竹子林下行连接备用线 2")

    # 参考列：取“下行入段线2/4”中更靠右的列
    ref_x_candidates = []
    for q in (d36p, d38p, d35p, d37p):
        if q and q in pos:
            ref_x_candidates.append(pos[q][0])
    x_ref = max(ref_x_candidates) if ref_x_candidates else sx
    y_mid = sy

    # 从右到左依次放置：d39, s35, d30，并以 x_ref 左侧一个区间为起点
    def clamp_x(x):
        return max(sx, min(sx + right_col * dx, x))

    cur_x = x_ref - dx
    if d39p:
        pos[d39p] = (clamp_x(cur_x), y_mid)
        cur_x -= dx
    if s35p:
        pos[s35p] = (clamp_x(cur_x), y_mid)
        cur_x -= dx
    if d30p:
        pos[d30p] = (clamp_x(cur_x), y_mid)


    # 从左至右：西乡上行连接折返线2、西乡下行连接折返线2、西乡下/上行连接折返线4
    xixiang_order_names = [
        "西乡上行连接折返线 2",
        "西乡下行连接折返线 2",
    ]
    # 第三个优先使用“下行4”，若不存在则回退到“上行4”
    third_name = "西乡下行连接折返线 4"
    if third_name not in name_map:
        third_name = "西乡上行连接折返线 4"
    xixiang_order_names.append(third_name)

    pts = [name_map.get(n) for n in xixiang_order_names]
    pts = [p for p in pts if p is not None and p in pos]
    if len(pts) == 3:
        # 参考水平范围：优先用“西乡上/下行”的列；否则用当前三点的最小/最大列
        ref_xs = []
        for anchor_name in ("西乡上行", "西乡下行"):
            ap = name_map.get(anchor_name)
            if ap in pos:
                ref_xs.append(pos[ap][0])
        if len(ref_xs) >= 2:
            x_start, x_end = min(ref_xs), max(ref_xs)
        else:
            cur_xs = [pos[p][0] for p in pts]
            x_start, x_end = min(cur_xs), max(cur_xs)
        # 略加边距
        margin = 0.8 * dx
        x_start, x_end = x_start - margin, x_end + margin
        y_mid = sy
        # 均匀布置三点
        for i, p in enumerate(pts):
            x = x_start + i * (x_end - x_start) / 2.0
            pos[p] = (clamp_x(x), y_mid)


    return pos

# 初始加载（环境变量优先）
load_topology(int(os.environ.get("METRO_TOPOLOGY", 1)))


