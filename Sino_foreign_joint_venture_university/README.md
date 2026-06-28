# 中外合作办学智能指南

一站式采集、分析、咨询中国中外合作办学项目的工具。帮助考生和家长快速了解全国中外合办项目的全貌，并根据个人条件获得个性化推荐。

## 项目背景

近年来中国高校中外合作办学项目激增，但信息高度分散：教育部平台只有审批信息，学费、师资、住宿、海外交换等考生真正关心的内容需要逐个去官网查找。本项目解决这一痛点：

1. **自动采集** 教育部中外合作办学监管信息平台的全部项目数据
2. **深度补充** 通过搜索引擎 + AI 提取各项目的学费、录取分数、师资、住宿、海外交换等详细信息
3. **院校标注** 为中方院校标注 985/211/双一流层级，为外方院校标注 QS 排名和学科特长
4. **智能咨询** 支持条件筛选和 AI 问答（Claude / 智谱 GLM），根据考生画像推荐最适合的项目
5. **可视化展示** Streamlit Web App 提供总览仪表盘、项目筛选、多项目对比等功能

## 数据覆盖

当前已采集：
- 849 个项目（北京、上海、江苏、浙江、广东、福建）
- 326 所中方院校画像
- 632 所外方院校画像（含 QS 排名）

## 快速开始

### 1. 环境准备

```bash
cd Sino_foreign_joint_venture_university
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 Anthropic API Key
```

### 3. CLI 使用

```bash
# 采集数据（从教育部平台）
python main.py scrape --region 2          # 采集上海
python main.py scrape                     # 采集所有地区

# 补充深度信息（学费/师资/住宿/分数等）
python main.py enrich --region 1 --priority   # 北京，优先重点院校
python main.py enrich --limit 10              # 测试前10个

# 构建院校数据库（985/211/QS排名标注）
python main.py institutions                   # 首次构建
python main.py institutions --rebuild         # 强制重建

# AI 分析项目特色
python main.py analyze

# 交互式咨询
python main.py consult

# 查看数据统计
python main.py stats
```

### 4. Streamlit Web App

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

浏览器访问 http://localhost:8501

如需 AI 咨询功能，在 `streamlit_app/.streamlit/secrets.toml` 中配置：
```toml
ANTHROPIC_API_KEY = "sk-ant-xxx"
ZHIPU_API_KEY = "xxx"  # 可选
```

### 5. 用户系统

Web App 内置轻量级用户系统（SQLite）：

- **注册/登录**：任何人可用有效邮箱注册，注册即生效
- **普通功能**：登录后可访问总览、项目筛选、项目对比
- **AI 咨询权限**：需要以下任一条件：
  - 管理员审批通过
  - 用户自行提供智谱 GLM API Key（访问 [bigmodel.cn](https://bigmodel.cn) 注册获取）
- **管理员面板**：审批用户请求、管理用户、选择AI模型后端
- **使用统计**：管理员可查看用户活动日志和AI问答详情

> 用户提供的 API Key 仅存储在浏览器会话中，不会持久化到服务器。

## 项目结构

```
Sino_foreign_joint_venture_university/
├── main.py                 # CLI 入口
├── config.py               # 配置（API key、路径、常量）
├── scraper.py              # 教育部平台数据采集
├── enricher.py             # 深度信息补充（搜索+AI提取）
├── institutions.py         # 院校成色标注（985/211/QS）
├── analyzer.py             # Claude 项目特色分析
├── agent.py                # 交互式咨询 Agent
├── prompts/                # Prompt 模板
│   ├── analysis.py
│   ├── consultation.py
│   └── enrichment.py
├── data/                   # 数据文件
│   ├── programs.json               # 原始采集数据
│   ├── programs_enriched.json      # 深度补充后的数据
│   ├── chinese_universities.json   # 中方院校画像库
│   └── foreign_universities.json   # 外方院校画像库
├── streamlit_app/          # Web 应用
│   ├── app.py              # 首页
│   ├── pages/              # 子页面（总览/筛选/对比/AI咨询）
│   └── utils/              # 工具模块
├── requirements.txt
└── .env.example
```

## 数据字段说明

每个项目包含以下维度的信息：

| 维度 | 字段 |
|------|------|
| 基本信息 | 项目名称、中外方院校、专业、学制、层次、地区 |
| 费用 | 年学费、总学费 |
| 录取 | 录取分数线、招生方式 |
| 院校成色 | 中方层级(985/211/双一流/一本/二本/高职)、外方QS排名、学科特长 |
| 师资 | 教师总数、构成、外方授课形式 |
| 海外交流 | 是否有交换、时长、目的地 |
| 升学通道 | 硕士直升说明 |
| 生活 | 住宿条件、授课地点 |
| 就业 | 实习安排、就业指导、就业率 |

## 技术栈

- Python 3.10+
- requests + BeautifulSoup（数据采集）
- Anthropic Claude SDK（AI 分析与咨询）
- Streamlit + Plotly（Web 可视化）
- 智谱 GLM（可选备用 LLM）

## 注意事项

- 采集数据需要网络连接，教育部平台偶尔限流，建议分批采集
- `enrich` 命令每个项目约耗时 40-50 秒（含搜索间隔），支持中断续传
- 部分项目的详细信息（学费、师资等）取决于公开数据的丰富程度
- 数据仅供参考，以各项目官方最新招生简章为准
