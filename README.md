# Mock Interview — AI 面试模拟 Agent

基于 LangGraph 构建的面试模拟 Agent：运行时传入简历和岗位 JD，Agent 充当面试官开展完整多轮模拟面试，结合本地知识库（RAG）动态提问与追问，面试结束后输出多维度评估报告和改进建议。

## 特性

- **LangGraph 状态机编排**：8 个节点串联完整面试流程，条件路由控制追问 / 切题 / 结束
- **Human-in-the-Loop**：`interrupt()` 暂停等待候选人回答，`SqliteSaver` 持久化会话，支持断点恢复
- **出题与评分分离**：独立 Prompt + 不同 temperature，规避自问自答偏差
- **RAG 知识库增强**：Markdown 标题语义切块 + Chroma 向量检索，面试问题结合专业知识上下文
- **三级难度选择**：用户手动选择简单 / 中等 / 困难，全局贯穿规划、出题与追问，Prompt 强约束不越档
- **智能追问控制**：评分 >= 7 分直接切题，仅在回答不合格时追问，避免吃满追问次数导致主题覆盖不足
- **单问题约束**：Prompt 强制每次只问一个问题，禁止编号列表和多子问题拆分
- **Web 界面**：FastAPI 后端 + 原生前端，支持文件上传、实时对话、进度展示和报告查看
- **历史面试记录**：面试结束自动保存到浏览器 localStorage，支持回看完整对话和报告、单条删除
- **代码侧边界兜底**：题数上限、追问次数、评分维度对齐、缺字段清洗，关键数据不依赖 LLM 正确输出
- **结构化输出**：全链路 Pydantic 模型约束，简历 / JD / 提纲 / 评分 / 报告均为强类型对象

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言 / 构建 | Python 3.10+，uv，hatchling |
| Agent 框架 | LangGraph，LangChain |
| Web 后端 | FastAPI（SSE 事件流 + 静态文件托管） |
| Web 前端 | 原生 HTML / CSS / JavaScript（无框架） |
| 向量库 | Chroma（本地持久化） |
| 检查点 | langgraph-checkpoint-sqlite（SqliteSaver） |
| LLM | 阿里百炼 API（OpenAI-compatible 协议） |
| 对话模型 | `qwen-plus`（出题 / 评分可分别配置） |
| 嵌入模型 | `text-embedding-v4` |
| 数据模型 | Pydantic v2 + pydantic-settings |
| 文档解析 | python-docx，pypdf |
| 重试 | tenacity（指数退避，3 次） |
| 测试 / 代码规范 | pytest，ruff |

## 面试流程

```text
传入简历 + JD + 难度（简单/中等/困难）
    → parse_inputs      LLM 解析为结构化简历 / JD 对象
    → plan_interview    生成面试提纲（主题 + 考察重点，难度统一为用户选择值）
    → 循环：
        retrieve_knowledge  按当前主题检索知识库 top-k
        ask_question        首问或追问（单问题，严格按全局难度出题）
        wait_answer         interrupt() 暂停，等候选人输入
        judge_answer        独立评分（4 维度，低 temperature）
        decide_next         代码边界决策：高分切题 / 追问 / 切题 / 结束
    → generate_report   聚合各轮评分，生成最终评估报告
```

### 评分维度

固定 4 个维度（`evaluation/rubric.py` 为单一权威来源）：

1. **专业准确性** — 技术概念、原理、实现的正确程度
2. **表达结构** — 回答的条理性、逻辑性、清晰度
3. **岗位匹配度** — 与 JD 要求的契合程度
4. **应变能力** — 面对追问和陌生问题的反应与思考

### 边界控制

| 约束 | 默认值 | 说明 |
| --- | --- | --- |
| `max_questions` | 15 | 总题数上限（含追问），达到则强制结束 |
| `max_follow_ups` | 2 | 单主题最大追问次数，达到则强制切题 |
| `pass_score_threshold` | 7.0 | 评分达到该值直接切题，不再追问 |
| `retrieval_top_k` | 4 | 知识库检索返回条数 |

## 项目结构

```text
Mock Interview/
├── pyproject.toml                  # 依赖与工具配置
├── .env.example                    # 环境变量示例
├── my_resume.md                    # 示例简历
├── my_jd.md                        # 示例 JD
├── PLANNING.md                     # 项目规划与设计决策
├── 优化方案-追问控制与难度分级.md    # 追问控制与难度分级的改动方案
├── frontend/                       # Web 前端（原生 HTML/CSS/JS）
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── data/
│   ├── kb/                         # Markdown 知识库源文件
│   ├── vectorstore/                # Chroma 本地向量库（入库后生成）
│   └── sessions/                   # 面试会话记录（按 session_id 分目录）
├── scripts/
│   ├── ingest_kb.py                # 知识库入库脚本
│   └── run_interview.py            # 面试 CLI 入口
├── src/interview_agent/
│   ├── config.py                   # pydantic-settings 配置
│   ├── models.py                   # Pydantic 领域模型
│   ├── state.py                    # LangGraph 状态定义 + reducer
│   ├── graph.py                    # 图组装与编译
│   ├── web/                        # FastAPI Web 服务
│   │   ├── app.py                  # 路由 + SSE 事件流 + 静态托管
│   │   ├── sessions.py             # 会话管理 + LangGraph 运行器
│   │   ├── schemas.py              # 请求/响应模型
│   │   └── parsers.py              # 文件上传解析（md/txt/docx/pdf）
│   ├── nodes/                      # 8 个图节点
│   │   ├── parse_inputs.py
│   │   ├── plan_interview.py
│   │   ├── retrieve_knowledge.py
│   │   ├── ask_question.py
│   │   ├── wait_answer.py
│   │   ├── judge_answer.py
│   │   ├── decide_next.py
│   │   └── generate_report.py
│   ├── llm/
│   │   ├── client.py               # 百炼客户端（语义化方法 + 重试 + 异常翻译）
│   │   └── prompts.py              # 分场景 Prompt 常量
│   ├── knowledge/
│   │   ├── chunker.py              # Markdown 标题语义切块
│   │   ├── embedder.py             # text-embedding-v4 封装
│   │   ├── retriever.py            # Chroma 检索
│   │   └── ingest.py               # 入库主流程
│   ├── storage/
│   │   ├── checkpoint.py           # SqliteSaver / InMemorySaver 统一入口
│   │   └── session_store.py        # 会话 JSON 存储（流水 + 摘要）
│   └── evaluation/
│       ├── rubric.py               # 评分维度常量 + 对齐函数
│       └── report_generator.py     # 报告生成薄封装
└── tests/                          # pytest 测试（12 个测试文件）
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入阿里百炼 API Key：

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> API Key 也可通过系统环境变量 `DASHSCOPE_API_KEY` 或 `API_KEY` 提供，三者兼容。

### 3. 构建知识库

将 Markdown 格式的知识库文件放入 `data/kb/` 目录，然后运行入库脚本：

```bash
# 全量重建（删除旧 collection 后重新入库）
python -m scripts.ingest_kb --reset

# 增量入库单个文件
python -m scripts.ingest_kb --file data/kb/新文档.md

# 指定知识库目录
python -m scripts.ingest_kb --kb-path /path/to/kb --reset
```

入库脚本参数：

| 参数 | 说明 |
| --- | --- |
| `--kb-path` | Markdown 知识库目录（默认取配置 `KB_PATH`） |
| `--file` | 只入库单个 `.md` 文件（与 `--kb-path` 互斥） |
| `--reset` | 重建前删除旧 collection（不能与 `--file` 同时使用） |
| `--verbose` | 输出调试日志 |

### 4. 运行面试

#### 方式一：命令行（CLI）

```bash
python -m scripts.run_interview --resume my_resume.md --jd my_jd.md --difficulty 中等
```

面试 CLI 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--resume` | （必填） | 简历原文 `.md` 文件路径 |
| `--jd` | （必填） | JD 原文 `.md` 文件路径 |
| `--difficulty` | `中等` | 面试难度：`简单` / `中等` / `困难` |
| `--session-id` | `interview-YYYYMMDD-HHMMSS` | 会话唯一标识 |
| `--checkpoint-mode` | `memory` | `memory`（进程内，重启即丢）或 `sqlite`（持久化） |

交互说明：

- 面试官提问后，输入回答并按**空行**提交本题
- 输入 `quit` 可主动结束整个面试
- 评分阶段 LLM 评判通常耗时 5–30 秒，CLI 会显示等待提示

#### 方式二：Web 界面

```bash
uvicorn interview_agent.web.app:app --reload --port 8000
```

浏览器打开 `http://127.0.0.1:8000`，功能包括：

- 上传简历与 JD 文件（支持 .md / .txt / .docx / .pdf）
- 选择面试难度（简单 / 中等 / 困难）
- 实时对话界面，展示面试进度、考察主题、候选人信息
- SSE 实时推送分析进度、面试官提问、评分结果
- 面试结束后展示多维度评估报告，支持下载 Markdown 报告
- 历史面试记录自动保存到浏览器本地，可回看完整对话与报告

### 5. 运行测试

```bash
pytest
```

测试默认注入假 API Key（`conftest.py`），不触达网络，可直接运行。

## 配置说明

所有配置通过环境变量或 `.env` 文件提供，`config.py` 使用 pydantic-settings 读取。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | （必填） | 阿里百炼 API Key |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容模式 base_url |
| `KB_PATH` | `data/kb` | 知识库源文件目录 |
| `KB_VECTOR_PATH` | `data/vectorstore` | Chroma 向量库存储目录 |
| `DB_PATH` | `data/sessions/interview.sqlite` | LangGraph checkpoint SQLite 路径 |
| `INTERVIEWER_MODEL` | `qwen-plus` | 出题 / 解析 / 规划用对话模型 |
| `JUDGE_MODEL` | `qwen-plus` | 评分 / 报告用对话模型 |
| `EMBEDDING_MODEL` | `text-embedding-v4` | 向量化模型 |
| `RETRIEVAL_TOP_K` | `4` | 知识库检索返回条数 |
| `MAX_FOLLOW_UPS` | `2` | 单主题最大追问次数 |
| `MAX_QUESTIONS` | `15` | 整个面试最大题数 |

> 相对路径会自动解析为相对于项目根目录的绝对路径。

### LangSmith 跟踪（可选）

`.env.example` 中包含 LangSmith 配置，LangChain SDK 会自动读取以下环境变量开启链路追踪：

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-api-key
LANGSMITH_PROJECT=mock
```

不需要时可直接删除或注释这些行。

## 设计要点

### 出题与评分分离

- `ask_question` 使用 `temperature=0.8`，允许问题灵活多样
- `judge_answer` 使用 `temperature=0.2`，保持评分严格低随机性
- 两者使用独立的 System Prompt，避免同一模型既当选手又当裁判

### 三级难度策略

- 用户在启动时选择全局难度（简单 / 中等 / 困难），贯穿规划、出题与追问全链路
- `plan_interview` 阶段 Prompt 强制所有主题 difficulty 统一等于用户选择值，代码侧再强制回填兜底
- `ask_question` / `follow_up` 阶段 Prompt 给出每档难度的具体出题标准，要求严格按档出题、不得越档
- 回答错误时可在当前档位内降一层验证基础理解，但不得向上突破全局难度上限

### 追问控制与单问题约束

- **高分直接切题**：`decide_next` 节点中，评分 `overall_score >= 7.0` 时直接切题，不再追问，避免吃满追问次数导致主题覆盖不足
- **优先级顺序**：题数上限 → 追问次数上限 → 高分切题 → LLM suggestion，代码侧硬判断优先于模型建议
- **单问题约束**：出题与追问 Prompt 强制每次只问一个核心问题，禁止编号列表和多子问题拆分，确保候选人可在一段文字内回答

### 代码侧兜底策略

项目遵循"代码做边界否决，Agent 做内容决策"原则：

- **评分维度对齐**：`normalize_score_items()` 确保 LLM 漏输出维度时按 overall_score 均值补齐，去重并固定顺序
- **关键数据覆盖**：`session_id`、`question`、`answer`、`rounds` 等字段由代码强制回填，不依赖 LLM 正确输出
- **追问类型约束**：`follow_up()` 代码侧强制 `question_type="follow_up"`
- **空值清洗**：`evidence` / `suggestion` 字段为 `None` 时统一替换为空串

### LLM 调用可靠性

- **双保险 JSON 解析**：`response_format={"type":"json_object"}` + System Prompt 末尾强制只输出 JSON + 代码侧 strip 代码围栏
- **异常分类**：`DashScopeClientError`（不可重试）与 `DashScopeRetryableError`（可重试）分离
- **自动重试**：tenacity 指数退避，针对 429 / 5xx / 超时 / 网络抖动重试 3 次
- **脱敏错误**：异常信息只摘 code + message 前 200 字符，不回显完整 prompt

### 知识库切块策略

`chunker.py` 实现结构化语义切块：

1. 按 Markdown 标题建立文档结构树
2. `##` 级标题默认独立成块，`###` / `####` 超过 token 阈值时继续下切
3. 代码块 / 表格 / Mermaid 图永不切分
4. 叶子节点内容过长时按句子粒度聚合
5. 最后合并小于最小 token 阈值的块
6. 注入 overlap 并打包元数据（source / doc_title / heading_path / chunk_index 等）

## 更新日志

### 2026-08-25

- **简历/JD 解析缓存**：对纯文本内容算 SHA256，内容不变时复用上次解析结果，跳过重复 LLM 调用；缓存存 `data/parse_cache/`，自动按内容失效
- **Web 步骤显示同步**：修复前端状态滞后一个节点的问题，解析/规划/检索阶段与实际执行对齐
- **Prompt 优化**：出题强制结合候选人项目经历，禁止要求现场编写完整代码；追问节点增加活人感约束；精简冗余条款

## 许可证

MIT
