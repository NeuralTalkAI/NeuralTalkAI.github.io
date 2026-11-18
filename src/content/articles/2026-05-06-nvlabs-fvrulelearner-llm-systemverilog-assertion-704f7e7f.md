关键词：**形式验证**、:mdnice-bold-italic[SystemVerilog 断言]、Q-Tree、**RAG**、JasperGold

> 让大模型根据自然语言写出 SystemVerilog Assertion，看似只是“代码生成”的一个分支，**真正难点却不在语法，而在语义**：同一个“握手后必须响应”的需求，可能涉及时序蕴含、复位屏蔽、信号稳定性、延迟窗口、位级抽象等多个微妙选择。

论文数据显示，NL-to-SVA 任务中超过 80%的功能错误源于时序算子误用，其中仅`|->`与`|=>`的混淆就导致了 50%-64%的失效，这类错误文本相似度极高，但会让形式验证结果完全相悖。

![文章插图](/articles/2026-05-06-nvlabs-fvrulelearner-llm-systemverilog-assertion-704f7e7f/images/01.webp)

::: image-caption
给定一个自然语言到 SVA 的任务，大语言模型推理通常会生成功能不正确的 SVA，而我们的 FVRULELEARNER 框架利用学习到的 OP-TREE 生成功能正确的 SVA。 该图直观呈现传统 LLM 与 FVRULELEARNER 的 SVA 生成差异，直指 LLM 在 NL-to-SVA 任务中因时序算子理解不足、训练数据有限导致的功能错误问题。传统 LLM 无法精准匹配硬件时序语义，而 FVRULELEARNER 依托算子推理树的结构化推理，锚定 SVA 算子的正确使用逻辑，从根源解决功能失配问题，验证了 OP-TREE 对提升 SVA 功能正确性的核心价值，也奠定了框架的技术创新基础。
:::

::: block-1
LLM 一旦把 `|->` 写成 `|=>`，或漏掉某个 enable 条件，文本相似度可能仍然很高，但形式验证结果会完全不同。
:::

![文章插图](/articles/2026-05-06-nvlabs-fvrulelearner-llm-systemverilog-assertion-704f7e7f/images/02.webp)

- **FVRULELEARNER: Operator-Level Reasoning Tree (OP-TREE)-Based Rules Learning for Formal Verification**
- FVRuleLearner 是面向NL2SVA的规则学习流水线。当前发布主线为`src`目录下的纯提示词链路，采用JasperGold进行语法与功能评估。
- 代码：https://github.com/NVlabs/FVRuleLearner
- 论文：https://arxiv.org/abs/2604.03245
- 9000 字，阅读 40 分钟，播客 18 分钟

相关推荐

- [像写软件一样造芯片：Google XLS 让软件工程师写Verilog，用编译器思维颠覆硬件开发](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899137&idx=1&sn=c98294c41e5453f33c67ee1b3c826838&scene=21#wechat_redirect)
- [斯坦福、CMU 提出 VeriCoder：基于 LLM 的 RTL 代码生成，VerilogEval 上功能正确性 SOTA](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888243&idx=1&sn=76d2a20cfb1cfecc0a2c86d72ddfeaf7&scene=21#wechat_redirect)
- [Vibe-GPU：从需求到落地！Cursor + Gemini 3 Pro 打造四核心GPU，支持FP8类型 ，验证硬件 RTL 开发能力](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898966&idx=1&sn=86a445ce447c8a2aac00d51cff31c4ee&scene=21#wechat_redirect)

> NVlabs/FVRuleLearner 的核心价值，正是把这些失败样本从一次性错误，转化为可被后续检索、排序、解释和复用的规则资产。

![文章插图](/articles/2026-05-06-nvlabs-fvrulelearner-llm-systemverilog-assertion-704f7e7f/images/03.webp)

::: image-caption
FVRULELEARNER 流程概述及训练中 OP-TREE 的示例，该图拆解了 FVRULELEARNER 训练、测试双阶段核心流程，训练阶段通过 OP-TREE 三层结构模拟验证工程师调试逻辑，提炼有效算子推理轨迹；测试阶段通过算子 - 语义混合检索匹配轨迹，经规则适配生成新任务的 Op-Rule。OP-TREE 让推理过程可解释，解决了 LLM 生成 SVA 的黑盒问题，同时双阶段设计保障了规则的有效性与跨任务泛化能力。
:::

它不是简单地堆提示词，而是:mdnice-bold-italic[在训练阶段构建三层 OP-Tree 追问错误原因，通过上下文诊断、理论锚定、规则生成生成可验证的算子级规则，仅保留经 JasperGold 验证功能等价的推理路径；在推理阶段采用算子-语义混合评分的 RAG 机制，结合信号抽象的规则适配生成精准断言。]

![文章插图](/articles/2026-05-06-nvlabs-fvrulelearner-llm-systemverilog-assertion-704f7e7f/images/04.webp)

::: image-caption
通用规则与算子级规则的对比。该图对比两类规则的 SVA 生成效果，通用规则因抽象性无法指导 LLM 精准使用时序算子，最终生成功能错误的断言；Op-Rule 是针对 SVA 算子的细粒度可执行指令，明确界定时序算子的使用场景，彻底消除语义歧义。这一对比验证了算子级规则的精准性是解决 LLM 时序语义缺陷的关键，也是 FVRULELEARNER 区别于传统规则学习的核心创新。
:::

::: block-1
实验表明，该框架相比 SOTA 基线语法正确性提升 3.95%，:mdnice-bold-italic[功能正确性平均提升 31.17%，各类算子功能错误平均减少 70.33%，] 且仅需 48%训练数据即可达到性能饱和，同时**开源了 1000 条真实硬件设计的 NL2SVA-OpenCore 基准数据集。**
:::

![文章插图](/articles/2026-05-06-nvlabs-fvrulelearner-llm-systemverilog-assertion-704f7e7f/images/05.webp)

::: image-caption
LLM 模型在三类 NL-to-SVA 测试集上的实验结果。 该表对比 GPT-4o、Claude 4.5 Sonnet、o3-mini 在不同基线与 FVRULELEARNER 下的 BLEU、语法正确性、功能正确性等指标，FVRULELEARNER 在所有模型、数据集上均大幅领先，功能正确性平均提升 31.17%，语法正确性接近 100%。结果证明框架不依赖特定 LLM，具备极强的普适性，能全面弥补各类 LLM 在 SVA 生成上的语义缺陷。
:::

## 本文目录

- 一、快速上手使用
- 二、项目要解决的根问题：SVA 生成不是翻译，而是可验证语义对齐
  - 2.1 从自然语言到形式断言的语义鸿沟
  - 2.2 三类任务与数据入口
- 三、总体架构：训练沉淀规则，推理检索规则，评估闭环验证
  - 3.1 主入口是一台状态机
  - 3.2 配置不是附属品，而是实验协议
- 四、Q-Tree：把一次错误拆成可学习的推理路径
  - 4.1 为什么需要树，而不是一条反思文本
  - 4.2 Q-Tree 构建流程
- 五、RAG 与增强检索：不是找相似答案，而是找相似错误修复经验
  - 5.1 RAGAgent 如何组织文档
  - 5.2 Q-Tree 增强检索的入口
- 六、LLM 编排：轻量路径与 AutoGen 路径并存
  - 6.1 FVProcessor 是流水线调度器
  - 6.2 规则如何进入提示词
- 七、JasperGold 闭环：文本像不等于功能对
  - 7.1 Tcl 脚本定义形式评估语义
  - 7.2 评估输出如何汇总
- 八、SystemVerilog 资源：testbench 不是背景板，而是语义锚点
  - 8.1 annotated_tb 承载验证上下文
  - 8.2 opencore 路径的特殊性
- 九、知识如何落盘：suggestions 与 qtrees 是项目的长期记忆
  - 9.1 Q-Tree 保存为机器与人都能读的格式
  - 9.2 与普通 RAG 的根本区别
- 十、限制、边界与值得借鉴之处
  - 10.1 需要谨慎表述的限制
  - 10.2 对 AI+EDA 系统的启发
  - 10.3 如果把它比作编译器
- 结语：真正的突破在于让形式验证经验可积累

![文章插图](/articles/2026-05-06-nvlabs-fvrulelearner-llm-systemverilog-assertion-704f7e7f/images/06.webp)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、快速上手使用

> FVRuleLearner 当前 release 的主线是 README 中所说的 prompt-only 路径，入口位于 `src/`，主要任务包括 `nl2sva_human`、`nl2sva_machine` 和 `nl2sva_opencore`。最小安装流程如下：

```
# 来源：README.mdconda create -n fvrulelearner python=3.10conda activate fvrulelearnerpip install -r requirements.txt
```

运行前需要准备 LLM 与 JasperGold 环境：

```
# 来源：README.mdexport OPENAI_API_KEY=...# 或：export ANTHROPIC_API_KEY=...export PATH=/path/to/jasper/bin:$PATHexport LM_LICENSE_FILE=<your_jasper_license_server>export CDS_LIC_FILE=<your_jasper_license_server>
```

仓库根目录执行：

```
# 来源：README.mdcd FVRuleLearnerpython3 src/main.py
```

核心配置集中在 `src/config.py`。训练时设置：

```
# 来源：src/config.py# Select one execution stage: train / inference / eval.# global_task = 'inference'global_task = 'train'# global_task = 'eval'# Supported release tasks:# task = "nl2sva_human"# task = "nl2sva_machine"task = "nl2sva_opencore"LLM_gateaway = "openai"# LLM_gateaway = "claude"llm_model = 'gpt-4o'# llm_model = "claude-sonnet-4-5-20250929"
```

训练会输出到：

```
# 来源：README.mdsrc/logs/train_<timestamp>_<hostname>_<user>/  suggestions.pkl  qtrees.pkl  qtrees.json  log.txt  FLAGS.klepto
```

推理时将 `global_task` 改为 `"inference"`，并把 `load_suggestions_path` 指向之前的训练日志目录。也可以通过 `FVRULELEARNER_TRAIN_LOGDIR` 覆盖默认路径。更多数据集映射、推荐参数和 eval-only 用法，可参考仓库 README：https://github.com/NVlabs/FVRuleLearner/blob/main/README.md。

## 二、项目要解决的根问题：SVA 生成不是翻译，而是可验证语义对齐

### 2.1 从自然语言到形式断言的语义鸿沟

> NL2SVA 任务表面上是把一句自然语言描述翻译为 SVA，例如“当写有效且 FIFO 未满时，写指针应更新”。但在形式验证语境中，这句话往往隐含一组工程约束：

- 断言应在哪个时钟边沿采样？
- reset 期间是否禁用？
- antecedent 与 consequent 应用 `|->` 还是 `|=>`？
- 是否需要 `$stable`、`$past`、`##[n:m]`？
- 信号是否来自 testbench 抽象，而非原始 RTL？
- 参考断言等价性如何判断，而不仅是字符串相似？

FVRuleLearner 的论文题目是 Operator-Level Reasoning Tree (OP-Tree)-Based Rules Learning for Formal Verification[1]。仓库 README 明确指出它是 NL2SVA 的 rule-learning pipeline，使用 JasperGold 做 syntax/functionality evaluation。**换句话说，它不满足于“生成一条看起来像 SVA 的代码”，而是希望学习“为什么这条断言错了，以及下次遇到相似 operator-level 模式该怎么修”。**

### 2.2 三类任务与数据入口

> 配置文件中把 release 任务限制在三类：

```
# 来源：src/config.pyif task == "nl2sva_human":    dataset_path = os.path.join(ROOT, "FVEval", "data_nl2sva", "data", "nl2sva_human.csv")elif task == "nl2sva_machine":    dataset_path = os.path.join(ROOT, "FVEval", "data_nl2sva", "data", "nl2sva_machine_updated.csv")elif task == "nl2sva_opencore":    dataset_path = os.path.join(ROOT, "FVEval", "data_1k", "module_sva_nl_manual_editing.csv")else:    raise ValueError(f"Unsupported release task: {task}")
```

项目并不是一个任意 Verilog 仓库的通用断言生成器，而是**围绕特定 benchmark 格式组织的研究原型**。:mdnice-bold-italic[每条数据通常包含自然语言 prompt、参考断言、设计名、task id、testbench 或相关上下文。]

- Python 主体负责组织样本、调用 LLM、生成 SVA、保存结果；
- FVEval 负责把结果与参考答案放入 JasperGold 可执行的评估框架中。

## 三、总体架构：训练沉淀规则，推理检索规则，评估闭环验证

### 3.1 主入口是一台状态机

> `src/main.py` 的主逻辑非常短，却揭示了项目的三阶段设计：`train`、`inference`、`eval`。

```
# 来源：src/main.pydef main():    if FLAGS.global_task in ['inference', 'train']:        if FLAGS.model == 'agent':            start_time = time.time()            from fv_harwareagent_example import FVProcessor            processor = FVProcessor()            processor.main_fv()            saver.save_stats('overall_runtime', time.time()-start_time)        elif FLAGS.model == 'timing_agent':            from hardware_agent.examples.timing_summary_hardwareagent_example import main            main()        else:            raise NotImplementedError()        if FLAGS.global_task == 'inference':            from eval import eval            eval(Path(saver.logdir))    elif FLAGS.global_task == "eval":        from eval import eval        eval(FLAGS.folder_to_eval)
```

这里的关键点是：

- 训练和推理都走 `FVProcessor`，但推理结束后会自动触发评估；
- eval-only 则直接对已有 inference 目录重新跑评估。这使得整个系统形成三段式流水线：

```
# 来源：这里基于 src/main.py 与 README 简化train:  数据集样本 -> LLM生成SVA -> 与参考断言比较 -> Q-Tree追问错误 -> 保存 suggestions.pkl/qtrees.jsoninference:  新prompt -> 检索相似examples/suggestions/qtrees -> LLM生成或修正SVA -> JasperGold语法修复 -> 保存输出eval:  LLM输出目录 -> 文本相似度评估 + JasperGold功能性评估 -> CSV汇总
```

### 3.2 配置不是附属品，而是实验协议

> 在 `src/config.py` 中，训练和推理的行为被清晰分开。训练阶段关闭 RAG 与 JasperGold 主评估，重点生成 suggestion 与 Q-Tree：

```
# 来源：src/config.pyif global_task == 'train'and'nl2sva'in task:    # Training builds suggestions / Q-Trees; retrieval is disabled here.    use_RAG = False    use_JG = False    RAG_content = ['Suggestions']    # Number of self-improvement iterations per training sample.    num_iter = 25    train_temp_setting = "random"    if train_temp_setting == "random":        change_only_bleu_same = True        low_temp = 0        high_temp = 1.0    use_QTree = True    if use_QTree:        # Number of questions expanded per Q-Tree level.        question_tree_width = 3    PEC_method = 'JasperGold'
```

推理阶段则打开 RAG，并启用 Q-Tree enhanced retrieval：

```
# 来源：src/config.pyelif global_task == 'inference'and'nl2sva'in task:    use_RAG = True    similarity_str = ["Question"]    eval_content = ["similarity","jg"]    if use_RAG:        RAG_content = ['Examples', 'Suggestions']        if'Examples'in RAG_content:            prompting_instruction = 0            Examples_top_k = 1        if'Suggestions'in RAG_content:            Suggestions_top_k = 3            Suggestions_Reasoning = False            # Enable FVRuleLearner's Q-Tree-enhanced retrieval.            retrieval_on_ranking = True            if retrieval_on_ranking:                qtree_similarity_top_k = 5                qtree_ranking_mode = 'prompt'                rule_source = 'generate'                llm_score_weight = 0.5                operator_score_weight = 0.5                operator_pruning_threshold = 0.0            deduplication = True            operator_explanation = False            filter_functionality = True            retrieve_str = ["Suggestions"]            load_suggestions_path = DEFAULT_TRAIN_LOGDIR
```

这段配置很值得注意：它把“例子检索”和“规则检索”分开。

- Examples 是经典 in-context learning；
- **Suggestions 则是从历史错误修复中学习到的规则。**

当然，后者才是 FVRuleLearner 的核心。

## 四、Q-Tree：把一次错误拆成可学习的推理路径

### 4.1 为什么需要树，而不是一条反思文本

> 很多 LLM 自我修复系统会在失败后简单问一句：“请解释错误并给出建议。”**FVRuleLearner 选择了更结构化的路线：Q-Tree。它把生成断言与参考断言之间的差异，拆成多层问题：**

- 背景知识：自然语言、testbench、生成断言、参考断言、SVA 语法常识；
- 探索性问题：从 operator、timing、comparison 等不同角度提出问题；
- 具体分析问题：针对每个差异继续追问；
- 规则生成：把分析压缩成一条可复用规则。

核心数据结构在 `src/qtree_builder.py`：

```
# 来源：src/qtree_builder.pyclass QuestionLevel(Enum):    """Levels in the Q-Tree hierarchy"""    BACKGROUND = "background"    EXPLORATORY = "exploratory"    SPECIFIC_ANALYSIS = "specific_analysis"    RULE_GENERATION = "rule_generation"@dataclassclass QuestionNode:    """Node in the Q-Tree structure"""    id: str    question: str    level: QuestionLevel    parent_id: Optional[str] = None    children: List[str] = field(default_factory=list)    answer: Optional[str] = None    rules_generated: List[str] = field(default_factory=list)    metrics: Dict[str, Any] = field(default_factory=dict)class QTreeBuilder:    """Builds Q-Tree for assertion analysis and rule generation"""    def __init__(self, agents: Dict[str, Any], config_flags=None):        self.agents = agents        self.nodes: Dict[str, QuestionNode] = {}        self.tree_results = []        self.max_questions_per_level = getattr(config_flags, 'question_tree_width', 3) if config_flags else3        self._last_diversity_metrics = None        self._row_data = None        self.qa_time = 0.0
```

把它想象成一次形式验证专家 code review：专家不会只说“错了”，而会问“时序蕴含错了吗？”“比较操作符是否匹配四态逻辑？”“条件放在 antecedent 还是 consequent？”**这些问题形成树状结构，最终沉淀为规则。**

### 4.2 Q-Tree 构建流程

```
# 来源：src/qtree_builder.pydef build_qtree_for_assertion_pair(self,                                 generated_assertion: str,                                 reference_assertion: str,                                 row: Any,                                 metrics: Dict[str, Any]) -> List[QuestionNode]:    """    Build a Q-Tree to analyze differences between generated and reference assertions    Tree Structure:    Root: Background Knowledge (context about the problem)    Level 1: Exploratory questions about major differences    Level 2: Specific analysis questions for each difference    Level 3: Rule generation based on the analysis    """    tree_nodes = []    node_counter = 0    self._row_data = row    background_knowledge = self._get_background_knowledge(row, generated_assertion, reference_assertion)    exploratory_questions = self._generate_exploratory_questions(        generated_assertion, reference_assertion, metrics, row    )    for question in exploratory_questions:        node = QuestionNode(            id=f"explore_{node_counter}",            question=question,            level=QuestionLevel.EXPLORATORY,            parent_id=None,            metrics=metrics        )        node.answer = self._answer_exploratory_question(            question, generated_assertion, reference_assertion, row        )    # Level 2: Generate specific analysis questions    for parent_node in exploratory_nodes:        specific_questions = self._generate_specific_analysis_questions(            parent_node, generated_assertion, reference_assertion        )    # Level 3: Generate rules based on analysis    for parent_node in specific_nodes:        rules = self._generate_rules_from_analysis(            parent_node, generated_assertion, reference_assertion        )
```

这段代码的设计意味很强：Q-Tree 并不直接“修代码”，而是先把错误空间结构化。**对于 SVA 这种高度依赖操作符语义的语言，结构化追问比一次性反思更稳定**。比如：

- `|->` 与 `|=>` 是同样的“蕴含”吗？不是，一个是同周期重叠蕴含，一个是下一周期非重叠蕴含；
- `==` 与 `===` 只是风格差异吗？不是，后者是四态 case equality；
- `&&` 与 `||` 是否可凭语言直觉选择？不能，它决定 antecedent 是否过宽或过窄。

::: block-1
Q-Tree 的价值在于把这些差异变成“可解释的问题节点”，再把问题节点压缩为后续可检索的 rule。
:::

## 五、RAG 与增强检索：不是找相似答案，而是找相似错误修复经验

### 5.1 RAGAgent 如何组织文档

> FVRuleLearner 的 `RAGAgent` 支持 Examples 与 Suggestions 两类检索。Examples 用于找相似问答样例，Suggestions 用于找历史规则。初始化时会选择 embedding 模型，默认是 `all-MiniLM-L6-v2`。

```
# 来源：src/rag_agent.pyclass RAGAgent:    def __init__(self, documents, rag_type, embedding_model='sbert'):        self.delimiter = "<---->"        self.rag_type = rag_type        self.qtree_engine = None        if rag_type == "Suggestions":            if FLAGS.filter_functionality:                documents = [doc for doc in documents if self.check_functionality(doc)]                print(f"@@@DEBUG: Length of the filtered documents is @@@ {len(documents)} @@@")        self.original_documents = [doc.replace(self.delimiter, "\n").strip() for doc in documents]        self.documents = [self.extract_section(doc, FLAGS.similarity_str, full_string=True) for doc in documents]        if rag_type == "Suggestions":            self.retrieve_documents = [self.extract_section(doc, FLAGS.retrieve_str, full_string=True) for doc in documents]        elif rag_type == "Examples":            self.retrieve_documents = [self.extract_section(doc, ["all"], full_string=True) for doc in documents]        self.model = self._load_model(embedding_model)        self.embeddings = self._index_documents()
```

这里有一个重要过滤：`filter_functionality=True` 时，只保留 Functionality 为 1.0 的 suggestion。:mdnice-bold-italic[也就是说，项目默认不把所有失败经验都当知识，而只把经过功能性验证的修复经验放进检索池。 这一点对形式验证非常关键，因为错误规则比没有规则更危险。]

### 5.2 Q-Tree 增强检索的入口

> 当 `retrieval_on_ranking=True` 且 rag_type 是 Suggestions 时，普通向量检索会升级为 Q-Tree 检索。

```
# 来源：src/rag_agent.pydef retrieve_with_qtree(self, query, top_k=5, testbench=None):    """    Retrieve suggestions using Q-Tree based inference.    Supports multiple modes via FLAGS.qtree_retrieval_mode:    - 'prompt': Original Q-Tree inference    - 'initial_sva_generation': Generate SVA, hybrid ranking, operator pruning    - 'rule_generation': Generate rules from Q-Trees    """    if FLAGS.retrieval_on_ranking:        try:            from qtree_enhanced_retrieval import integrate_enhanced_retrieval            qtree_file = os.path.join(FLAGS.load_suggestions_path, "qtrees.json")            if os.path.exists(qtree_file):                with open(qtree_file, 'r') as f:                    qtree_data = json.load(f)            result = integrate_enhanced_retrieval(                query=query,                qtree_data=qtree_data,                testbench=testbench,                csv_df=csv_df,                top_k=top_k            )            if isinstance(result, dict):                suggestions = result.get('suggestions', [])                initial_sva = result.get('initial_sva', None)                operator_explanations = result.get('operator_explanations', {})                return {'suggestions': suggestions, 'initial_sva': initial_sva, 'operator_explanations': operator_explanations}
```

这段逻辑体现了一个很现实的工程判断：检索到的知识未必总是一段 suggestion；它也可能直接生成一条 initial SVA，或附带 operator explanations。因此返回值被设计成字典，包含 `suggestions`、`initial_sva` 和 `operator_explanations`。后续调用方根据结果选择“直接用初始断言”还是“把规则追加进 prompt 再生成”。

## 六、LLM 编排：轻量路径与 AutoGen 路径并存

### 6.1 FVProcessor 是流水线调度器

> `FVProcessor` 是训练和推理的主调度类。初始化阶段，它根据任务创建 benchmark launcher，并在推理时加载 examples 与 suggestions。

```
# 来源：src/fv_harwareagent_example.pyclass FVProcessor:    def __init__(self):        self.dataset_path = FLAGS.dataset_path        self.model_name = FLAGS.llm_model        self.agent_arch = FLAGS.agent_arch        self.num_icl_examples = FLAGS.num_icl        self.debug = FLAGS.debug        self.save_dir = FLAGS.save_dir        self.experiment_id = self.dataset_path.split(".csv")[0].split("/")[-1]        self.example_rag_agent = None        self.suggestion_rag_agent = None        self.assertion_generator = None        if FLAGS.global_task == 'train':            self.train_launcher = OrderedDict()            for task in FLAGS.src_examples:                bmark_launcher_src = self._create_bmark_launcher(task)                self.train_launcher[task] = bmark_launcher_src        elif FLAGS.global_task == 'inference':            self.bmark_launcher = self._create_bmark_launcher(FLAGS.task)
```

随后在 `main_fv()` 中创建 user/Coding/Reflection 等 agent。对于 OpenAI/Claude 主线，项目使用更轻量的直接调用路径；对于其他 gateway，则可以回落到 `HardwareAgent` 的 AutoGen 编排。

### 6.2 规则如何进入提示词

最关键的生成逻辑可以看 `_run_lightweight_nl2sva`。它先拼接 Examples，再检索 Suggestions。如果 Q-Tree 返回了 initial SVA 且没有 suggestion，则可以直接返回；否则把 suggestion 作为“Additional knowledge/suggestions”追加进 prompt。

```
# 来源：src/fv_harwareagent_example.pydef _run_lightweight_nl2sva(self, agents, system_prompt, fv_prompt, row):    agents["Coding"].system_message = system_prompt    message = fv_prompt    if getattr(FLAGS, "use_RAG", False):        full_question = f"Question: Create a SVA assertion that checks: {row.prompt}"        if self.example_rag_agent isnotNoneand"Examples"in getattr(FLAGS, "RAG_content", []):            retrieved_docs = self.example_rag_agent.retrieve(                full_question,                top_k=getattr(FLAGS, "Examples_top_k", 1),            )            examples_list = [f"- {doc}\n"for doc in retrieved_docs]            message = (                f"{message}\n\n"                "Additional context from similar documents:\n"                + "".join(examples_list)            )        if self.suggestion_rag_agent isnotNoneand"Suggestions"in getattr(FLAGS, "RAG_content", []):            retrieved_result = self.suggestion_rag_agent.retrieve(                full_question,                top_k=getattr(FLAGS, "Suggestions_top_k", 3),                testbench=row.testbench,            )            if isinstance(retrieved_result, dict):                retrieved_docs = retrieved_result.get("suggestions", [])                initial_sva = retrieved_result.get("initial_sva")            else:                retrieved_docs = retrieved_result            if len(retrieved_docs) == 0and initial_sva andnot getattr(FLAGS, "operator_explanation", False):                saver.save_stats("used_suggestions", 0)                return initial_sva            suggestions_list = []            seen_suggestions = set()            for doc in retrieved_docs:                cleaned_doc = re.sub(r"^Suggestions:\s*", "", str(doc).strip())                for suggestion in cleaned_doc.split("\n"):                    if getattr(FLAGS, "deduplication", True):                        if suggestion in seen_suggestions:                            continue                        seen_suggestions.add(suggestion)                    suggestions_list.append(suggestion)            appended_suggestions = "\n".join(suggestions_list)            if appended_suggestions:                saver.save_stats("used_suggestions", 1)                message = (                    f"{message}\n\n"                    "Additional knowledge/suggestions to follow/obey:\n"                    f"{appended_suggestions}"                )    response = initiate_chat_with_retry(        agents["user"],        agents["Coding"],        message=message,    )    return response
```

这就是 FVRuleLearner 相比“普通 RAG 生成”的不同：它检索的不是参考答案本身，而是“应该遵守的修复规则”。**从模型行为角度看，这相当于给 LLM 加了一层形式验证专家的经验笔记。**

## 七、JasperGold 闭环：文本像不等于功能对

### 7.1 Tcl 脚本定义形式评估语义

FVEval 下的 Tcl 脚本负责让 JasperGold 执行属性检查。以 human/machine 任务为例，脚本先分析生成的 `.sva` 文件，再调用 PEC 工具检查 LM 断言与参考断言的等价性。

```
# 来源：FVEval/tool_scripts/run_jg_nl2sva_human.tcl# Analyze property filesclear -allanalyze -clearanalyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva# Elaborate design and propertieselaborateclear -allinclude tool_scripts/pec/pec.tcleset signal_list [split $SIGNAL_LIST ","]prop_eq_checker $LM_ASSERT_TEXT $REF_ASSERT_TEXT "" "" $signal_list
```

这段 Tcl 很短，但它是整个项目“严肃性”的来源。没有形式工具，LLM 生成的 SVA 只能靠 BLEU、ROUGE、exact match 这类文本指标评估；有了 JasperGold，系统才能判断语法是否可解析、断言语义是否与参考属性等价或接近。

### 7.2 评估输出如何汇总

`FVEval/fv_eval/evaluation.py` 会同时生成文本相似度和 JG 评估结果，并在推理模式下合并。

```
# 来源：FVEval/fv_eval/evaluation.pydef run_evaluation(self):    for exp_name, result_list in self.llm_results:        if ("similarity"in FLAGS.eval_content and FLAGS.global_task == 'eval') or (FLAGS.global_task == 'inference'):            text_similiarty_eval_results = self.evaluate_text_similiarty(result_list)            text_similiarty_eval_results = [                asdict(r) for r in text_similiarty_eval_results            ]            text_similiarty_eval_results = pd.DataFrame(text_similiarty_eval_results)            text_similiarty_eval_results.to_csv(                f"{self.save_dir}/{exp_name}_sim.csv", index=False            )            results = text_similiarty_eval_results        if ("jg"in FLAGS.eval_content and FLAGS.global_task == 'eval') or (FLAGS.global_task == 'inference'):            jg_eval_results = self.evaluate_jg(result_list)            jg_eval_results = [asdict(r) for r in jg_eval_results]            jg_eval_results = pd.DataFrame(jg_eval_results)            jg_eval_results.to_csv(f"{self.save_dir}/{exp_name}_jg.csv", index=False)            results = jg_eval_results        if ("similarity"in FLAGS.eval_content and"jg"in FLAGS.eval_content and FLAGS.global_task == 'eval') or (FLAGS.global_task == 'inference'):            combined_results = pd.merge(                text_similiarty_eval_results,                jg_eval_results,                on=["experiment_id", "task_id", "model_name"],            )            # JasperGold Bug Quick Fix            combined_results.loc[combined_results['bleu'] == 1.0, 'functionality'] = 1.0            combined_results.loc[combined_results['bleu'] == 1.0, 'func_relaxed'] = 1.0
```

这里也有一个需要谨慎解读的细节：代码中存在 “JasperGold Bug Quick Fix”，当 BLEU 为 1.0 时将 functionality 与 func_relaxed 置为 1.0。这是工程修补逻辑，不应在文章中夸大为理论保证。更准确的说法是：项目同时使用文本指标和 JasperGold 指标，并对完全字符串匹配的情况做了保守修正。

## 八、SystemVerilog 资源：testbench 不是背景板，而是语义锚点

### 8.1 annotated_tb 承载验证上下文

仓库中有不少 `annotated_tb` SystemVerilog 文件，例如 FIFO、RAM、FSM、counter 等。这些文件不是普通示例，而是 NL2SVA 任务中的语义上下文：自然语言 prompt 往往依赖 testbench 中定义的抽象信号、状态寄存器、辅助 wire。

以 FIFO 片段为例：

```
// 来源：FVEval/data_nl2sva/annotated_tb/fifo_1r1w.svwire wr_push;wire rd_pop;wire tb_reset;assign tb_reset = (reset_ == 1'b0);wire fifo_full;assign wr_push = wr_vld && wr_ready;assign rd_pop = rd_vld && rd_ready;reg [DATA_WIDTH-1:0]             fifo_array [FIFO_DEPTH-1:0]; //fifo array - shift registerreg [FIFO_DEPTH_log2-1:0]        fifo_rd_ptr;                 //fifo array - rd_ptrwire                             actual_fifo_pop;             // actual pop == popreg                              fifo_empty;                  // fifo emptywire [DATA_WIDTH-1:0]            fifo_out_data;               // dout
```

如果一个 prompt 说“写入握手发生时……”，正确断言可能应该使用 `wr_push`，而不是机械拼接 `wr_vld && wr_ready`；如果涉及 reset，应该使用 testbench 中定义的 `tb_reset`。因此，SVA 生成不是孤立翻译，而是围绕 testbench 命名约定与辅助抽象进行语义对齐。

### 8.2 opencore 路径的特殊性

`nl2sva_opencore` 使用 `FVEval/data_1k/module_sva_nl_manual_editing.csv`，配置中还使用了不同的 Tcl：

```
# 来源：src/config.pyif task == "nl2sva_machine":    tcl_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_only_analysis.tcl")    tcl_eval_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_nl2sva_machine.tcl")elif task == "nl2sva_human":    tcl_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_only_analysis.tcl")    tcl_eval_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_nl2sva_human.tcl")elif task == "nl2sva_opencore":    tcl_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_only_syntax_opencore.tcl")    tcl_eval_file_path = os.path.join(ROOT, "FVEval", "tool_scripts", "run_jg_nl2sva_opencore.tcl")
```

这说明不同 benchmark 的评估条件并不完全相同。写技术解读时应避免把一个任务的评估方式泛化到所有任务。

## 九、知识如何落盘：suggestions 与 qtrees 是项目的长期记忆

### 9.1 Q-Tree 保存为机器与人都能读的格式

训练阶段产物包括 `suggestions.pkl`、`qtrees.pkl`、`qtrees.json`。其中 `qtrees.json` 是最适合调试和理解的格式。保存逻辑在 `src/saver.py`：

```
# 来源：src/saver.pydef save_qtree_correction(self, qtree_data, task_id, design_name=None, prompt=None, testbench=None, generated_sva=None, previous_metrics=None, final_metrics=None):    """Save a qtree that led to functionality correction"""    qtree_entry = {        'qtree_data': qtree_data,        'task_id': task_id,        'design_name': design_name,        'prompt': prompt,        'testbench': testbench,        'previous_metrics': previous_metrics,        'final_metrics': final_metrics,        'generated_sva': generated_sva    }    self.qtree_corrections.append(qtree_entry)def dump_qtree_corrections_to_disk(self):    """Save all qtree corrections to disk (similar to suggestions)"""    pkl_filename = os.path.join(self.logdir, "qtrees.pkl")    with open(pkl_filename, "wb") as pkl_file:        pickle.dump(self.qtree_corrections, pkl_file)    json_filename = os.path.join(self.logdir, "qtrees.json")    with open(json_filename, 'w') as f:        json.dump(self.qtree_corrections, f, indent=2)
```

这段代码的哲学可以概括为：一次修复不只是一个更好的 SVA，而是一条带上下文、指标、prompt、testbench、生成结果的“经验记录”。后续推理阶段再把这些记录检索出来，就形成了项目所谓 rule-learning pipeline。

### 9.2 与普通 RAG 的根本区别

普通 RAG 往往是“把相似文档塞进 prompt”。FVRuleLearner 的 RAG 更像一个经验库：

- `Examples` 提供相似输入输出；
- `Suggestions` 提供历史修复建议；
- `qtrees.json` 提供错误分析路径；
- `operator_explanations` 提供 SVA 操作符背景；
- `JasperGold` 提供语法与功能反馈。

因此，它的核心不是“检索更多上下文”，而是“检索被验证过的纠错经验”。

## 十、限制、边界与值得借鉴之处

### 10.1 需要谨慎表述的限制

- 第一，项目依赖外部商业 EDA 工具 JasperGold。README 明确要求 `jg` 在 PATH 中，并设置 `LM_LICENSE_FILE`、`CDS_LIC_FILE`。没有这些环境，语法与功能评估闭环无法完整复现。
- 第二，当前 release 主线是 prompt-only path under `src/`，并非一个即插即用的通用 RTL 形式验证平台。它围绕 NL2SVA benchmark、CSV 数据集、预设 Tcl 脚本与固定配置组织。
- 第三，配置中存在硬编码 training IDs，用于避免数据泄漏。这对论文实验是必要的，但也意味着迁移到新数据集时需要重新设计 split 与检索池。
- 第四，部分工程逻辑体现研究原型特征，例如 eval 中的 BLEU=1.0 功能性修正、opencore Tcl 中某些 analyze/elaborate 语句被注释、`fv_harwareagent_example.py` 文件名存在拼写历史痕迹。这些都不影响理解核心思想，但提醒我们：应将其视为研究代码，而不是成熟商业产品。

### 10.2 对 AI+EDA 系统的启发

> FVRuleLearner 最值得借鉴的，不是某个具体 prompt，而是三个系统设计原则：

1. **把失败转化为结构化知识**：失败样本如果只用于重试，就是成本；如果拆成 Q-Tree，就是经验资产。
2. **让工具评估参与学习闭环**：JasperGold 不只是最终打分器，也是规则可信度的来源。
3. **检索规则而非照抄答案**：对于形式验证这类高精度任务，直接复用答案容易错位，复用“操作符级修正规则”更具泛化意义。

### 10.3 如果把它比作编译器

> 可以把 FVRuleLearner 看作一台“**面向断言生成的经验编译器** ”：

- 自然语言 prompt 是源程序；
- LLM 是不稳定的前端生成器；
- Q-Tree 是错误分析 IR；
- suggestions/qtrees 是优化 pass 的知识库；
- RAG 是 pass manager；
- JasperGold 是后端 verifier；
- eval CSV 是最终编译报告。

传统编译器把源码编译成机器码，并用类型系统、优化器、后端约束保证正确性；FVRuleLearner 则把自然语言编译成 SVA，并用规则学习、检索增强和形式工具评估尽量缩小语义偏差。

## 结语：真正的突破在于让形式验证经验可积累

> LLM 进入 EDA 的早期阶段，最容易被展示的是“它能写出一段代码”。**但对形式验证而言，能写只是起点，能被证明、能被修正、能从修正中学习，才是走向可用系统的关键**。FVRuleLearner 的贡献恰好在这里：:mdnice-bold-italic[它没有把 SVA 生成当成一次性翻译，而是把每一次错误拆解为 operator-level 问题，把每一次修复沉淀为 Q-Tree 和 suggestion，再在新任务中通过 RAG 与排序机制复用这些经验。]

这套思路也许不会让 LLM 一夜之间成为形式验证专家，但它提供了一条更工程化的路线：**让模型不再孤立地“猜答案”，而是在可追踪、可验证、可积累的规则空间中生成答案**。

::: block-1
对于 AI 辅助 EDA 来说，这比单纯提升一次生成的命中率更重要，因为它指向的是一种长期学习机制：把专家经验、工具反馈和模型推理，合并为一个持续增值的验证知识系统。
:::

参考资料

[1] Operator-Level Reasoning Tree (OP-Tree)-Based Rules Learning for Formal Verification: *https://arxiv.org/pdf/2604.03245*
