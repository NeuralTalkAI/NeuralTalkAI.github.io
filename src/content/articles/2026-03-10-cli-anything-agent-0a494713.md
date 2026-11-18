关键词：**AI Agent**、软件自动化、**命令行接口(CLI)、** 大模型应用、**零代码适配**

一行命令，让任意软件接入 OpenClaw、nanobot、Cursor、Claude Code 等 Agent 框架。

> 如今，AI Agent（智能体）正在以前所未有的速度演进。从早期的文本问答，到如今的复杂推理，大语言模型（LLM）的“大脑”已经足够聪明。然而，当我们需要 Agent 帮我们完成一项具体的现实任务——比如“用 Blender 渲染一个带物理碰撞的 3D 场景”、“在 GIMP 中批量去背图片”或者“用 LibreOffice 生成一份包含财报数据的 PDF”时，现有的 AI Agent 往往束手无策。

**痛点显而易见：今天的软件是为“人类”设计的（GUI界面），而明天的主要用户将是“AI Agent”。**

业界目前的解决方案主要有两类，但都存在致命缺陷：

1. **基于视觉与 GUI 自动化的 RPA（RPA/Pixel-clicking）：** 这种方式通过截图让大模型识别按钮并模拟鼠标点击。极其脆弱（Brittle），UI 更新、分辨率改变、甚至是弹出的系统通知都会让流程瞬间崩溃。
2. **基于受限的 API 调用：** 许多专业级桌面软件根本没有开放标准的 REST API，哪怕有，其覆盖面也常常不及软件本身功能的 10%。

![文章插图](/articles/2026-03-10-cli-anything-agent-0a494713/images/01.gif)

![文章插图](/articles/2026-03-10-cli-anything-agent-0a494713/images/02.webp)

- CLI-Anything: 让所有软件都能被 Agent 驱动
- 今天的软件为人而生👨‍💻，明天的用户是 Agent🤖。CLI-Anything：连接 AI Agent 与全世界软件的桥梁
- https://github.com/HKUDS/CLI-Anything

> 为了填补“AI Agent”与“现实世界复杂软件”之间的鸿沟，香港大学数据智能实验室（HKUDS）开源了一项极具革命性的工作——`CLI-Anything`。

它不仅是一个工具，更是一套方法论，其核心愿景是：**通过一条命令，将世界上任何带有代码库的复杂软件，自动转化为 AI Agent 原生可控的形态（Agent-Native）。**

## 本文目录

- 🚀 CLI-Anything 极简快速上手
  - 环境要求
- 二、CLI-Anything 解决的核心问题与技术难点
  - 2.1 为什么让大模型控制软件这么难？
  - 2.2 CLI-Anything 的破局思路
- 三、 核心创新点与技术原理解析
  - 创新点一：7 阶段全自动生成管线 (7-Phase Automated Pipeline)
  - 创新点二：零妥协的“真实后端渲染” (Authentic Software Integration)
  - 创新点三：Agent-First 的设计哲学——统一 REPL 与结构化 JSON
- 四、 架构拆解
  - 图解1：CLI-Anything 宏观架构设计
  - 图解2：CLI-Anything 的交互打字演示
  - Benchmark：100% 通过率的工业级验证
- 五、 迈向“Agent-Computer Interaction (ACI)”的新纪元

## 🚀 CLI-Anything 极简快速上手

### 环境要求

- Claude Code（支持插件）
- Python 3.10+
- 目标软件（GIMP/Blender/LibreOffice 等）

1. 添加插件市场

```
/plugin marketplace add HKUDS/CLI-Anything
```

1. 安装插件

```
/plugin install cli-anything
```

1. 一键生成 CLI

```
/cli-anything <软件路径/仓库地址># 示例：/cli-anything ./gimp
```

1. 安装并使用

```
# 安装到 PATHcd <软件名>/agent-harness && pip install -e .# 查看帮助cli-anything-<软件名> --help# 直接使用cli-anything-gimp project new --width 1920 --height 1080 -o poster.json# 进入交互式 REPLcli-anything-gimp
```

## 二、CLI-Anything 解决的核心问题与技术难点

### 2.1 为什么让大模型控制软件这么难？

> 传统的桌面和生产力软件（如 GIMP、Audacity、Kdenlive）在架构上存在“逻辑层与渲染层/UI层高度耦合”的历史遗留问题。

- **状态管理的复杂性（State Management）：** 软件是包含隐式状态的（如“当前选中的图层”、“当前播放的时间轴”），而 LLM 默认是无状态的（Stateless）。
- **非标准化的工程文件：** 每种软件都有自己独有的、结构极其复杂的工程文件格式（如 Kdenlive 的 MLT XML，LibreOffice 的 ODF，Blender 的 .blend），AI 直接手写这些二进制或超长 XML 文件的成功率几乎为零。

### 2.2 CLI-Anything 的破局思路

> CLI-Anything 没有选择去教大模型“如何点击按钮”，也没有选择让开发者去“手动写 API 封装”。它构建了一个完全自动化的流水线，利用大模型自身的代码理解和生成能力，**为这些软件逆向生成一套标准化、结构化、带状态维护的通用命令行接口（CLI）。**

只要有了 CLI，大模型就可以通过类似于编写 Bash 脚本的方式，以最高效、最确定（Deterministic）的文本形式去调度任何软件的底层能力。

## 三、 核心创新点与技术原理解析

`CLI-Anything` 的成功不仅在于思路的转变，更在于其工程实现上的三大核心创新。我们结合项目的底层代码机制，详细拆解其技术亮点。

### 创新点一：7 阶段全自动生成管线 (7-Phase Automated Pipeline)

> CLI-Anything 被设计为一个 Claude Code 插件（位于 `cli-anything-plugin` 目录下），它只需一句简单的指令 `/cli-anything <软件路径或仓库>`，便会自动拉起一个高达 7 个阶段的全自动流水线。

这 7 个阶段完美模拟了资深研发工程师的工作流：

1. **分析 (Analyze)：** 扫描目标软件的源码，映射 GUI 动作与底层 API / 文件格式的对应关系。
2. **设计 (Design)：** 架构 CLI 命令组（Command Groups），设计对象数据模型（State Model）和输出格式。
3. **实现 (Implement)：** 构建基于 `Click` 的 Python CLI 接口，集成 REPL 和撤销/重做（Undo/Redo）机制。
4. **规划测试 (Plan Tests)：** 创建 `TEST.md`，制定单元测试与端到端（E2E）测试的测试用例。
5. **编写测试 (Write Tests)：** 生成海量测试脚本，覆盖项目创建、文件解析、真实后端调用等。
6. **文档化 (Document)：** 更新文档与测试结果。
7. **发布 (Publish)：** 生成 `setup.py`，实现零配置安装，将软件直接注入到系统的 PATH 中。

**具体映射与伪代码解析：** 在这个过程中，`cli-anything-plugin` 实际上充当了智能体的“导师”。它内置了一套名为 `HARNESS.md` 的标准化 SOP（标准作业程序）。

```
# 伪代码：CLI-Anything 7阶段生成管线的核心调度逻辑 (参考 cli-anything-plugin 机制)def generate_cli_harness(software_path):    # 阶段 1-2：分析与设计    ast_tree = parse_source_code(software_path)    api_endpoints = extract_backend_apis(ast_tree)    cli_architecture = llm_architect_design(api_endpoints)        # 阶段 3：实现核心代码 (利用 Click 框架)    for command in cli_architecture.commands:        generate_click_command(            name=command.name,            args=command.args,            # 关键：强制要求所有命令支持 --json 标志            enforce_json_output=True        )            # 阶段 4-5：生成基于 pytest 的多层级验证    generate_test_suite(cli_architecture)        # 安装包生成    create_setup_py(software_name=extract_name(software_path))    print(f"Success! CLI for {software_path} is ready for Agents.")
```

这种自动化管线的创新在于：**授人以鱼不如授人以渔**。它不仅给了开发者 9 个现成的软件接口，更赋予了社区将成千上万个冷门/定制化软件转化为 Agent-Native 工具的能力。

### 创新点二：零妥协的“真实后端渲染” (Authentic Software Integration)

> 现有的许多 Agent 工具往往会用 Python 库去“平替”原软件的功能（比如用 `Pillow` 去代替 Photoshop 处理图片），这会导致高级功能的严重丢失。

CLI-Anything 的**核心护城河** 在于它采取了“Native Backend”方案。CLI 本身只负责两件事：**维护状态** 与**生成合法的工程文件**，而最终的渲染与执行，100% 委托给真正的软件原生后端。

- 对于 **LibreOffice**：生成的 CLI（`cli_anything_libreoffice`）在组装完 ODF 文件结构后，会通过 `libreoffice --headless --convert-to pdf` 进行真实渲染，确保图表、宏等企业级格式毫无偏差。
- 对于 **Blender**：生成合法的 Python 渲染脚本，再通过 `blender --background --python render.py` 调用内置的 `bpy` 进行光线追踪。

```
# 伪代码：真实后端调用的实现范例 (以 LibreOffice 导出 PDF 为例)import subprocessdef export_document(project_file_path, output_pdf_path):    """    不使用劣质的第三方库解析，而是直接调起 LibreOffice 无头模式 (Headless Mode)。    确保所见即所得，保留了最原汁原味的专业级渲染能力。    """    cmd = [        "soffice",         "--headless",         "--convert-to", "pdf",         "--outdir", os.path.dirname(output_pdf_path),        project_file_path    ]    result = subprocess.run(cmd, capture_output=True)    if result.returncode == 0:        return {"status": "success", "file": output_pdf_path}    else:        raise RealBackendError("LibreOffice 渲染失败")
```

### 创新点三：Agent-First 的设计哲学——统一 REPL 与结构化 JSON

> 大模型处理文本极其高效，但在交互上需要一定的“记忆”和“环境上下文”。为了降低 Agent 学习使用新软件的门槛，CLI-Anything 为所有生成的 CLI 注入了统一的 REPL (Read-Eval-Print Loop) 会话界面。

在项目的 `cli-anything-plugin/repl_skin.py`（或各个软件下的 `utils/repl_skin.py`）文件中，我们可以看到 HKUDS 团队精心设计的统一下拉/展示层：

```
# 真实代码片段 (节选自 cli-anything-plugin/repl_skin.py)class ReplSkin:    def __init__(self, software_name, version="1.0.0"):        self.software_name = software_name        # 统一不同软件的高亮主题色，降低人类与Agent调试时的认知负担        self.accent_color = _ACCENT_COLORS.get(software_name, _DEFAULT_ACCENT)            def prompt(self, project_name=None, modified=False):        # 统一的状态提示符，包含当前操作的项目和修改状态        # 类似: blender[MyScene]*>         state_indicator = f"[{project_name}]"if project_name else""        mod_indicator = "*"if modified else""        returnf"{self.software_name}{state_indicator}{mod_indicator}> "
```

此外，**所有命令均原生支持 `--json` 输出**。当人类交互时，CLI 会渲染漂亮的终端表格和进度条；当检测到是 Agent 挂载时，CLI 返回扁平化的 JSON。这种“多模态”输出完美解决了传统 CLI 工具文本难以结构化解析的痛点，彻底根除了繁琐的正规表达式（Regex）匹配提取工作。

## 四、 架构拆解

> 为了让系统表现具有说服力，作者在 README 中提供了极为详实的 Benchmark 测试与架构图。

### 图解1：CLI-Anything 宏观架构设计

![文章插图](/articles/2026-03-10-cli-anything-agent-0a494713/images/03.webp)

::: image-caption
上图展示了 CLI-Anything 的核心架构。左侧是传统的“黑盒”GUI 软件；右侧则是大模型 Agent。中间通过 CLI-Anything 搭建了桥梁。你可以看到，中间层包含了完整的 `Project State` (项目状态管理) 和双模接口：针对人的 `Terminal UI` 和针对 Agent 的 `JSON RPC`。这种分离设计确保了 Agent 可以像调用 API 一样丝滑地控制复杂的渲染管线。
:::

### 图解2：CLI-Anything 的交互打字演示

![文章插图](/articles/2026-03-10-cli-anything-agent-0a494713/images/04.webp)

::: image-caption
上图直观演示了 Agent 是如何工作的。不需要控制鼠标挪动，Agent 只需要像开发者一样敲击命令：`cli-anything-gimp project new`、`cli-anything-gimp layer add`。这种“文本态”的命令序列天生契合大语言模型 Next-token Prediction 的工作机理，容错率极高，且自带极强的逻辑可解释性。
:::

### Benchmark：100% 通过率的工业级验证

> 最令人震撼的是项目的测试数据。这不是一个仅仅停留在论文概念上的玩具，而是一个经历了 **1,436 个严苛测试用例** 的工程巨作。

测试集涵盖了 9 个跨领域的顶级开源复杂软件，涉及图像编辑、3D建模、矢量图、音频、视频剪辑以及办公套件：

| 软件名称 (Software) | 所属领域 (Domain) | CLI 命令入口 | 核心调用的底层渲染后端 | 测试用例数 |
| --- | --- | --- | --- | --- |
| **GIMP** | 图像编辑 | `cli-anything-gimp` | Pillow + GEGL/Script-Fu | ✅ 107 |
| **Blender** | 3D建模与渲染 | `cli-anything-blender` | bpy (Python scripting) | ✅ 208 |
| **Inkscape** | 矢量图绘制 | `cli-anything-inkscape` | Direct SVG/XML 操作 | ✅ 202 |
| **Audacity** | 音频制作 | `cli-anything-audacity` | Python wave + sox | ✅ 161 |
| **LibreOffice** | 办公套件 (Doc/XLS) | `cli-anything-libreoffice` | ODF 生成 + 无头 LO 进程 | ✅ 158 |
| **OBS Studio** | 直播与录屏 | `cli-anything-obs-studio` | JSON scene + obs-websocket | ✅ 153 |
| **Kdenlive** | 视频剪辑 | `cli-anything-kdenlive` | MLT XML + melt renderer | ✅ 155 |
| **Shotcut** | 视频剪辑 | `cli-anything-shotcut` | Direct MLT XML + melt | ✅ 154 |
| **Draw.io** | 拓扑图与架构图 | `cli-anything-drawio` | mxGraph XML + draw.io CLI | ✅ 138 |
| **汇总 (TOTAL)** | - | - | - | **✅ 1,436** |

以上总计 1,436 个测试用例（包含 1,011 个独立单元测试和 425 个端到端 E2E 测试），实现了 **100% 的惊人通过率**。

这意味着什么？

这代表着通过 `CLI-Anything` 自动生成的 CLI 接口并不是粗糙的半成品，而是具备**产品级稳定性 (Production-Grade)。**

::: block-1
其端到端测试（E2E tests (true backend)）严格验证了的工程文件（如 ZIP 封装的 ODF、MLT XML）丢给原生软件渲染时，不仅不会报错，甚至还会对输出的魔数（Magic Bytes）、像素矩阵、音频 RMS 峰值进行严格效验（如验证输出 PDF 首行是否包含 `%PDF-`）。这一严谨的测试矩阵彻底打破了以往 AI Agent 演示中“中看不中用”的窘境。
:::

## 五、 迈向“Agent-Computer Interaction (ACI)”的新纪元

- 计算机科学的历史，就是一部交互方式不断抽象的演化史。
- 从最初的纸带打孔，到命令行的出现，再到施乐与苹果开创的 GUI (图形用户界面)，人类与计算机的沟通（Human-Computer Interaction, HCI）越来越直观。

> 但随着 AI Agent 的爆发，GUI 反而成了束缚智能体手脚的牢笼。我们不需要让拥有超越常人代码理解能力的 LLM 去“痛苦地学习如何移动鼠标寻找一个带有齿轮图标的设置按钮”。文本，尤其是结构化的命令行，才是 AI 的第一母语。

**HKUDS 团队带来的 CLI-Anything，标志着我们正式向“Agent-Computer Interaction (ACI)”的时代迈进。** 它以一种四两拨千斤的方式，把所有现存的、服务于人类的经典软件资产，瞬间平滑地迁移到了智能体生态之中。无论是企业级的业务报表自动化，还是复杂的 3D 模型批量化渲染，只需一句 `/cli-anything`，大模型的触角将延伸到数字世界的每一个角落。

对于所有大模型应用开发者、AI Agent 从业者以及自动化工程师来说，`CLI-Anything` 绝对是今年最不容错过的现象级开源项目。

> **开源项目指路** 🔗 **GitHub Repository:** `HKUDS/CLI-Anything`如果你觉得这个项目对你的 Agent 开发有启发，别忘了去给研究团队点个 Star (⭐)！
