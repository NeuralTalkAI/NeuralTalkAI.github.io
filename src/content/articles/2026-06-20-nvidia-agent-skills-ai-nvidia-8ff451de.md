关键词：**Agent Skills**、技能签名验证、**自动化同步管线**、插件分发、:mdnice-bold-italic[AI 智能体治理]

> 当你在 Claude Code 里对 AI 说"帮我用 cuOpt 解一个线性规划问题"，**智能体不是胡编乱造，而是精准地按照 NVIDIA 官方文档的最佳实践**，一步步帮你调用 API、配置服务器、提交求解。:mdnice-bold-italic[它为什么能做到？因为它脑子里装了一份经过 NVIDIA 密码学签名认证的"技能手册"。]

![文章插图](/articles/2026-06-20-nvidia-agent-skills-ai-nvidia-8ff451de/images/01.webp)

- NVIDIA Agent Skills：技能是可移植指令集，用于指导 AI Agent 以最优方式使用英伟达软件，涵盖 CUDA-X 函数库、AI BluePrint 以及各类平台工具。:mdnice-bold-italic[本代码仓库仅作为目录汇总：各项技能由对应产品代码仓库独立维护，并通过自动化同步流水线每日镜像至此处。平台会持续新增各类技能，欢迎时常查看获取更新。我们以开源形式搭建这套基础架构，欢迎各界参与贡献。如需了解后续规划，可查阅项目路线图。]
- **项目代码：https://github.com/NVIDIA/skills**
- 文档：https://docs.nvidia.com/skills
- 博客：NVIDIA Verified Agent Skills: Capability Governance for AI Agents[1]

::: related-title
**相关推荐**
:::

::: related-links
- **[cann-learning-hub：八章教程、二十五篇博客、两大SKILL，撑起昇腾 NPU 开发者的完整知识图谱](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902138&idx=1&sn=878d817c8462758c2c6e188739ad57c3&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**
- **[技能即记忆：AutoSkill 让 LLM 智能体从交互经验中自我进化，实现终身学习，代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898295&idx=1&sn=6bf15a2f47d940cea0165f5d1f50aa61&scene=21#wechat_redirect)**

:::

> 这份手册来自 `NVIDIA/skills` 仓库——**一个看似只是 Markdown 文件集合的 GitHub 项目，背后却是一套精密的工程系统：30 多个产品团队在各自仓库里维护技能源码**，GitHub Actions 每日自动同步、合规检查、签名校验，最终把上百项经过安全扫描的技能推送到 Claude Code、Codex、Cursor、Kiro 等主流 AI 编程智能体中。

今天我们就来拆解这个系统是如何运转的。

![文章插图](/articles/2026-06-20-nvidia-agent-skills-ai-nvidia-8ff451de/images/02.webp)

::: image-caption
技能资产从开发到全域分发的八步标准化管控流水线，整体划分三大阶段。前两步是产品团队自主负责的创作阶段（AUTHOR）：1 把代码存入源码仓库，2 完成代码与文档人工评审。中间 3 至 6 步为上架前核验关卡（VERIFY），是核心风控环节，依次开展安全漏洞扫描、合规风险评估、生成标准化技能信息卡、完成数字签名，层层校验资产安全与合规性。最后 7、8 步属于发布分发阶段（PUBLISH）：核验通过的资产录入统一资产目录，再同步分发至各业务渠道。整套流程覆盖技能资产全生命周期，分层隔离开发、审核、安全校验、发布环节，前置拦截各类风险，确保上架资产可信、可追溯、可统一分发。
:::

![文章插图](/articles/2026-06-20-nvidia-agent-skills-ai-nvidia-8ff451de/images/03.webp)

::: image-caption
各类技能目录按照用户所需完成的业务类型划分技能，而本代码仓库留存着各项技能的产品归属与原始来源信息。清楚自身业务任务时，可直接依托对应工作流查找技能；若需要完整的最新技能清单，可通过 skills.sh 工具或 NVIDIA Build 平台获取。
:::

## 本文目录

- 快速上手
- 一、这不是一个普通的文档仓库
  - 1.1 什么是 Agent Skill？
  - 1.2 仓库的宏观拓扑
- 二、组件注册与同步管线：让 30 个团队不打架
  - 2.1 `components.d/` 的巧妙设计
  - 2.2 同步管线的合规门禁
- 三、签名验证——技能的密码学身份证
  - 3.1 为什么需要签名？
  - 3.2 内容完整性检查的实现
- 四、插件构建系统——从技能到可安装的产品
  - 4.1 双模式物化策略
  - 4.2 多市场分发
  - 4.3 CI 漂移防护
- 五、README 自动重建——配置驱动文档
  - 5.1 从 YAML 到 Markdown 表格的流水线
  - 5.2 临时例外的优雅处理
- 六、安全扫描——技能的"安检门"
- 七、全景回望——一个工程范式的启示

![文章插图](/articles/2026-06-20-nvidia-agent-skills-ai-nvidia-8ff451de/images/04.webp)

## 快速上手

> 安装一个 NVIDIA 官方技能只需要一条命令：

```
npx skills add nvidia/skills
```

CLI 会自动弹出交互式菜单让你选择技能和安装目标。如果你已经知道要装什么，跳过交互：

```
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api-python --yes
```

针对特定 AI 智能体安装：

```
npx skills add nvidia/skills \  --skill cuopt-numerical-optimization-api-python \  --agent claude-code \  --agent codex
```

浏览可用技能目录：

```
npx skills add nvidia/skills --list
```

更多高级用法（全局安装、手动复制、更新与移除等），参考 `Advanced installation`[2]。

## 一、这不是一个普通的文档仓库

### 1.1 什么是 Agent Skill？

> 在 AI 智能体时代，"技能"（Skill）不是一段可执行代码，而是一份**结构化的指令集**。说白了，它是写给 AI 智能体看的"操作手册"——告诉智能体在遇到特定任务时应该怎么做、调用哪些 API、遵循哪些最佳实践。

每个技能的核心是一个 `SKILL.md` 文件。这个 Markdown 文件的头部用 YAML frontmatter 声明了技能的名称和描述，正文则是详细的指令内容。智能体在启动时扫描到 `SKILL.md`，加载其元数据；当用户的任务匹配时，再加载完整指令——这就是规范中所说的**渐进式披露（Progressive Disclosure）** 模型。

::: block-1
NVIDIA/skills 仓库遵循开放的 Agent Skills Specification[3]，这意味着同一份技能文件，可以被 Claude Code、Codex、Cursor、Kiro 等不同的 AI 智能体框架识别和加载。**一次编写，处处运行** ——这个在传统软件领域喊了几十年的口号，在 AI 智能体的技能层正在成为现实。
:::

### 1.2 仓库的宏观拓扑

整个仓库的目录结构乍看是扁平的，实际上有着精巧的分层设计：

```
NVIDIA/skills/├── skills/                   # 核心：100+ 个经验证的技能目录（每日同步）├── components.d/             # 产品注册表：每个产品一个 YAML 文件├── plugins/                  # 打包后的插件发行包（面向 Claude/Codex）├── plugins.d/                # 插件构建配置├── .github/workflows/        # CI/CD：同步、签名校验、README 生成├── .github/scripts/          # 自动化脚本：README 重建、插件构建、完整性校验├── docs/                     # 长文档（通过 Fern 发布到 docs.nvidia.com/skills）├── nv-agent-root-cert.pem    # OMS 签名验证的信任锚点└── skills.sh.json            # Skills.sh 市场分组配置
```

关键洞察是：**技能内容不在这个仓库里编写**。30+ 个产品团队在各自的源仓库（如 `NVIDIA/cuopt`、`rapidsai/cudf`、`ai-dynamo/dynamo`）中维护技能，然后通过自动化管线同步到这里。这个仓库本质上是一个**经过治理的分发中心**。

## 二、组件注册与同步管线：让 30 个团队不打架

### 2.1 `components.d/` 的巧妙设计

> 当 30 多个产品团队同时往一个仓库提交代码时，最大的痛点是什么？合并冲突。NVIDIA 的解法非常优雅——`components.d/` 目录采用了**每个产品一个文件** 的注册模式。

以 cuOpt 为例：

```
# 来源：components.d/cuopt.ymlname:cuOptrepo:NVIDIA/cuoptdescription:GPU-acceleratedoptimization—vehiclerouting,linearprogramming...skills:-path:skills/cuopt-developer/    catalog_dir:cuopt-developer-path:skills/cuopt-install/    catalog_dir:cuopt-install-path:skills/cuopt-numerical-optimization-api-python/    catalog_dir:cuopt-numerical-optimization-api-python# ... 共 12 个技能
```

这个 YAML 声明了三件事：源仓库在哪（`NVIDIA/cuopt`）、哪些目录包含技能（`path`）、以及技能在目录中的名字（`catalog_dir`）。一个新产品要上架？创建一个 YAML 文件，开个 PR——就这么简单。由于每个团队只修改自己的文件，PR 之间永远不会冲突。

同步工作流在运行时，用 `yq` 把所有 YAML 聚合成一个临时文件：

```
# 来源：.github/scripts/regenerate-readme.shyq ea '[.] | {"components": .}' components.d/*.yml > /tmp/components.aggregated.yml
```

这条看似简单的命令背后是一个关键的架构决策：**配置去中心化存储，运行时中心化聚合**。它让团队自治与系统一致性得到了完美平衡。

### 2.2 同步管线的合规门禁

> 同步不是简单的 `rsync`。核心工作流 `sync-skills.yml`（长达近 48000 字节！）在每次同步时执行一系列严格的合规检查。每个技能必须同时携带以下四件"护照"才能进入目录：

1. **`SKILL.md`** —— 技能指令本体
2. **`skill-card.md`** —— 技能身份与治理卡
3. **`skill.oms.sig`** —— OMS 格式的分离签名文件
4. **评估数据集** —— 位于 `evals/evals.json` 等路径下的基准测试数据

缺少任何一项，该技能会被**直接丢弃**，不会出现在最终发布的目录中。这种"宁缺毋滥"的策略确保了进入目录的每一个技能都是完整的、可审计的。

## 三、签名验证——技能的密码学身份证

### 3.1 为什么需要签名？

> 传统软件分发有包管理器的哈希校验和 GPG 签名。但 AI 智能体技能是一个全新领域——它本质上是一堆 Markdown 和脚本文件组成的目录。怎么证明你安装的技能确实是 NVIDIA 发布的那份，中间没有被篡改？

NVIDIA 选择了 OpenSSF 的 Model Signing（OMS）格式。OMS 扩展了 Sigstore 的 bundle 规范，让签名可以覆盖**整个目录树**，而不仅仅是单个文件。每个技能目录下的 `skill.oms.sig` 就是这样一个分离签名文件。

验证过程非常直观：

```
pip install model-signingmodel_signing verify certificate SKILL_DIR \  --signature SKILL_DIR/skill.oms.sig \  --certificate_chain nv-agent-root-cert.pem \  --ignore_unsigned_files
```

这条命令做了什么？它用仓库根目录下的 `nv-agent-root-cert.pem`（NVIDIA 的信任锚点证书）验证签名的真实性，然后逐文件比对签名中记录的哈希值与磁盘上的实际内容。

### 3.2 内容完整性检查的实现

> 仓库中有一个精妙的 Python 脚本专门负责内容完整性校验。它的核心逻辑是：解码 OMS 签名中的 DSSE/in-toto 负载，取出每个签名文件的 SHA-256 摘要，然后与磁盘上的实际文件逐一比对。

```
# 来源：.github/scripts/verify_content_integrity.pydef signed_resources(sig_path: Path) -> list[dict]:    """解码 DSSE/in-toto 负载，返回签名资源列表"""    bundle = json.loads(sig_path.read_text())    payload = base64.b64decode(bundle["dsseEnvelope"]["payload"])    statement = json.loads(payload)    return statement["predicate"]["resources"]def verify_skill(skill_dir: Path) -> list[str]:    """对比签名中记录的文件哈希与磁盘上的实际哈希"""    sig = skill_dir / SIG_NAME    resources = signed_resources(sig)    problems = []    for res in resources:        name = res["name"]        want = res["digest"]        # 签名时记录的 SHA-256        target = skill_dir / name        ifnot target.is_file():            problems.append(f"{name}: 签名中有记录，磁盘上缺失")            continue        got = hashlib.sha256(target.read_bytes()).hexdigest()        if got != want:            problems.append(f"{name}: 哈希不匹配，内容已漂移")    return problems
```

这段代码有一个值得注意的设计：它区分了 **PR 范围检查** 和**全量扫描** 两种模式。PR 触发时，只检查变更涉及的技能目录（速度快）；定时任务或手动触发时，扫描全量目录（兜底网）。这体现了经典的 CI 设计思想——**快速反馈 + 周期性全量保障**。

```
# 来源：.github/scripts/verify_content_integrity.pydef main() -> int:    event = os.environ.get("GITHUB_EVENT_NAME", "")    if event == "pull_request":        # PR 模式：只检查变更涉及的技能目录        targets = changed_skill_dirs(base_sha, head_sha)    else:        # 全量模式：扫描整个 skills/ 目录        targets = all_skill_dirs()
```

这里有一个容易被忽略的风险场景：团队 A 在源仓库签名了 commit 1 的技能内容，然后又提交了几个新 commit 改动了文件，但没有重新签名。同步管线拉取了最新内容，而签名还是旧的——**签名真实存在且可验证，但内容已经不匹配了**。这就是所谓的"签名漂移"问题，而这个脚本正是用来捕获它的

## 四、插件构建系统——从技能到可安装的产品

### 4.1 双模式物化策略

> 技能要真正进入用户的编程环境，需要被打包成不同 AI 智能体框架能识别的插件格式。`build-plugins.py` 脚本承担了这个角色，它支持两种截然不同的**物化策略**：

```
# 来源：.github/scripts/build-plugins.pyVALID_MATERIALIZATIONS = ("copy", "symlink")def materialize_skills(plugin_name, entries, mode="copy"):    """    copy:    rsync 每个技能到 plugins/<name>/skills/<skill>/             生成真实文件，体积更大，但兼容 Codex（Codex 0.132 静默丢弃符号链接）    symlink: 创建相对符号链接指向 skills/ 下的规范位置             无重复 SKILL.md，自动与主目录保持同步             兼容 Claude install 和 npx skills add，不兼容 Codex 本地安装    """
```

为什么需要两种模式？因为不同的 AI 智能体客户端对文件系统的处理方式不同。Codex 在本地安装时会静默丢弃符号链接，所以必须用 `copy` 模式生成真实文件。而 Claude Code 支持符号链接，用 `symlink` 模式可以节省存储空间并自动保持同步。这种**针对分发渠道定制物化策略** 的做法，展现了工程团队对下游生态的深入了解。

### 4.2 多市场分发

> 构建脚本不仅生成插件目录，还自动维护两个 `marketplace.json` 文件——分别面向 Claude Code 和 Codex 的插件市场：

```
# 来源：.github/scripts/build-plugins.pydef upsert_claude_marketplace(catalog_specs, curated_names):    """更新 .claude-plugin/marketplace.json"""    for name in sorted(catalog_specs):        spec = catalog_specs[name]        ifnot is_marketplace_enabled(spec, "claude"):            continue        new_plugins.append({            "name": name,            "source": f"./plugins/{name}",            "description": spec["description"],        })def upsert_agents_marketplace(catalog_specs, curated_names):    """更新 .agents/plugins/marketplace.json"""    for name in sorted(catalog_specs):        spec = catalog_specs[name]        new_plugins.append({            "name": name,            "source": {"source": "local", "path": f"./plugins/{name}"},            "policy": dict(AGENTS_PLUGIN_POLICY),  # 安装策略与认证策略            "category": spec.get("category", "Developer Tools"),        })
```

注意这里的分发模型差异：Claude 市场只需要简单的名称、源路径和描述；而 Codex 市场额外需要安装策略（`AVAILABLE`）和认证策略（`ON_INSTALL`）。构建系统还支持通过 `marketplace_enabled` 字段控制某个插件是否在特定市场上架——一个技能可以只发布到 Claude 而不发布到 Codex，反之亦然。

此外，`skills.sh.json` 文件定义了面向 Skills.sh 市场的分组配置，把 100+ 个技能按照 12 个领域分类：Agentic AI、Physical AI、Vision AI、Training AI、Decision Optimization、GPU Development、Quantum Computing 等。这不仅是为了展示好看，更是让用户在选择技能时有清晰的心智模型。

### 4.3 CI 漂移防护

> 构建脚本还内置了一个漂移检查机制：

```
# 来源：.github/scripts/build-plugins.pyif args.check:    result = subprocess.run(        ["git", "status", "--porcelain",         "plugins/", ".claude-plugin/marketplace.json",         ".agents/plugins/marketplace.json"],        cwd=REPO_ROOT, capture_output=True, text=True,    )    if result.stdout.strip():        die("plugin tree drifted from sources; "            "run .github/scripts/build-plugins.sh and commit the result")
```

这意味着在 CI 中，如果有人手动修改了 `plugins/` 下的生成文件而没有通过 `build-plugins.py` 重新生成，构建会直接失败。**生成文件必须由脚本产出** ——这是防止配置漂移的铁律。

## 五、README 自动重建——配置驱动文档

### 5.1 从 YAML 到 Markdown 表格的流水线

> 仓库的 README 中有两个动态表格——技能目录表和帮助链接表——它们不是手写的，而是由 `regenerate-readme.sh` 脚本从 `components.d/*.yml` 自动生成的。

这个脚本的核心逻辑是：

1. 聚合所有 YAML 配置到临时文件
2. 遍历每个组件，统计其 `skills/` 目录下有多少个合法的 `SKILL.md`
3. **只有技能数量大于 0 的组件才会被渲染到 README 中**
4. 支持同名组件的跨来源合并（比如 Physical AI 产品的技能分布在同步源和手动暂存两个来源）

```
# 来源：.github/scripts/regenerate-readme.sh# 只渲染有合法技能的组件component_skill_count() {local idx=$1local total=0whileread -r catalog_dir; do    if [ -d "skills/$catalog_dir" ]; then      cnt=$(find "skills/$catalog_dir" -name SKILL.md -type f | wc -l)      total=$((total + cnt))    fidone < <(yq -r ".components[$idx].skills[].catalog_dir""$CONFIG")echo"$total"}
```

最终，脚本使用 `awk` 进行同名组件的行级聚合和去重，然后用标记位（`<!-- skills-table-start -->` / `<!-- skills-table-end -->`）替换 README 中的相应区域。整个过程是**幂等的** ——运行多少次，结果都一样。

### 5.2 临时例外的优雅处理

代码中有一个标注为 `TEMPORARY — remove after Computex 2026` 的手动组件配置路径。这是为 Physical AI 产品做的临时例外——它们的上游仓库尚未公开，所以技能被手动暂存在 `.github/scripts/manual-components.yml` 中。脚本对手动组件的链接全部渲染为破折号（"—"），但技能列表照常展示。

这种"临时特例 + 清晰注释 + 计划移除日期"的处理方式，比直接在代码里硬编码特殊逻辑高明得多。它让每个读代码的人都知道这段代码为什么存在、何时应该被移除。

## 六、安全扫描——技能的"安检门"

> 在签名之前，每个技能还需要过一道安全扫描。NVIDIA 开发了一个叫 SkillSpector 的工具（开源在 `NVIDIA/SkillSpector`），覆盖 **64 种漏洞模式，横跨 16 个类别**：

- Prompt 注入
- 数据外泄
- 权限提升
- 供应链风险
- 过度授权
- 工具滥用
- 流氓智能体行为
- 记忆投毒
- ……

SkillSpector 支持两种分析模式：**静态分析** （快速确定性检查，捕获可疑字符串和 API 调用）和**语义分析** （使用 LLM 对比技能声称的功能与实际代码行为是否一致）。

::: block-1
这一层的意义在于：AI 智能体技能的安全威胁与传统软件完全不同。一个看起来无害的 Markdown 文件，可能通过隐藏指令让智能体执行危险操作。SkillSpector 正是为这种新型威胁模型设计的。
:::

## 七、全景回望——一个工程范式的启示

> 退一步看，NVIDIA/skills 仓库展示了一个完整的**AI 智能体技能治理范式**：

1. **去中心化编写 → 中心化分发**：技能在 30+ 个产品仓库中由领域专家编写，通过自动化管线汇聚到统一目录。
2. **合规即准入**：缺少签名、技能卡或评估数据集的技能，自动被排除在分发之外。
3. **密码学可验证**：OMS 签名让每个消费者都能独立验证技能内容的真实性和完整性。
4. **多渠道自适应分发**：同一份技能源码，通过不同的物化策略和市场配置，适配 Claude、Codex、Cursor、Kiro 等多个智能体框架。
5. **配置驱动一切**：README 表格、插件包、市场元数据——全部由 YAML 配置自动生成，人工维护的只有配置文件本身。

::: block-1
这不仅仅是"NVIDIA 的一个开源仓库"。它更像是 AI 智能体时代的 **npm + GPG 签名 + 包审计** 的雏形。随着 AI 智能体越来越深入地参与软件开发，"智能体该知道什么"以及"如何确保它知道的是正确的"，将成为整个行业必须回答的问题。
:::

NVIDIA 用这个仓库率先给出了一个工程上可行的答案：**把知识打包成标准化、可签名、可验证、可分发的技能单元**。这或许就是 AI 智能体供应链安全的起点。

参考资料

[1] NVIDIA Verified Agent Skills: Capability Governance for AI Agents: *https://developer.nvidia.com/blog/nvidia-verified-agent-skills-provide-capability-governance-for-ai-agents/*

[2] `Advanced installation`: *https://github.com/NVIDIA/skills/blob/main/docs/advanced-install.mdx*

[3] Agent Skills Specification: *https://agentskills.io/specification*

::: related-title
**相关推荐**
:::

::: related-links
- **[cann-learning-hub：八章教程、二十五篇博客、两大SKILL，撑起昇腾 NPU 开发者的完整知识图谱](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447902138&idx=1&sn=878d817c8462758c2c6e188739ad57c3&scene=21#wechat_redirect)**
- **[把 GPU 专家的毕生经验喂给 AI！tilelang-cuda-skills 项目：一套让编码智能体自主编写和调试 CUDA 算子的技能体系](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901304&idx=1&sn=91e9db3ffc8b5737ec98466b89a675ea&scene=21#wechat_redirect)**
- **[技能即记忆：AutoSkill 让 LLM 智能体从交互经验中自我进化，实现终身学习，代码开源！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447898295&idx=1&sn=6bf15a2f47d940cea0165f5d1f50aa61&scene=21#wechat_redirect)**

:::
