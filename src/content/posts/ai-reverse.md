# 一级标题：文章中的主要章节

当大语言模型参与代码分析时，分析结果不仅取决于代码本身，也会受到上下文中文字信息的影响。理解这种影响，有助于我们更稳妥地设计自动化分析流程。这里展示**加粗文字**、*斜体文字*，以及行内代码 `const verified = true` 的实际效果。

## 二级标题：从一次分析任务开始

这是一段正文排版示例。参考页面采用 16px 字号和 1.8 倍行距，段落之间保留适当空白，中文与 English、数字 123 混排时也能保持清晰的阅读节奏。

> 这是一段引用。引用区域使用浅橙色背景和 3px 橙色左边线，可以用来放置补充说明、论文摘要或需要特别注意的内容。

### 三级标题：需要关注的三个环节

无序列表的显示效果：

- **输入是否可靠**：检查代码、注释与外部资料的来源。
- **过程是否可复核**：保留分析步骤和关键证据。
- **结论是否可验证**：通过独立工具或人工检查进行交叉验证。

有序列表的显示效果：

1. 收集并整理原始材料。
2. 运行自动化分析程序。
3. 复核结果并归档证据。

## 代码块

下面是一段 Python 代码，长代码块可以在窄屏中横向滚动：

```python
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    source: str
    verified: bool
    confidence: float

result = AnalysisResult(
    source="wechat-article",
    verified=True,
    confidence=0.98,
)

print(f"verification: {result.verified}")
```

## 数学公式

行内数学公式使用单个美元符号，例如注意力分数 $A = \operatorname{softmax}(QK^T / \sqrt{d_k})$，以及质能方程 $E = mc^2$。

多行数学公式使用两个美元符号：

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

也支持多行对齐与矩阵：

$$
\begin{aligned}
L(\theta) &= -\sum_{i=1}^{N} y_i \log \hat{y}_i \\
\hat{y} &= \operatorname{softmax}(Wx + b)
\end{aligned}
$$

$$
\mathbf{M} =
\begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
$$

## 表格与图片

| 环节 | 建议做法 |
| --- | --- |
| 收集材料 | 记录来源与获取时间 |
| 自动分析 | 保存原始输出 |
| 人工复核 | 核对关键结论 |
| 归档分享 | 附上证据与适用范围 |

![分析流程示意图](/images/workflow.svg)

### 三级标题：写在最后

知识库的价值在于让内容可以被搜索、引用和持续更新。每篇文章保留原始链接，也方便回到公众号查看最初版本。
