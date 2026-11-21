import { visit } from 'unist-util-visit'

const supportedBlocks = new Set(['block-1', 'image-caption', 'related-title', 'related-links'])
const supportedInline = new Set(['mdnice-highlight', 'mdnice-bold-italic'])

/** mdnice permits a space after :::; remark-directive expects :::block-name. */
export function normalizeMdniceContainers(markdown: string) {
  const names = 'block-[1-5]|image-caption|related-title|related-links'
  const md = markdown
    .replace(/\r\n?/g, '\n')
    .replace(new RegExp(`^:::\\s+(${names})\\s*$`, 'gm'), ':::$1')
    .replace(new RegExp(`^:::\\s+(${names})(\\S[^\\n]*)$`, 'gm'), ':::$1\n$2')
    // 容器开头前缺空行（紧贴普通段落/列表行），导致整段被当成普通文本
    .replace(/(^|\n)((?![>\s]|```)[^\n]*\S)\n(:::\s?(?:block-[1-5]|image-caption|related-title|related-links)(?:\s|$))/gm, '$1$2\n\n$3')
    // block01 / block1 这类名称变体统一成 block-1
    .replace(/^:::\s*block[-]?0*([1-9])(?!\d)/gm, ':::block-$1')
    .replace(/^:{4,}\s*$/gm, ':::')
  return fixClosers(md)
}

/**
 * 行级闭合 fence 修复：
 * 1. 行尾粘连的 ` :::`，若下一非空行已是独立 `:::`，视为重复闭合直接删除；否则拆成独立一行。
 * 2. 独立 `:::` 紧跟在正文/列表行后面时补一个空行（否则会被并成同一段/列表项）。
 * 3. 用栈配对 :::name 开头与 ::: 闭合，删除无匹配开头多余的孤立闭合行。
 */
function fixClosers(md: string) {
  const lines = md.split('\n')
  const out: string[] = []
  const isFence = (s: string) => /^:::\s*$/.test(s)
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i]
    const glued = ln.match(/^(?!\s|>|```|(?: {4}|\t))([^\n]*?\S)[ \t]+:{3,}$/)
    if (glued && !/^:::/.test(ln)) {
      let j = i + 1
      while (j < lines.length && !lines[j].trim()) j++
      if (j < lines.length && isFence(lines[j])) {
        out.push(glued[1]) // 重复闭合，丢弃 :::
      } else {
        out.push(glued[1], ':::')
      }
      continue
    }
    out.push(ln)
  }
  // 独立闭合行前补空行
  const out2: string[] = []
  for (const ln of out) {
    const prev = out2.length ? out2[out2.length - 1] : ''
    if (isFence(ln) && prev.trim() && !isFence(prev) && !prev.endsWith('```')) {
      out2.push('')
    }
    out2.push(ln)
  }
  // 栈配对，删除孤立闭合行；嵌套容器扁平化（remark-directive 不支持嵌套）
  const nameRe = /^:::(block-[1-5]|image-caption|related-title|related-links)\s*$/
  const stack: Array<{ ghost: boolean }> = []
  const out3: string[] = []
  for (const ln of out2) {
    const m = ln.match(nameRe)
    if (m) {
      if (stack.length) {
        stack.push({ ghost: true }) // 内层容器：去掉围栏，内容保留在外层里
      } else {
        stack.push({ ghost: false })
        out3.push(ln)
      }
    } else if (isFence(ln)) {
      if (stack.length) {
        const top = stack.pop()!
        if (!top.ghost) out3.push(ln)
      }
      // 孤立闭合行直接丢弃
    } else {
      out3.push(ln)
    }
  }
  return out3.join('\n')
}

/**
 * remark-directive treats any ":xxx" in plain text (e.g. "16:30", "1:1的CTA")
 * as a directive and drops its content. Revert every directive outside the
 * mdnice whitelist back to the raw source text.
 */
function restoreRawText(tree: any, src: string) {
  const replace = (node: any, index: number | undefined, parent: any, blockLevel: boolean) => {
    const pos = node.position
    if (!pos || !parent || typeof index !== 'number') return
    const raw = src.slice(pos.start.offset, pos.end.offset)
    parent.children[index] = blockLevel
      ? { type: 'paragraph', children: [{ type: 'text', value: raw.replace(/\n+/g, ' ') }] }
      : { type: 'text', value: raw }
  }
  visit(tree, 'textDirective', (node: any, index: any, parent: any) => {
    if (!supportedInline.has(node.name)) replace(node, index, parent, false)
  })
  visit(tree, ['leafDirective', 'containerDirective'], (node: any, index: any, parent: any) => {
    if (!supportedBlocks.has(node.name)) replace(node, index, parent, true)
  })
}

/** 合并段内相邻 text 节点，让跨节点残留的 ** 能配对处理。 */
function mergeAdjacentText(tree: any) {
  visit(tree, (node: any) => {
    const kids = node.children
    if (!kids) return
    const merged: any[] = []
    for (const k of kids) {
      const last = merged[merged.length - 1]
      if (last && last.type === 'text' && k.type === 'text') {
        last.value += k.value
      } else {
        merged.push(k)
      }
    }
    node.children = merged
  })
}

/**
 * GFM 自动链接会把 URL 后面的中文、** 一起吞进 link 节点（URL 里不可能
 * 含中文或 *）。截断 URL，把尾巴还原成普通文本。
 */
function fixBrokenAutolinks(tree: any) {
  const fixes: Array<{ parent: any; index: number; head: string; tail: string }> = []
  visit(tree, 'link', (node: any, index: any, parent: any) => {
    if (!parent || typeof index !== 'number' || !/^https?:\/\//.test(node.url)) return
    const cut = node.url.search(/[^A-Za-z0-9\-._~:/?#[\]@!$&'()+,;=%]/)
    if (cut < 0) return
    fixes.push({ parent, index, head: node.url.slice(0, cut), tail: node.url.slice(cut) })
  })
  for (const { parent, index, head, tail } of fixes.reverse()) {
    parent.children.splice(index, 1,
      { type: 'link', url: head, children: [{ type: 'text', value: head }] },
      { type: 'text', value: tail },
    )
  }
}

/**
 * CommonMark 的强调定界符规则在 `**加粗**中文`（闭标记后紧跟 CJK）或
 * `中文**加粗**`（开标记前是 CJK）场景会解析失败，残留字面 `*`。
 * 对 mdast 文本节点里残留的配对标记做二次切分。
 */
function fixLeftoverEmphasis(tree: any) {
  fixBrokenAutolinks(tree)
  mergeAdjacentText(tree)
  const replacements: Array<{ parent: any; index: number; nodes: any[] }> = []
  visit(tree, 'text', (node: any, index: any, parent: any) => {
    if (!parent || typeof index !== 'number' || !node.value.includes('*')) return
    const re = /\*\*\*\*([^*]+?)\*\*\*\*|\*\*([^*\n]+?)\*\*|\*([^*\s][^*\n]*?)\*/g
    const nodes: any[] = []
    let last = 0
    let m: RegExpExecArray | null
    let matched = false
    while ((m = re.exec(node.value))) {
      matched = true
      if (m.index > last) nodes.push({ type: 'text', value: node.value.slice(last, m.index) })
      const content = m[1] ?? m[2] ?? m[3]
      // 单个 * 包围纯数字/运算符的内容视为乘法表达式，跳过
      if (!m[1] && !m[2] && /^[\d\s.+\-*/=()^]+$/.test(content)) {
        last = m.index + m[0].length
        continue
      }
      const inner = splitLinks(content)
      if (m[1]) nodes.push({ type: 'strong', children: [{ type: 'emphasis', children: inner }] })
      else if (m[2]) nodes.push({ type: 'strong', children: inner })
      else nodes.push({ type: 'emphasis', children: inner })
      last = m.index + m[0].length
    }
    if (!matched) return
    if (last < node.value.length) nodes.push({ type: 'text', value: node.value.slice(last) })
    replacements.push({ parent, index, nodes })
  })
  for (const { parent, index, nodes } of replacements.reverse()) {
    parent.children.splice(index, 1, ...nodes)
  }
  // 收尾：配对后仍残留的字面 **** / ** 属于未配对标记（导出损坏），去掉
  visit(tree, 'text', (node: any) => {
    if (node.value.includes('**')) {
      node.value = node.value.replace(/\*\*\*\*/g, '').replace(/\*\*/g, '')
    }
  })
}

/** 把内容字符串里的 [text](url) 切成 link 节点与 text 节点的组合。 */
function splitLinks(content: string): any[] {
  const re = /\[([^\]]+)\]\(([^)\s]+)\)/g
  const out: any[] = []
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(content))) {
    if (m.index > last) out.push({ type: 'text', value: content.slice(last, m.index) })
    out.push({ type: 'link', url: m[2], children: [{ type: 'text', value: m[1] }] })
    last = m.index + m[0].length
  }
  if (last < content.length) out.push({ type: 'text', value: content.slice(last) })
  if (!out.length) out.push({ type: 'text', value: content })
  return out
}

/**
 * 公式吞段防护：原始 md 里 $$ 不成对时，remark-math 会把整段正文吞进
 * 数学节点。这类节点的源码里会混入中文或 ** 标记。把异常节点拆回正文，
 * 用原文（带定界符）重新解析，让内部的小公式和加粗恢复正常。
 */
export function remarkUnswallowMath(this: any) {
  const proc: any = this
  const isSwallowed = (value: string) =>
    /[。，、；：？！]/.test(value) ||
    value.includes('**') ||
    value.includes('\n\n') ||
    value.length > 600
  return function (tree: any) {
    for (let pass = 0; pass < 5; pass++) {
      const fixes: Array<{ parent: any; index: number; raw: string; block: boolean }> = []
      visit(tree, ['math', 'inlineMath'], (node: any, index: any, parent: any) => {
        if (!parent || typeof index !== 'number' || !isSwallowed(node.value)) return
        fixes.push({ parent, index, block: node.type === 'math', raw: node.value.trim() })
      })
      if (!fixes.length) break
      for (const { parent, index, raw, block } of fixes.reverse()) {
        let sub: any
        try { sub = proc.parse(raw) } catch { continue }
        let kids = sub.children
        if (block && kids.length === 1 && kids[0].type === 'paragraph') kids = kids[0].children
        const nodes = !block
          ? kids
          : sub.children.length > 1
            ? sub.children
            : [{ type: 'paragraph', children: kids }]
        parent.children.splice(index, 1, ...nodes)
        for (const n of nodes) fixLeftoverEmphasis(n)
      }
    }
  }
}

/** Render mdnice-style ::: containers while preserving nested Markdown. */
export default function remarkMdniceBlocks() {
  return (tree: any, file: any) => {
    const src = String(file)
    restoreRawText(tree, src)
    fixLeftoverEmphasis(tree)

    visit(tree, 'textDirective', (node: any) => {
      if (!supportedInline.has(node.name)) return
      node.data = {
        ...node.data,
        hName: 'span',
        hProperties: { className: [node.name] },
      }
    })
    visit(tree, 'paragraph', (node: any, _index: any, parent: any) => {
      const first = node.children?.[0]
      if (first?.type === 'text' && /^\s*交流加群请在/.test(first.value)) {
        node.data = { ...node.data, hProperties: { className: ['join-hint'] } }
        return
      }
      const text = (node.children || []).map((c: any) => c.value ?? '').join('').trim()
      if (parent?.type !== 'containerDirective' && /^(端侧相关|相关|更多|不相关)推荐(见文末)?$/.test(text)) {
        node.data = { ...node.data, hName: 'div', hProperties: { className: ['related-title'] } }
      }
    })
    visit(tree, 'containerDirective', (node: any) => {
      if (!supportedBlocks.has(node.name)) return
      const children = node.children || []
      node.data = {
        ...node.data,
        hName: node.name === 'image-caption' ? 'div' : 'section',
        hProperties: { className: [node.name] },
      }
      if (node.name === 'image-caption') {
        node.children = children.map((child: any) => child.type === 'paragraph'
          ? { ...child, data: { ...child.data, hName: 'div' } }
          : child)
      }
      if (node.name === 'block-1') {
        node.children = [{
          type: 'div',
          children,
          data: { hName: 'div', hProperties: { className: ['block-1-inner'] } },
        }]
      }
    })
  }
}
