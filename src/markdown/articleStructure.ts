import { visit } from 'unist-util-visit'

export type TocItem = {
  id: string
  text: string
  depth: 2 | 3
}

/** Use the same stable heading IDs in the rendered article and its TOC. */
export function getArticleToc(markdown: string): TocItem[] {
  const items: TocItem[] = []
  let headingIndex = 0
  let inFence = false
  for (const line of markdown.split(/\r?\n/)) {
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    const match = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line)
    if (!match) continue
    const depth = match[1].length
    const text = match[2].replace(/[*_`]/g, '').trim()
    const id = `article-heading-${headingIndex++}`
    if ((depth === 2 || depth === 3) && text !== '本文目录') {
      items.push({ id, text, depth })
    }
  }
  return items
}

/** 图片统一用站点根路径（/articles/...、/images/...）。
 *  SEO 路由改造后文章页是真实多级路径（/post/<slug>/），相对路径会基于文章目录解析导致 404，
 *  必须保持根路径；站点部署在域名根，无需考虑子路径。 */
export function normalizeArticleImageUrls(markdown: string) {
  return markdown
    .replace(/\]\(\.\.\/\.\.\/\.\.\/public\//g, '](/')
}

/** The exported static list is replaced by the complete, linked heading tree. */
export function stripManualTocList(markdown: string) {
  const lines = markdown.split(/\r?\n/)
  const tocLine = /^\s*#{0,6}\s*本文目录\s*[：:]?\s*$/
  const tocEntry = /^\s*(?:[-*+]\s+|\d+\.\s+|\d+\.\d+\s+)/
  const out: string[] = []
  let i = 0
  let stripped = false
  while (i < lines.length) {
    if (!stripped && tocLine.test(lines[i])) {
      const isHeading = lines[i].includes('#')
      // 跳过目录条目行（条目间允许空行）
      let j = i + 1
      let removedAny = false
      while (j < lines.length) {
        if (lines[j].trim() === '') { j++; continue }
        if (tocEntry.test(lines[j])) { removedAny = true; j++; continue }
        break
      }
      // 至少删掉一条目录条目才认为是手动目录，否则保留原文
      if (removedAny) {
        stripped = true
        if (isHeading) {
          // 保留「本文目录」标题作为锚点，后面接自动生成的可点击目录
          out.push(lines[i])
          i = j
        } else {
          // 纯文本形式的目录整体移除
          i = j
          if (i < lines.length && lines[i].trim() === '') i++
        }
        continue
      }
    }
    out.push(lines[i])
    i++
  }
  return out.join('\n')
}

export function remarkArticleHeadingIds() {
  return (tree: any) => {
    let headingIndex = 0
    visit(tree, 'heading', (node: any) => {
      const id = `article-heading-${headingIndex++}`
      node.data = {
        ...node.data,
        hProperties: { ...node.data?.hProperties, id },
      }
    })
  }
}

