import { visit } from 'unist-util-visit'

const supportedBlocks = new Set(['block-1', 'image-caption', 'related-title', 'related-links'])
const supportedInline = new Set(['mdnice-highlight', 'mdnice-bold-italic'])

/** mdnice permits a space after :::; remark-directive expects :::block-name. */
export function normalizeMdniceContainers(markdown: string) {
  return markdown.replace(/^:::\s+(block-[1-5]|image-caption|related-title|related-links)\s*$/gm, ':::$1')
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

/** Render mdnice-style ::: containers while preserving nested Markdown. */
export default function remarkMdniceBlocks() {
  return (tree: any, file: any) => {
    const src = String(file)
    restoreRawText(tree, src)

    visit(tree, 'textDirective', (node: any) => {
      if (!supportedInline.has(node.name)) return
      node.data = {
        ...node.data,
        hName: 'span',
        hProperties: { className: [node.name] },
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
