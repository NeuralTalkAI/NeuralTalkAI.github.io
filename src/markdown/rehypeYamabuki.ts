import { visit } from 'unist-util-visit'

const element = (tagName: string, properties: Record<string, unknown>, children: any[] = []) => ({
  type: 'element', tagName, properties, children,
})

/** Match the HTML contract used by online-markdown's mdnice renderer. */
export default function rehypeYamabuki() {
  return (tree: any) => {
    visit(tree, 'element', (node: any, index: number | undefined, parent: any) => {
      if (/^h[1-6]$/.test(node.tagName)) {
        node.children = [
          element('span', { className: ['prefix'] }),
          element('span', { className: ['content'] }, node.children),
          element('span', { className: ['suffix'] }),
        ]
      }
      if (node.tagName === 'blockquote') {
        let depth = 1
        let ancestor = parent
        while (ancestor) {
          if (ancestor.tagName === 'blockquote') depth += 1
          ancestor = ancestor.parent
        }
        node.properties = { ...node.properties, className: [`multiquote-${depth}`] }
      }
      if (node.tagName === 'li' && !node.children.some((child: any) => child.tagName === 'section')) {
        node.children = [element('section', {}, node.children)]
      }
      if (node.tagName === 'table' && parent && index !== undefined) {
        parent.children[index] = element('div', { className: ['table-container'] }, [node])
      }
      if (node.properties?.className?.some((name: string) => name === 'katex' || name === 'katex-display')) {
        const classes = node.properties.className
        if (classes.includes('katex-display')) {
          if (parent && index !== undefined) parent.children[index] = element('section', { className: ['block-equation'] }, [node])
        } else {
          node.properties.className = [...classes, 'inline-equation']
        }
      }
    })
  }
}
