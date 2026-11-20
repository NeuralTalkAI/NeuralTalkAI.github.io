import fs from 'node:fs'
import path from 'node:path'

const dir = 'src/content/articles'
const files = fs.readdirSync(dir).filter(f => f.endsWith('.md'))
const rows = []
for (const f of files) {
  const text = fs.readFileSync(path.join(dir, f), 'utf8')
  const cjk = (text.match(/[一-鿿]/g) || []).length
  // 去掉目录行后的正文字数
  const body = text.replace(/^#{0,6}\s*本文目录[^\n]*\n([\s\S]*?)$/m, (m, rest) => {
    const lines = rest.split('\n')
    const kept = []
    let started = false
    for (const line of lines) {
      if (/^\s*(?:[-*+]\s+|\d+\.\s+)/.test(line)) { started = true; continue }
      if (started && line.trim() === '') continue
      if (started) { started = false }
      kept.push(line)
    }
    return kept.join('\n')
  })
  const bodyCjk = (body.match(/[一-鿿]/g) || []).length
  const title = (text.match(/^#\s+(.+)/m) || [])[1] || f
  rows.push({ file: f, title: title.slice(0, 50), cjk, bodyCjk })
}
rows.sort((a, b) => a.bodyCjk - b.bodyCjk)
console.log('total:', rows.length)
console.log('\n== 正文中文字数最少的 40 篇 ==')
for (const r of rows.slice(0, 40)) console.log(`${r.bodyCjk}\t全文${r.cjk}\t${r.title}\t${r.file}`)
fs.writeFileSync('../article-char-counts.json', JSON.stringify(rows, null, 1))
