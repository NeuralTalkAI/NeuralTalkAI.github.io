import fs from 'node:fs'
import path from 'node:path'

const data = fs.readFileSync('src/data/articles-data.ts', 'utf8')
const re = /slug:\s*'([^']+)'[\s\S]{0,600}?title:\s*'((?:[^'\\]|\\.)*)'/g
const titles = {}
let m
while ((m = re.exec(data))) titles[m[1]] = m[2].replace(/\\'/g, "'")

const dir = 'src/content/articles'
const files = fs.readdirSync(dir).filter(f => f.endsWith('.md'))
const rows = []
for (const f of files) {
  const slug = f.replace(/\.md$/, '')
  const text = fs.readFileSync(path.join(dir, f), 'utf8')
  const cjk = (text.match(/[一-鿿]/g) || []).length
  const hasToc = /本文目录/.test(text)
  const hasRelated = /相关推荐|相关阅读/.test(text)
  // 目录之后、相关推荐之前的正文
  let afterToc = text
  if (hasToc) {
    const i = text.indexOf('本文目录')
    afterToc = text.slice(i)
    const lines = afterToc.split('\n')
    const kept = []
    let skipping = true
    for (const line of lines) {
      if (skipping && (/^\s*(?:[-*+]\s+|\d+\.\s+)/.test(line) || line.trim() === '')) continue
      skipping = false
      kept.push(line)
    }
    afterToc = kept.join('\n')
  }
  let body = afterToc.replace(/::: related-(?:title|links)[\s\S]*?(?=\n:::\s*$)/g, '')
  body = body.replace(/相关推荐[\s\S]*$/m, '')
  const bodyCjk = (body.match(/[一-鿿]/g) || []).length
  rows.push({ slug, title: titles[slug] || slug, cjk, bodyCjk, hasToc, hasRelated })
}
rows.sort((a, b) => a.bodyCjk - b.bodyCjk)
fs.writeFileSync('../short-articles.json', JSON.stringify(rows, null, 1))
const short = rows.filter(r => r.bodyCjk < 400)
console.log('total:', rows.length, '| bodyCjk<400:', short.length)
for (const r of short) console.log(`${r.bodyCjk}\t${r.title.slice(0, 40)}\t${r.slug}`)
