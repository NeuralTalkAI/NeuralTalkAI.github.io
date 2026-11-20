import fs from 'node:fs'
import path from 'node:path'

const data = fs.readFileSync('src/data/articles-data.ts', 'utf8')
const re = /slug:\s*"([^"]+)"[\s\S]{0,400}?title:\s*"((?:[^"\\]|\\.)*)"/g
const titles = {}
let m
while ((m = re.exec(data))) titles[m[1]] = m[2].replace(/\\"/g, '"')

const dir = 'src/content/articles'
const files = fs.readdirSync(dir).filter(f => f.endsWith('.md'))
const rows = []
for (const f of files) {
  const slug = f.replace(/\.md$/, '')
  const text = fs.readFileSync(path.join(dir, f), 'utf8')
  const cjk = (text.match(/[一-鿿]/g) || []).length
  const hasToc = /本文目录/.test(text)
  const tail = text.slice(-200)
  rows.push({ slug, title: titles[slug] || slug, cjk, hasToc, tail })
}
rows.sort((a, b) => a.cjk - b.cjk)
fs.writeFileSync('../article-stats.json', JSON.stringify(rows, null, 1))
const short = rows.filter(r => r.cjk < 500)
console.log('total:', rows.length, '| 全文CJK<500:', short.length)
